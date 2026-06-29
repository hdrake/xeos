"""Generate frozen "truth" fixtures for cross-validating xeos's vendored kernels.

Run this in the *reference* environment (see ``environment.yml`` in this folder),
NOT in xeos's own (lightweight) runtime environment.  It imports the authoritative
reference packages, evaluates density / alpha / beta on a fixed (t, s, p) grid,
and writes ``truth.json`` (committed) with full provenance.  The regular test
suite (``test_backends.py``) then validates against that committed file and does
NOT need these heavy reference packages installed.

Usage
-----
    python generate_truth.py            # writes ./truth.json

Reference packages -> backend validated:
    fastjmd95               -> jmd95
    momlevel.eos.wright     -> wright97-reduced   (same functional form as -full)
    polyTEOS10.py           -> teos10-poly55      (downloaded if absent)
    gsw                     -> teos10
    MITgcmutils.density     -> unesco, mdjwf

The idealized second-order Roquet forms (roquet-linear, roquet-cabbeling, ...)
have no standalone Python reference and are validated structurally in test_api.py
(exact analytic derivatives vs finite differences; literal check values).

The numbers fed to xeos and to each reference are identical; xeos performs no
silent temperature/salinity conversion, so this is an apples-to-apples check.
The conservative-temperature backends (teos10*, poly55) are simply fed the grid
values as (CT, SA); the potential-temperature backends as (theta, Sp).
"""

import json
import os
import urllib.request
from importlib.metadata import version, PackageNotFoundError

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
POLYTEOS_URL = (
    "https://raw.githubusercontent.com/fabien-roquet/polyTEOS/master/polyTEOS10.py"
)

# Fixed validation grid (broadcast to all combinations).
T_VALS = [-2.0, 5.0, 15.0, 30.0]
S_VALS = [30.0, 35.0, 38.0]
P_VALS = [0.0, 1000.0, 4000.0]  # dbar


def _ver(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "unknown"


def _ensure_polyteos():
    path = os.path.join(HERE, "polyTEOS10.py")
    if not os.path.exists(path):
        urllib.request.urlretrieve(POLYTEOS_URL, path)
    return path


def main():
    import gsw
    import fastjmd95
    from momlevel.eos import wright
    from MITgcmutils import density as mitgcm_density

    _ensure_polyteos()
    import sys
    sys.path.insert(0, HERE)
    import polyTEOS10

    tt, ss, pp = (np.array(v, dtype=float) for v in np.meshgrid(
        T_VALS, S_VALS, P_VALS, indexing="ij"))
    t = tt.ravel()
    s = ss.ravel()
    p = pp.ravel()

    cases = {}

    def fd_alpha_beta(densfunc, ss, tt, pp, h=1.0e-3):
        """alpha/beta by centred difference of an independent reference density.

        Uses the same step the facade uses, so xeos's own finite-difference path
        is validated against a *different* density implementation (catches sign /
        unit mistakes in that path). densfunc signature is (salt, theta, p_dbar).
        """
        rho0 = np.asarray(densfunc(ss, tt, pp))
        a = -(np.asarray(densfunc(ss, tt + h, pp))
              - np.asarray(densfunc(ss, tt - h, pp))) / (2.0 * h) / rho0
        b = (np.asarray(densfunc(ss + h, tt, pp))
             - np.asarray(densfunc(ss - h, tt, pp))) / (2.0 * h) / rho0
        return a.tolist(), b.tolist()

    # jmd95 (fastjmd95): rho(s, t, p_dbar)
    jmd95_a, jmd95_b = fd_alpha_beta(fastjmd95.rho, s, t, p)
    cases["jmd95"] = {"rho": fastjmd95.rho(s, t, p).tolist(),
                      "alpha": jmd95_a, "beta": jmd95_b}

    # wright97-reduced (momlevel): density(T, S, p_Pa)
    p_pa = p * 1.0e4
    cases["wright97-reduced"] = {
        "rho": np.asarray(wright.density(t, s, p_pa)).tolist(),
        "alpha": np.asarray(wright.alpha(t, s, p_pa)).tolist(),
        "beta": np.asarray(wright.beta(t, s, p_pa)).tolist(),
    }

    # teos10-poly55 (polyTEOS10_bsq): (SA, CT, p_dbar) -> rho, a=-dr/dCT, b=dr/dSA
    rho_p, a_p, b_p, _, _ = polyTEOS10.polyTEOS10_bsq(s, t, p)
    rho_p = np.asarray(rho_p)
    cases["teos10-poly55"] = {
        "rho": rho_p.tolist(),
        "alpha": (np.asarray(a_p) / rho_p).tolist(),
        "beta": (np.asarray(b_p) / rho_p).tolist(),
    }

    # teos10 (gsw): rho(SA, CT, p_dbar), alpha, beta
    cases["teos10"] = {
        "rho": np.asarray(gsw.rho(s, t, p)).tolist(),
        "alpha": np.asarray(gsw.alpha(s, t, p)).tolist(),
        "beta": np.asarray(gsw.beta(s, t, p)).tolist(),
    }

    # roquet-spv: validated against MOM6's authoritative Fortran (no trustworthy
    # Python reference exists — polyTEOS10_55t has a deltaS typo). Skipped if
    # gfortran is unavailable; the committed truth.json keeps the frozen values.
    from _build_roquet_spv_fortran import roquet_spv_truth, gfortran_version
    spv_truth = roquet_spv_truth(s, t, p)
    if spv_truth is not None:
        cases["roquet-spv"] = spv_truth
    else:
        print("WARNING: gfortran not found; roquet-spv truth not regenerated.")

    # unesco / mdjwf (MITgcmutils.density): density(salt, theta, p_dbar)
    unesco_a, unesco_b = fd_alpha_beta(mitgcm_density.unesco, s, t, p)
    cases["unesco"] = {"rho": np.asarray(mitgcm_density.unesco(s, t, p)).tolist(),
                       "alpha": unesco_a, "beta": unesco_b}
    mdjwf_a, mdjwf_b = fd_alpha_beta(mitgcm_density.mdjwf, s, t, p)
    cases["mdjwf"] = {"rho": np.asarray(mitgcm_density.mdjwf(s, t, p)).tolist(),
                      "alpha": mdjwf_a, "beta": mdjwf_b}

    out = {
        "_README": "Frozen reference values; regenerate with generate_truth.py. "
                   "Inputs t,s,p fed identically to xeos and to each reference.",
        "provenance": {
            "reference_versions": {
                "gsw": _ver("gsw"),
                "fastjmd95": _ver("fastjmd95"),
                "momlevel": _ver("momlevel"),
                "seawater": _ver("seawater"),
                "MITgcmutils": _ver("MITgcmutils"),
                "numpy": _ver("numpy"),
                "polyTEOS10": POLYTEOS_URL,
                "gfortran (roquet-spv)": gfortran_version(),
                "MOM_EOS_Roquet_SpV.F90": "github.com/mom-ocean/MOM6 main",
            },
            "grid": {"T": T_VALS, "S": S_VALS, "P_dbar": P_VALS},
        },
        "inputs": {"t": t.tolist(), "s": s.tolist(), "p_dbar": p.tolist()},
        "cases": cases,
    }
    with open(os.path.join(HERE, "truth.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"Wrote truth.json with {len(t)} points and cases: {sorted(cases)}")


if __name__ == "__main__":
    main()
