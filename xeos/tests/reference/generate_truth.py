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

Reference -> backend validated:
    MITgcm ini_eos.F + find_rho.F      -> jmd95, unesco, mdjwf (gfortran)
    MOM6 MOM_EOS_Wright_{red,full}.F90 -> wright97-reduced, wright97-full (gfortran)
    MOM6 MOM_EOS_UNESCO.F90            -> jmd95@mom6 (the MOM6 'UNESCO' = JMD95 fit)
    MOM6 MOM_EOS_Roquet_SpV.F90        -> roquet-spv (gfortran)
    SeawaterPolynomials.jl (Julia)     -> the six idealized roquet-* forms
    polyTEOS10.py                      -> teos10-poly55      (downloaded if absent)
    gsw                                -> teos10

Model-source generators (compile MOM6 Fortran, run Oceananigans' Julia package)
live in ``_build_*.py`` next to this script; they need gfortran and/or julia but
emit only numbers, so the committed ``truth.json`` stays toolchain-free to consume.

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

    _ensure_polyteos()
    import sys
    sys.path.insert(0, HERE)
    import polyTEOS10
    # Model-source generators (compile MOM6/MITgcm Fortran / run Oceananigans Julia).
    from _build_wright_fortran import wright_truth
    from _build_unesco_fortran import unesco_mom6_truth
    from _build_mitgcm_fortran import mitgcm_truth
    from _build_seawaterpolynomials_julia import (
        seawaterpolynomials_truth, julia_version)

    tt, ss, pp = (np.array(v, dtype=float) for v in np.meshgrid(
        T_VALS, S_VALS, P_VALS, indexing="ij"))
    t = tt.ravel()
    s = ss.ravel()
    p = pp.ravel()

    cases = {}

    # jmd95 / unesco / mdjwf: MITgcm's own Fortran -- coefficients parsed verbatim
    # from model/src/ini_eos.F, density formulas from model/src/find_rho.F, compiled
    # with gfortran (replaces the fastjmd95 / MITgcmutils Python ports). alpha/beta
    # by centred FD of the compiled density (matching xeos's FD-fallback path).
    mitgcm = mitgcm_truth(s, t, p)
    if mitgcm is not None:
        cases.update(mitgcm)
    else:
        print("WARNING: gfortran not found; jmd95/unesco/mdjwf truth not regenerated.")

    # wright97-reduced / wright97-full: MOM6 Fortran (MOM_EOS_Wright_{red,full}.F90),
    # compiled with gfortran. This replaces the momlevel Python port (which only
    # implements the reduced-range coefficients) and gives wright97-full its first
    # numeric truth. rho + analytic alpha/beta straight from the model source.
    for variant, wright_id in (("red", "wright97-reduced"), ("full", "wright97-full")):
        wt = wright_truth(variant, s, t, p)
        if wt is not None:
            cases[wright_id] = wt
        else:
            print(f"WARNING: gfortran not found; {wright_id} truth not regenerated.")

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

    # Idealized second-order Roquet forms: evaluated by SeawaterPolynomials.jl
    # (the Julia package Oceananigans uses). First numeric truth for these six
    # backends, which were previously validated structurally only.
    swp = seawaterpolynomials_truth(s, t, p)
    if swp is not None:
        cases.update(swp)
    else:
        print("WARNING: julia not found; idealized Roquet truth not regenerated.")

    # MOM6 'UNESCO'/'JACKETT_MCD' is, despite the name, the Jackett & McDougall
    # (1995) fit -- i.e. the jmd95 kernel. Validate jmd95 against MOM6 Fortran too,
    # a second model source besides MITgcm's own (above). The "backend" field tells
    # test_backends.py which kernel this case validates.
    unesco_mom6 = unesco_mom6_truth(s, t, p)
    if unesco_mom6 is not None:
        cases["jmd95@mom6"] = {"backend": "jmd95", **unesco_mom6}

    out = {
        "_README": "Frozen reference values; regenerate with generate_truth.py. "
                   "Inputs t,s,p fed identically to xeos and to each reference.",
        "provenance": {
            "reference_versions": {
                "gsw": _ver("gsw"),
                "numpy": _ver("numpy"),
                "polyTEOS10": POLYTEOS_URL,
                "gfortran": gfortran_version(),
                "julia": julia_version(),
                # Model source compiled/run on demand (LGPL/MIT; not committed):
                "ini_eos.F + find_rho.F (jmd95, unesco, mdjwf)":
                    "github.com/MITgcm/MITgcm master",
                "MOM_EOS_Wright_red.F90 / _full.F90 (wright97-*)":
                    "github.com/mom-ocean/MOM6 main",
                "MOM_EOS_UNESCO.F90 (jmd95@mom6)":
                    "github.com/mom-ocean/MOM6 main",
                "MOM_EOS_Roquet_SpV.F90 (roquet-spv)":
                    "github.com/mom-ocean/MOM6 main",
                "SeawaterPolynomials.jl (roquet-* idealized)":
                    "github.com/CliMA/SeawaterPolynomials.jl",
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
