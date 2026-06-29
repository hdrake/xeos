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
    fastjmd95           -> jmd95
    momlevel.eos.wright -> wright97-reduced   (same functional form as -full)
    polyTEOS10.py       -> teos10-poly55      (downloaded if absent)
    gsw                 -> teos10

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

    # jmd95 (fastjmd95): rho(s, t, p_dbar)
    cases["jmd95"] = {"rho": fastjmd95.rho(s, t, p).tolist()}

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

    out = {
        "_README": "Frozen reference values; regenerate with generate_truth.py. "
                   "Inputs t,s,p fed identically to xeos and to each reference.",
        "provenance": {
            "reference_versions": {
                "gsw": _ver("gsw"),
                "fastjmd95": _ver("fastjmd95"),
                "momlevel": _ver("momlevel"),
                "seawater": _ver("seawater"),
                "numpy": _ver("numpy"),
                "polyTEOS10": POLYTEOS_URL,
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
