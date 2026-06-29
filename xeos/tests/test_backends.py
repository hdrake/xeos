"""Cross-validate xeos's vendored kernels against frozen reference values.

The expected values in ``reference/truth.json`` are produced primarily by compiling
each model's own source (MITgcm / MOM6 / MPAS-Ocean Fortran via gfortran;
Oceananigans' ``SeawaterPolynomials.jl`` via Julia), with ``gsw`` the one remaining
Python reference -- see ``reference/README.md``.  The same numeric ``(t, s, p)``
inputs are fed to xeos here; agreement to ~1e-6 confirms the kernels reproduce their
model sources rather than merely approximating them.
"""

import json
import os

import numpy as np
import pytest

import xeos

_TRUTH_PATH = os.path.join(os.path.dirname(__file__), "reference", "truth.json")
with open(_TRUTH_PATH) as _fh:
    TRUTH = json.load(_fh)

_T = np.array(TRUTH["inputs"]["t"])
_S = np.array(TRUTH["inputs"]["s"])
_P = np.array(TRUTH["inputs"]["p_dbar"])


def _eos(case_key):
    """Build the EOS a truth case validates, skipping if an optional dependency is
    missing.  A case may set ``"backend"`` to validate a kernel under a different
    key (e.g. ``jmd95@mom6`` -> the ``jmd95`` backend), so one backend can be
    checked against more than one model's source."""
    eos_id = TRUTH["cases"][case_key].get("backend", case_key)
    try:
        eos = xeos.equation_of_state(eos_id)
        eos.rho(float(_T[0]), float(_S[0]), float(_P[0]))  # trigger lazy imports
        return eos
    except ImportError as exc:
        pytest.skip(f"optional dependency missing for {eos_id}: {exc}")


@pytest.mark.parametrize("case_key", sorted(TRUTH["cases"]))
def test_density_matches_reference(case_key):
    eos = _eos(case_key)
    expected = np.array(TRUTH["cases"][case_key]["rho"])
    got = np.asarray(eos.rho(_T, _S, _P))
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("case_key", sorted(TRUTH["cases"]))
def test_alpha_beta_match_reference(case_key):
    case = TRUTH["cases"][case_key]
    if "alpha" not in case:
        pytest.skip(f"no alpha/beta reference for {case_key}")
    eos = _eos(case_key)
    np.testing.assert_allclose(np.asarray(eos.alpha(_T, _S, _P)),
                               np.array(case["alpha"]), rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(np.asarray(eos.beta(_T, _S, _P)),
                               np.array(case["beta"]), rtol=1e-5, atol=1e-9)


def test_provenance_present():
    # Guard against an un-stamped/hand-edited fixture.
    versions = TRUTH["provenance"]["reference_versions"]
    assert versions["gsw"] != "unknown"
    # model-source toolchain stamps must be present (a regeneration on a box missing
    # one would stamp null here, flagging that those cases were left stale).
    assert versions["gfortran"]
    assert versions["julia"]
