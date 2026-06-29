"""Cross-validate xeos's vendored kernels against frozen reference values.

The expected values in ``reference/truth.json`` were produced by the authoritative
reference packages (gsw, fastjmd95, momlevel, polyTEOS10) in a pinned environment
-- see ``reference/README.md``.  The same numeric ``(t, s, p)`` inputs are fed to
xeos here; agreement to ~1e-6 confirms the kernels reproduce their sources rather
than merely approximating them.
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


def _eos(eos_id):
    """Build the EOS, skipping if an optional backend dependency is missing."""
    try:
        eos = xeos.equation_of_state(eos_id)
        eos.rho(float(_T[0]), float(_S[0]), float(_P[0]))  # trigger lazy imports
        return eos
    except ImportError as exc:
        pytest.skip(f"optional dependency missing for {eos_id}: {exc}")


@pytest.mark.parametrize("eos_id", sorted(TRUTH["cases"]))
def test_density_matches_reference(eos_id):
    eos = _eos(eos_id)
    expected = np.array(TRUTH["cases"][eos_id]["rho"])
    got = np.asarray(eos.rho(_T, _S, _P))
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("eos_id", sorted(TRUTH["cases"]))
def test_alpha_beta_match_reference(eos_id):
    case = TRUTH["cases"][eos_id]
    if "alpha" not in case:
        pytest.skip(f"no alpha/beta reference for {eos_id}")
    eos = _eos(eos_id)
    np.testing.assert_allclose(np.asarray(eos.alpha(_T, _S, _P)),
                               np.array(case["alpha"]), rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(np.asarray(eos.beta(_T, _S, _P)),
                               np.array(case["beta"]), rtol=1e-5, atol=1e-9)


def test_provenance_present():
    # Guard against an un-stamped/hand-edited fixture.
    versions = TRUTH["provenance"]["reference_versions"]
    assert versions["gsw"] != "unknown"
    assert versions["fastjmd95"] != "unknown"
