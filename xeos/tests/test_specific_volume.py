"""Specific-volume consistency (Q3 Gap 7).

* fallback path (every backend except ``roquet-spv``): rho * v == 1 to machine
  precision, since v is computed as 1/rho;
* ``roquet-spv`` ships an *analytic* specific volume -- check it equals 1/rho;
* ``teos10``: parity of the facade's specific volume against gsw's own specvol.
"""

import numpy as np
import pytest

import xeos

_T = np.array([0.0, 8.0, 18.0, 28.0])
_S = np.array([33.0, 35.0, 37.0])
_P = np.array([0.0, 1000.0, 4000.0])


def _mesh():
    return (a.ravel() for a in np.meshgrid(_T, _S, _P))


def _skip_if_missing(eos_id):
    try:
        eos = xeos.equation_of_state(eos_id)
        eos.rho(float(_T[0]), float(_S[0]), float(_P[0]))
        return eos
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"optional dependency missing for {eos_id}: {exc}")


@pytest.mark.parametrize("eos_id", [b for b in xeos.list_eos() if b != "roquet-spv"])
def test_rho_times_v_is_one_fallback(eos_id):
    eos = _skip_if_missing(eos_id)
    t, s, p = _mesh()
    rho = np.asarray(eos.rho(t, s, p))
    v = np.asarray(eos.specific_volume(t, s, p))
    np.testing.assert_allclose(rho * v, 1.0, rtol=1e-12, atol=1e-12)


def test_roquet_spv_analytic_specvol_matches_reciprocal():
    """roquet-spv provides an analytic specific volume; it must equal 1/rho."""
    eos = xeos.equation_of_state("roquet-spv")
    t, s, p = _mesh()
    rho = np.asarray(eos.rho(t, s, p))
    v = np.asarray(eos.specific_volume(t, s, p))
    np.testing.assert_allclose(v, 1.0 / rho, rtol=1e-12, atol=1e-12)


def test_teos10_specific_volume_matches_gsw():
    gsw = pytest.importorskip("gsw")
    eos = xeos.equation_of_state("teos10")
    t, s, p = _mesh()
    v = np.asarray(eos.specific_volume(t, s, p))
    np.testing.assert_allclose(v, gsw.specvol(s, t, p), rtol=1e-10, atol=1e-14)
