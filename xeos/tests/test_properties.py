"""Property-based (hypothesis) invariants over the oceanographic funnel (Gap 6).

For every backend, sampled (t, s, p) in-funnel must give a positive, physically
banded density; specific volume ~ 1/rho; finite alpha/beta.  Density must be
non-decreasing in pressure (numerically) for the genuinely compressible realistic
EOS only -- linear, mpas-linear and the idealized Roquet forms are excluded (the
thermobaric idealized forms legitimately have drho/dp < 0 at positive T).  A
separate deterministic check pins cross-mode parity (scalar / 1-D / 2-D numpy /
float32 vs float64 / dask vs eager).
"""

import numpy as np
import pytest
import xarray as xr
from hypothesis import given, settings
from hypothesis import strategies as st

import xeos

try:  # teos10 needs the optional gsw extra
    import gsw  # noqa: F401
    _HAVE_GSW = True
except ImportError:  # pragma: no cover
    _HAVE_GSW = False

_BACKENDS = [b for b in xeos.list_eos() if b != "teos10" or _HAVE_GSW]

# Compressible realistic EOS: drho/dp >= 0.  Excludes linear, mpas-linear and all
# idealized roquet-* forms (thermobaric ones have drho/dp < 0 at positive T).
_COMPRESSIBLE = {
    "jmd95", "unesco", "mdjwf", "wright97-full", "wright97-reduced",
    "mpas-jm", "mpas-wright", "teos10", "teos10-poly55", "roquet-spv",
}

_T = st.floats(0.0, 25.0)
_S = st.floats(33.0, 37.0)
_P = st.floats(0.0, 4000.0)


@pytest.mark.parametrize("eos_id", _BACKENDS)
@settings(max_examples=25, deadline=None)
@given(t=_T, s=_S, p=_P)
def test_rho_positive_and_banded(eos_id, t, s, p):
    eos = xeos.equation_of_state(eos_id)
    r = float(eos.rho(t, s, p))
    assert r > 0.0
    assert 985.0 < r < 1105.0


@pytest.mark.parametrize("eos_id", _BACKENDS)
@settings(max_examples=25, deadline=None)
@given(t=_T, s=_S, p=_P)
def test_specific_volume_is_reciprocal(eos_id, t, s, p):
    eos = xeos.equation_of_state(eos_id)
    r = float(eos.rho(t, s, p))
    v = float(eos.specific_volume(t, s, p))
    np.testing.assert_allclose(v, 1.0 / r, rtol=1e-9)


@pytest.mark.parametrize("eos_id", _BACKENDS)
@settings(max_examples=25, deadline=None)
@given(t=_T, s=_S, p=_P)
def test_alpha_beta_finite(eos_id, t, s, p):
    eos = xeos.equation_of_state(eos_id)
    assert np.isfinite(float(eos.alpha(t, s, p)))
    assert np.isfinite(float(eos.beta(t, s, p)))


@pytest.mark.parametrize("eos_id", sorted(_COMPRESSIBLE & set(_BACKENDS)))
@settings(max_examples=25, deadline=None)
@given(t=_T, s=_S, p=_P)
def test_density_nondecreasing_in_pressure(eos_id, t, s, p):
    eos = xeos.equation_of_state(eos_id)
    dp = 50.0
    slope = (float(eos.rho(t, s, p + dp)) - float(eos.rho(t, s, p))) / dp
    assert slope >= -1e-9


@pytest.mark.parametrize("eos_id", _BACKENDS)
def test_cross_mode_parity(eos_id):
    """scalar / 1-D / 2-D numpy / float32-vs-float64 / dask-vs-eager all agree."""
    eos = xeos.equation_of_state(eos_id)
    t2 = np.array([[5.0, 10.0], [15.0, 20.0]])
    s2 = np.array([[34.0, 35.0], [36.0, 35.5]])
    p2 = np.array([[0.0, 1000.0], [2000.0, 3000.0]])

    r2 = np.asarray(eos.rho(t2, s2, p2))
    r1 = np.asarray(eos.rho(t2.ravel(), s2.ravel(), p2.ravel()))
    np.testing.assert_allclose(r2.ravel(), r1, rtol=1e-12, atol=1e-12)

    # scalar matches the corresponding element
    scalar = float(eos.rho(float(t2[0, 0]), float(s2[0, 0]), float(p2[0, 0])))
    np.testing.assert_allclose(scalar, r2[0, 0], rtol=1e-12, atol=1e-12)

    # DataArray float32 vs float64 (loose tol -- reduced precision)
    dims = ("y", "x")
    r32 = eos.rho(xr.DataArray(t2.astype(np.float32), dims=dims),
                  xr.DataArray(s2.astype(np.float32), dims=dims),
                  xr.DataArray(p2.astype(np.float32), dims=dims))
    np.testing.assert_allclose(np.asarray(r32), r2, rtol=1e-3, atol=1e-2)

    # dask vs eager (must stay lazy, then agree)
    daT = xr.DataArray(t2, dims=dims)
    daS = xr.DataArray(s2, dims=dims)
    daP = xr.DataArray(p2, dims=dims)
    eager = eos.rho(daT, daS, daP)
    lazy = eos.rho(daT.chunk(1), daS.chunk(1), daP.chunk(1))
    assert lazy.chunks is not None
    np.testing.assert_allclose(lazy.values, eager.values, rtol=1e-12, atol=1e-12)
