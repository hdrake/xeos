"""The optional numba fast path must equal the numpy reference to ~machine
precision (the safeguard that makes the accelerated kernels trustworthy).

Requires numba; skipped otherwise.  Also checks that requesting acceleration for a
backend without a fast path (e.g. ``linear``) raises the documented
NotImplementedError.
"""

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("numba")

import xeos  # noqa: E402
from xeos.eos import EquationOfState  # noqa: E402
from xeos.registry import get_backend  # noqa: E402
from xeos.backends._accel import FAST_BACKENDS  # noqa: E402

#: accelerated backends that also carry fast analytic derivatives.
_FAST_DERIV_BACKENDS = tuple(
    e for e in FAST_BACKENDS if get_backend(e).drho_dt_fast is not None
)

_T = np.linspace(-1.0, 32.0, 6)
_S = np.linspace(30.0, 38.0, 5)
_P = np.linspace(0.0, 6000.0, 4)


def _mesh():
    return (a.ravel() for a in np.meshgrid(_T, _S, _P))


@pytest.mark.parametrize("eos_id", FAST_BACKENDS)
def test_accel_matches_numpy_numpy_inputs(eos_id):
    base = EquationOfState(eos_id)
    fast = EquationOfState(eos_id, accelerate=True)
    t, s, p = _mesh()
    np.testing.assert_allclose(np.asarray(fast.rho(t, s, p)),
                               np.asarray(base.rho(t, s, p)),
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("eos_id", FAST_BACKENDS)
def test_accel_matches_numpy_scalar(eos_id):
    base = EquationOfState(eos_id)
    fast = EquationOfState(eos_id, accelerate=True)
    np.testing.assert_allclose(float(fast.rho(10.0, 35.0, 2000.0)),
                               float(base.rho(10.0, 35.0, 2000.0)),
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("eos_id", FAST_BACKENDS)
def test_accel_matches_numpy_dataarray_and_dask(eos_id):
    base = EquationOfState(eos_id)
    fast = EquationOfState(eos_id, accelerate=True)
    t, s, p = (a.reshape(6, -1) for a in _mesh())
    dims = ("y", "x")
    daT, daS, daP = (xr.DataArray(a, dims=dims) for a in (t, s, p))

    expected = np.asarray(base.rho(daT, daS, daP))
    got_da = np.asarray(fast.rho(daT, daS, daP))
    np.testing.assert_allclose(got_da, expected, rtol=1e-12, atol=1e-12)

    lazy = fast.rho(daT.chunk(2), daS.chunk(2), daP.chunk(2))
    assert lazy.chunks is not None
    np.testing.assert_allclose(lazy.values, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("eos_id", FAST_BACKENDS)
def test_accel_alpha_beta_match_numpy(eos_id):
    """alpha/beta under acceleration match numpy -- for analytic-derivative
    backends via the fast derivative kernels, for FD-only backends via the
    (accelerated) density difference stencil."""
    base = EquationOfState(eos_id)
    fast = EquationOfState(eos_id, accelerate=True)
    t, s, p = _mesh()
    for q in ("alpha", "beta"):
        np.testing.assert_allclose(np.asarray(getattr(fast, q)(t, s, p)),
                                   np.asarray(getattr(base, q)(t, s, p)),
                                   rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("eos_id", _FAST_DERIV_BACKENDS)
def test_accel_derivative_kernels_match_numpy(eos_id):
    """The fast analytic drho_dt/drho_ds equal the numpy analytic ones, and the
    facade actually dispatches to them under accelerate=True."""
    base = EquationOfState(eos_id)
    fast = EquationOfState(eos_id, accelerate=True)
    assert fast._drho_dt_kernel is get_backend(eos_id).drho_dt_fast
    assert fast._drho_ds_kernel is get_backend(eos_id).drho_ds_fast
    t, s, p = _mesh()
    for q in ("drho_dt", "drho_ds"):
        np.testing.assert_allclose(np.asarray(getattr(fast, q)(t, s, p)),
                                   np.asarray(getattr(base, q)(t, s, p)),
                                   rtol=1e-12, atol=1e-12)


def test_fd_only_backend_has_no_fast_derivative():
    """jmd95 is FD-only: no analytic (fast) derivative, so accelerate falls back to
    differencing the accelerated density."""
    b = get_backend("jmd95")
    assert b.density_fast is not None
    assert b.drho_dt_fast is None and b.drho_ds_fast is None
    fast = EquationOfState("jmd95", accelerate=True)
    assert fast._drho_dt_kernel is None  # -> FD of fast._density


def test_accelerate_on_non_accelerated_backend_raises():
    """`linear` has no fast kernel (density_fast is None) -> NotImplementedError."""
    with pytest.raises(NotImplementedError):
        EquationOfState("linear", accelerate=True)


def test_accelerate_false_uses_numpy_density():
    eos = EquationOfState("jmd95", accelerate=False)
    assert eos._density is eos.backend.density
    assert eos._drho_dt_kernel is eos.backend.drho_dt
