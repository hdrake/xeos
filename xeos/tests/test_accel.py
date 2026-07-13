"""The optional numba fast path must equal the numpy reference to ~machine
precision (the safeguard that makes the accelerated kernels trustworthy).

Requires numba; skipped otherwise.  Also checks that requesting acceleration for a
backend without a fast path (e.g. ``linear``) raises the documented ImportError.
"""

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("numba")

import xeos  # noqa: E402
from xeos.eos import EquationOfState  # noqa: E402
from xeos.backends._accel import FAST_BACKENDS  # noqa: E402

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


def test_accelerate_on_non_accelerated_backend_raises():
    """`linear` has no fast kernel (density_fast is None) -> helpful ImportError."""
    with pytest.raises(ImportError):
        EquationOfState("linear", accelerate=True)


def test_accelerate_false_uses_numpy_density():
    eos = EquationOfState("jmd95", accelerate=False)
    assert eos._density is eos.backend.density
