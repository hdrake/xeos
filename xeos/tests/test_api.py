"""Facade behaviour: xarray/dask preservation, conventions, derived quantities."""

import numpy as np
import pytest
import xarray as xr

import xeos
from xeos.conventions import TemperatureKind, SalinityKind
from xeos.backends._wright import FULL, _make


def test_scalar_call_returns_float():
    eos = xeos.equation_of_state("jmd95")
    out = eos.rho(3.0, 35.5, 3000.0)
    assert np.isclose(float(out), 1041.83267, atol=1e-4)  # fastjmd95 doc value


def test_xarray_labels_and_attrs_preserved():
    z = [0, 1, 2]
    T = xr.DataArray([5.0, 10.0, 15.0], dims="z", coords={"z": z})
    S = xr.DataArray([35.0, 35.5, 36.0], dims="z", coords={"z": z})
    p = xr.DataArray([0.0, 500.0, 1000.0], dims="z", coords={"z": z})
    out = xeos.equation_of_state("jmd95").rho(T, S, p)
    assert isinstance(out, xr.DataArray)
    assert out.dims == ("z",)
    assert list(out["z"].values) == z
    assert out.attrs["units"] == "kg m-3"


def test_dask_stays_lazy_and_correct():
    dask = pytest.importorskip("dask")  # noqa: F841
    z = [0, 1, 2]
    T = xr.DataArray([5.0, 10.0, 15.0], dims="z", coords={"z": z})
    S = xr.DataArray([35.0, 35.5, 36.0], dims="z", coords={"z": z})
    p = xr.DataArray([0.0, 500.0, 1000.0], dims="z", coords={"z": z})
    eos = xeos.equation_of_state("jmd95")
    eager = eos.rho(T, S, p)
    lazy = eos.rho(T.chunk(1), S.chunk(1), p.chunk(1))
    assert lazy.chunks is not None  # not yet computed
    np.testing.assert_allclose(lazy.values, eager.values)


def test_functional_shim_matches_facade():
    eos = xeos.equation_of_state("wright97-full")
    np.testing.assert_allclose(
        float(xeos.rho(10.0, 35.5, 0.0, eos="wright97-full")),
        float(eos.rho(10.0, 35.5, 0.0)),
    )


def test_backend_declares_conventions():
    assert xeos.equation_of_state("jmd95").temperature is TemperatureKind.POTENTIAL
    assert xeos.equation_of_state("jmd95").salinity is SalinityKind.PRACTICAL
    assert xeos.equation_of_state("teos10-poly55").temperature is TemperatureKind.CONSERVATIVE
    assert xeos.equation_of_state("teos10-poly55").salinity is SalinityKind.ABSOLUTE


def test_linear_derivatives_are_exact_constants():
    rho0, talpha, sbeta = 1027.0, 2.0e-4, 7.4e-4
    eos = xeos.equation_of_state("linear", rho0=rho0, talpha=talpha, sbeta=sbeta)
    # rho = rho0 + drdt*T + drds*S
    assert np.isclose(float(eos.rho(0.0, 0.0, 0.0)), rho0)
    assert np.isclose(float(eos.drho_dt(10.0, 35.0, 0.0)), -rho0 * talpha)
    assert np.isclose(float(eos.drho_ds(10.0, 35.0, 0.0)), rho0 * sbeta)


def test_wright_alpha_matches_finite_difference():
    """wright97-full has no external reference; check analytic alpha vs FD of rho."""
    density, _, _ = _make(FULL)
    t, s, p_pa = 12.0, 35.0, 1000.0 * 1.0e4  # native Pa
    h = 1.0e-4
    fd = -(density(t + h, s, p_pa) - density(t - h, s, p_pa)) / (2 * h) / density(t, s, p_pa)
    eos = xeos.equation_of_state("wright97-full")
    analytic = float(eos.alpha(t, s, 1000.0))  # facade input in dbar
    np.testing.assert_allclose(analytic, fd, rtol=1e-6)


def test_pressure_unit_conversion():
    """Same physical pressure given in dbar vs Pa yields the same density."""
    in_dbar = xeos.equation_of_state("jmd95", pressure_input_unit="dbar")
    in_pa = xeos.equation_of_state("jmd95", pressure_input_unit="Pa")
    np.testing.assert_allclose(
        float(in_dbar.rho(10.0, 35.0, 1000.0)),
        float(in_pa.rho(10.0, 35.0, 1000.0 * 1.0e4)),
    )
