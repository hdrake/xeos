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


_IDEALIZED_ROQUET = [
    "roquet-linear", "roquet-cabbeling", "roquet-cabbeling-thermobaricity",
    "roquet-freezing", "roquet-second-order", "roquet-simplest-realistic",
]


@pytest.mark.parametrize("eos_id", _IDEALIZED_ROQUET)
def test_idealized_roquet_derivatives_match_finite_difference(eos_id):
    """No Python reference exists; check the analytic alpha/beta against FD of rho."""
    eos = xeos.equation_of_state(eos_id)
    ct, sa, p = 8.0, 34.0, 1500.0
    h = 1.0e-4
    rho = float(eos.rho(ct, sa, p))
    fd_alpha = -(float(eos.rho(ct + h, sa, p)) - float(eos.rho(ct - h, sa, p))) / (2 * h) / rho
    fd_beta = (float(eos.rho(ct, sa + h, p)) - float(eos.rho(ct, sa - h, p))) / (2 * h) / rho
    np.testing.assert_allclose(float(eos.alpha(ct, sa, p)), fd_alpha, rtol=1e-6)
    np.testing.assert_allclose(float(eos.beta(ct, sa, p)), fd_beta, rtol=1e-6)


# Independent check values at (CT=10, SA=35, p=0 -> Z=0), hand-computed from the
# Roquet et al. (2015) Table 3 coefficients (rho_ref=1024.6). These do not reuse
# the code's COEFFICIENTS dict, so a transcription typo there would be caught.
@pytest.mark.parametrize("eos_id,expected", [
    ("roquet-linear", 1049.838),                    # +0.7718*35 -0.1775*10
    ("roquet-cabbeling", 1050.3129),                # -0.0844*10 -4.561e-3*100
    ("roquet-cabbeling-thermobaricity", 1050.4573),  # -0.0651*10 -5.027e-3*100
    ("roquet-freezing", 1050.6193),                 # -0.0491*10 -5.027e-3*100
    ("roquet-second-order", 1051.5686),             # full 2nd-order at Z=0
    ("roquet-simplest-realistic", 1050.505),        # 0.77*35 -0.0495*10 -0.0055*100
])
def test_idealized_roquet_check_values(eos_id, expected):
    eos = xeos.equation_of_state(eos_id)
    assert np.isclose(float(eos.rho(10.0, 35.0, 0.0)), expected, atol=1e-3)


def test_thermobaricity_adds_pressure_dependence():
    # Cabbeling has no Z term (pressure-independent); adding thermobaricity introduces one.
    cab = xeos.equation_of_state("roquet-cabbeling")
    tb = xeos.equation_of_state("roquet-cabbeling-thermobaricity")
    assert np.isclose(float(cab.rho(10.0, 35.0, 0.0)), float(cab.rho(10.0, 35.0, 2000.0)))
    assert not np.isclose(float(tb.rho(10.0, 35.0, 0.0)), float(tb.rho(10.0, 35.0, 2000.0)))


def test_wright_full_sanity_vs_gsw():
    """wright97-full has no exact Python reference (momlevel ships the *reduced*
    coefficients). Sanity-bound its density against TEOS-10/gsw over an
    oceanographic range to catch gross coefficient-transcription typos
    (measured max deviation ~0.17 kg/m3; bound at 0.5)."""
    gsw = pytest.importorskip("gsw")
    T = np.array([0.0, 5.0, 10.0, 20.0, 30.0])
    S = np.array([33.0, 35.0, 37.0])
    P = np.array([0.0, 1000.0, 4000.0])
    t, s, p = (a.ravel() for a in np.meshgrid(T, S, P))
    wf = np.asarray(xeos.equation_of_state("wright97-full").rho(t, s, p))
    assert np.max(np.abs(wf - gsw.rho(s, t, p))) < 0.5


def test_low_salinity_beta_is_finite():
    """Regression: beta of the finite-difference backends must not be NaN near
    fresh water (the FD stencil contains sqrt(s))."""
    for eos_id in ("jmd95", "unesco", "mdjwf"):
        eos = xeos.equation_of_state(eos_id)
        for s in (0.0, 0.0005, 0.01):
            assert np.isfinite(float(eos.beta(10.0, s, 0.0))), (eos_id, s)


def test_pressure_unit_conversion():
    """Same physical pressure given in dbar vs Pa yields the same density."""
    in_dbar = xeos.equation_of_state("jmd95", pressure_input_unit="dbar")
    in_pa = xeos.equation_of_state("jmd95", pressure_input_unit="Pa")
    np.testing.assert_allclose(
        float(in_dbar.rho(10.0, 35.0, 1000.0)),
        float(in_pa.rho(10.0, 35.0, 1000.0 * 1.0e4)),
    )
