"""Edge-of-funnel plausibility + the out-of-range warning (Q3 Gap 5, code side).

At low salinity / high temperature / deep pressure -- inputs that are extreme but
still *inside* each backend's documented envelope -- the potential-temp /
practical-salinity backends must return finite, physically plausible densities
(and non-decreasing-in-pressure where applicable).  This is a finiteness /
plausibility guard, NOT a cross-model truth comparison (gsw/teos10 legitimately
return NaN at S=0, so they are excluded here).

Separately, the facade must *warn* (never clamp) when inputs leave ``valid_range``,
and must stay silent for in-range inputs.
"""

import warnings

import numpy as np
import pytest

import xeos

# Potential-temp / practical-salinity backends (defined for S=0).
_POT_BACKENDS = ["jmd95", "unesco", "mdjwf", "wright97-full", "wright97-reduced",
                 "linear", "mpas-linear", "mpas-jm", "mpas-wright"]
# Incompressible ones (no pressure dependence) are excluded from the drho/dp check.
_INCOMPRESSIBLE = {"linear", "mpas-linear"}


@pytest.mark.parametrize("eos_id", _POT_BACKENDS)
@pytest.mark.parametrize("s", [0.0, 5.0, 20.0])
def test_edge_inputs_are_finite_and_plausible(eos_id, s):
    eos = xeos.equation_of_state(eos_id)
    t, p = 35.0, 6000.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # these inputs are in-range anyway
        rho = float(eos.rho(t, s, p))
    assert np.isfinite(rho)
    assert 950.0 < rho < 1100.0
    if eos_id not in _INCOMPRESSIBLE:
        dp = 50.0
        slope = (float(eos.rho(t, s, p + dp)) - float(eos.rho(t, s, p))) / dp
        assert slope >= -1e-9


def test_out_of_range_warns():
    eos = xeos.equation_of_state("jmd95")
    with pytest.warns(UserWarning, match="valid range"):
        eos.rho(10.0, 35.0, 50000.0)  # 50000 dbar >> 10000 dbar envelope


def test_in_range_does_not_warn():
    eos = xeos.equation_of_state("jmd95")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning -> failure
        eos.rho(10.0, 35.0, 3000.0)


def test_out_of_range_temperature_and_salinity_warn():
    eos = xeos.equation_of_state("wright97-reduced")
    with pytest.warns(UserWarning, match="valid range"):
        eos.rho(60.0, 35.0, 0.0)      # T above 40
    with pytest.warns(UserWarning, match="valid range"):
        eos.rho(10.0, 45.0, 0.0)      # S above 40


def test_lazy_inputs_skip_range_check():
    """DataArray/dask inputs are not range-checked (no forced compute)."""
    xr = pytest.importorskip("xarray")
    eos = xeos.equation_of_state("jmd95")
    p = xr.DataArray([50000.0, 60000.0], dims="z")  # wildly out of range
    t = xr.DataArray([10.0, 10.0], dims="z")
    s = xr.DataArray([35.0, 35.0], dims="z")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        eos.rho(t, s, p)
