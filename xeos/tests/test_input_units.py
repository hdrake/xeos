"""Units at the boundary: what xeos checks on the way in, and labels on the way out.

``xeos`` never silently converts its inputs, so a mislabelled input is used as
given and quietly produces wrong numbers. The check exists to say so. The rule
is unit **equality, not convertibility**: a temperature in K and a salinity
labelled a plain ``"1"`` are both convertible to what the kernels want, and both
wrong by 273 and 1000 respectively.

Everything here is metadata-driven, so it works on lazy inputs and never forces
a compute -- unlike the ``valid_range`` check in ``test_valid_range.py``, which
has to look at the numbers.
"""

import sys
import warnings

import numpy as np
import pytest
import xarray as xr

import xeos
from xeos.conventions import (
    _SALINITY_OUTPUT_ATTRS,
    check_input_units,
    to_absolute_salinity,
)


def _da(value, units=None):
    return xr.DataArray(
        np.array([value]), dims=("z",), attrs={} if units is None else {"units": units}
    )


def _complaints(t_units, s_units, p_units="dbar", eos_id="teos10"):
    """Whatever `eos.rho` warns about the labels of its inputs."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        xeos.equation_of_state(eos_id).rho(
            _da(10.0, t_units), _da(35.0, s_units), _da(0.0, p_units)
        )
    return [str(w.message) for w in caught if "input units" in str(w.message)]


# --- what must pass silently -------------------------------------------------


@pytest.mark.parametrize("salinity_units", ["psu", "PSU", "pss-78", "g kg-1", "0.001"])
def test_every_spelling_of_the_right_salinity_unit_is_accepted(salinity_units):
    """MOM6 says "psu", CMIP6 says "0.001", TEOS-10 says "g kg-1" -- one unit."""
    pytest.importorskip("gsw")
    assert _complaints("degC", salinity_units) == []


def test_unlabelled_inputs_are_not_complained_about():
    """An absent label is not evidence of a wrong one."""
    pytest.importorskip("gsw")
    assert _complaints(None, None, None) == []


def test_unparseable_labels_are_not_complained_about():
    """xeos cannot judge a unit string it cannot read."""
    pytest.importorskip("gsw")
    assert _complaints("degrees_C_maybe", "very salty", None) == []


def test_plain_numpy_inputs_are_unaffected():
    """There are no attributes to check, and none are invented."""
    pytest.importorskip("gsw")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        xeos.equation_of_state("teos10").rho(10.0, 35.0, 0.0)
    assert [w for w in caught if "input units" in str(w.message)] == []


# --- what must be caught -----------------------------------------------------


def test_dimensionless_salinity_is_caught():
    """The 1000x bug: PSS-78 is dimensionless but *scaled*, so "1" understates it.

    This is convertible to `g kg-1` and therefore invisible to a
    convertibility check, which is exactly why the comparison is equality.
    """
    pytest.importorskip("cf_units")
    pytest.importorskip("gsw")
    (message,) = _complaints("degC", "1")
    assert "salinity is labelled '1'" in message
    assert "g kg-1" in message


def test_kelvin_temperature_is_caught():
    """Also convertible, also wrong -- by 273, since xeos does not convert."""
    pytest.importorskip("cf_units")
    pytest.importorskip("gsw")
    (message,) = _complaints("K", "psu")
    assert "temperature is labelled 'K'" in message


def test_pressure_is_checked_against_the_declared_input_unit():
    """`pressure_input_unit` is a promise about `p`; a label may contradict it."""
    pytest.importorskip("cf_units")
    pytest.importorskip("gsw")
    (message,) = _complaints("degC", "psu", "Pa")
    assert "pressure is labelled 'Pa'" in message

    # ...and the same array is fine for an EOS built to expect Pa.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        xeos.equation_of_state("teos10", pressure_input_unit="Pa").rho(
            _da(10.0, "degC"), _da(35.0, "psu"), _da(0.0, "Pa")
        )
    assert [w for w in caught if "input units" in str(w.message)] == []


def test_several_bad_labels_give_one_combined_warning():
    pytest.importorskip("cf_units")
    pytest.importorskip("gsw")
    (message,) = _complaints("K", "1")
    assert "temperature" in message and "salinity" in message


def test_check_input_units_is_pure_and_returns_complaints():
    """The predicate is usable on its own, without provoking a warning."""
    pytest.importorskip("cf_units")
    assert check_input_units(_da(10.0, "degC"), _da(35.0, "psu")) == []
    assert len(check_input_units(_da(10.0, "K"), _da(35.0, "1"))) == 2
    assert check_input_units() == []


def test_the_check_never_forces_a_compute():
    """Labels are metadata, so a dask-backed input is checked without loading."""
    pytest.importorskip("cf_units")
    dask = pytest.importorskip("dask.array")
    pytest.importorskip("gsw")
    lazy = xr.DataArray(35.0 * dask.ones(4, chunks=2), dims=("z",), attrs={"units": "1"})
    t = xr.DataArray(np.full(4, 10.0), dims=("z",), attrs={"units": "degC"})
    p = xr.DataArray(np.zeros(4), dims=("z",), attrs={"units": "dbar"})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = xeos.equation_of_state("teos10").rho(t, lazy, p)
    assert any("salinity is labelled '1'" in str(w.message) for w in caught)
    assert hasattr(result.data, "compute"), "input must still be lazy"


# --- what conversions label on the way out -----------------------------------


def test_absolute_salinity_output_is_labelled():
    pytest.importorskip("gsw")
    result = to_absolute_salinity(_da(35.0, "psu"), 0.0, -30.0, 45.0)
    assert result.attrs == _SALINITY_OUTPUT_ATTRS["absolute"]
    assert result.attrs["units"] == "g kg-1"


def test_the_lon_lat_free_fallback_says_it_is_reference_salinity():
    """It is a different quantity, and must not pass for absolute salinity."""
    pytest.importorskip("gsw")
    absolute = to_absolute_salinity(_da(35.0, "psu"), 0.0, -30.0, 45.0)
    reference = to_absolute_salinity(_da(35.0, "psu"), 0.0)

    assert reference.attrs["standard_name"] == "sea_water_reference_salinity"
    assert absolute.attrs["standard_name"] == "sea_water_absolute_salinity"
    assert "anomaly is not included" in reference.attrs["comment"]
    # Not merely differently labelled -- genuinely different numbers.
    assert float(absolute.values[0]) != float(reference.values[0])


def test_converted_salinity_round_trips_into_an_absolute_eos_without_complaint():
    """The output label is exactly what the input check wants to see."""
    pytest.importorskip("gsw")
    sa = to_absolute_salinity(_da(35.0, "psu"), 0.0, -30.0, 45.0)
    assert check_input_units(_da(10.0, "degC"), sa) == []


def test_numpy_input_still_returns_numpy():
    """Labelling must not turn a bare-array call into an xarray one."""
    pytest.importorskip("gsw")
    assert isinstance(to_absolute_salinity(np.array([35.0]), 0.0), np.ndarray)


def test_emitted_salinity_units_parse_under_udunits2():
    cf_units = pytest.importorskip("cf_units")
    for spec in _SALINITY_OUTPUT_ATTRS.values():
        assert cf_units.Unit(spec["units"]) == cf_units.Unit("g kg-1")


def test_the_check_is_a_no_op_without_cf_units(monkeypatch):
    """`cf-units` is optional, so a core-only install must not lose the EOS.

    Setting the module to None makes ``from cf_units import Unit`` raise
    ImportError, which is what a core-only install looks like from inside
    `_parse_units`.
    """
    pytest.importorskip("gsw")
    monkeypatch.setitem(sys.modules, "cf_units", None)
    # Labels that would otherwise be caught now simply cannot be judged.
    assert check_input_units(_da(10.0, "K"), _da(35.0, "1")) == []
    assert _complaints("K", "1") == []
    # ...and the actual computation is unaffected.
    assert np.isfinite(
        xeos.equation_of_state("teos10").rho(_da(10.0, "degC"), _da(35.0, "psu"), _da(0.0))
    ).all()
