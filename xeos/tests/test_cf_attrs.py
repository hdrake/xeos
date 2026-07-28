"""CF-style output attributes: units must be real, parseable, and kind-aware.

Three things are pinned here:

* ``beta``/``drho_ds`` units depend on the backend's :class:`SalinityKind` --
  practical salinity (PSS-78) is dimensionless, absolute salinity is g kg-1 --
  so the two families must *not* advertise the same units;
* every ``units`` string xeos emits parses under UDUNITS-2 (what CF requires and
  what cf-xarray / MetPy / iris need in order to do unit arithmetic).  This is
  the regression guard for the ``"(salinity unit)-1"`` placeholder that used to
  sit in ``beta``/``drho_ds`` (hdrake/xeos#8);
* relabelling is metadata-only -- the numbers still match ``gsw`` exactly.
"""

import numpy as np
import pytest
import xarray as xr

import xeos
from xeos.conventions import SalinityKind
from xeos.eos import _ATTRS, _SALINITY_ATTRS
from xeos.registry import get_backend

# One representative backend of each salinity kind. teos10 needs the gsw extra.
_ABSOLUTE = "teos10"
_PRACTICAL = "jmd95"

_QUANTITIES = ("rho", "specific_volume", "alpha", "beta", "drho_dt", "drho_ds")

# Expected units, quantity -> salinity kind (None where kind-independent).
_EXPECTED = {
    "rho": {None: "kg m-3"},
    "specific_volume": {None: "m3 kg-1"},
    "alpha": {None: "K-1"},
    "drho_dt": {None: "kg m-3 K-1"},
    "beta": {SalinityKind.PRACTICAL: "1", SalinityKind.ABSOLUTE: "kg g-1"},
    "drho_ds": {SalinityKind.PRACTICAL: "kg m-3", SalinityKind.ABSOLUTE: "kg2 m-3 g-1"},
}


def _expected_units(name, kind):
    table = _EXPECTED[name]
    return table[None] if None in table else table[kind]


def _fields(eos_id, t=10.0, s=35.0, p=0.0):
    """All six quantities as DataArrays (attrs only ride along on DataArrays)."""
    eos = xeos.equation_of_state(eos_id)
    args = [xr.DataArray(np.array([v]), dims=("z",)) for v in (t, s, p)]
    return {name: getattr(eos, name)(*args) for name in _QUANTITIES}


def test_beta_units_differ_between_salinity_kinds():
    """The whole point of the fix: same placeholder no longer covers both."""
    pytest.importorskip("gsw")
    absolute = _fields(_ABSOLUTE)
    practical = _fields(_PRACTICAL)
    assert get_backend(_ABSOLUTE).salinity is SalinityKind.ABSOLUTE
    assert get_backend(_PRACTICAL).salinity is SalinityKind.PRACTICAL
    assert absolute["beta"].attrs["units"] == "kg g-1"
    assert practical["beta"].attrs["units"] == "1"
    assert absolute["drho_ds"].attrs["units"] == "kg2 m-3 g-1"
    assert practical["drho_ds"].attrs["units"] == "kg m-3"


@pytest.mark.parametrize("eos_id", sorted(xeos.list_eos()))
def test_salinity_units_follow_the_backends_declared_kind(eos_id):
    """Every registered backend labels beta/drho_ds per its own SalinityKind."""
    if eos_id == "teos10":
        pytest.importorskip("gsw")
    kind = get_backend(eos_id).salinity
    fields = _fields(eos_id)
    for name in ("beta", "drho_ds"):
        assert fields[name].attrs["units"] == _expected_units(name, kind), name


@pytest.mark.parametrize("eos_id", [_PRACTICAL, _ABSOLUTE])
def test_temperature_coefficients_are_per_kelvin(eos_id):
    """alpha/drho_dt are reciprocals of a temperature *difference* -> K-1."""
    if eos_id == "teos10":
        pytest.importorskip("gsw")
    fields = _fields(eos_id)
    assert fields["alpha"].attrs["units"] == "K-1"
    assert fields["drho_dt"].attrs["units"] == "kg m-3 K-1"


@pytest.mark.parametrize("eos_id", [_PRACTICAL, _ABSOLUTE])
def test_all_emitted_units_parse_under_udunits2(eos_id):
    """CF requires UDUNITS-2-parseable units; check all six, both kinds."""
    cf_units = pytest.importorskip("cf_units")
    kind = get_backend(eos_id).salinity
    if eos_id == "teos10":
        pytest.importorskip("gsw")
    for name, field in _fields(eos_id).items():
        units = field.attrs["units"]
        assert units == _expected_units(name, kind), name
        cf_units.Unit(units)  # raises ValueError if UDUNITS-2 cannot parse it


def test_degc_and_kelvin_labels_are_interconvertible():
    """The alpha/drho_dt relabelling is cosmetic: the factor really is 1."""
    cf_units = pytest.importorskip("cf_units")
    for old, new in (("degC-1", "K-1"), ("kg m-3 degC-1", "kg m-3 K-1")):
        converted = cf_units.Unit(old).convert(np.array([1.0]), cf_units.Unit(new))
        np.testing.assert_array_equal(converted, np.array([1.0]))


def test_only_density_quantities_carry_a_standard_name():
    """Deliberate asymmetry: CF has no standard name for the haline coefficient,
    so none of the four coefficient fields gets one (hdrake/xeos#8)."""
    pytest.importorskip("gsw")
    for eos_id in (_PRACTICAL, _ABSOLUTE):
        fields = _fields(eos_id)
        assert fields["rho"].attrs["standard_name"] == "sea_water_density"
        assert (
            fields["specific_volume"].attrs["standard_name"]
            == "sea_water_specific_volume"
        )
        for name in ("alpha", "beta", "drho_dt", "drho_ds"):
            assert "standard_name" not in fields[name].attrs, name


def test_every_quantity_is_labelled():
    """No quantity may silently lose its units/long_name."""
    pytest.importorskip("gsw")
    for eos_id in (_PRACTICAL, _ABSOLUTE):
        for name, field in _fields(eos_id).items():
            assert field.attrs["units"], name
            assert field.attrs["long_name"], name


def test_attrs_tables_cover_every_salinity_kind():
    """A newly added SalinityKind must not silently KeyError at call time."""
    for name, spec in _SALINITY_ATTRS.items():
        assert set(spec["units"]) == set(SalinityKind), name
    assert set(_ATTRS) == {"rho", "specific_volume", "alpha", "drho_dt"}


def test_teos10_alpha_beta_still_match_gsw():
    """Metadata-only change: values are untouched, to the last digit."""
    gsw = pytest.importorskip("gsw")
    t, s, p = 10.0, 35.0, 0.0
    eos = xeos.equation_of_state("teos10")
    np.testing.assert_allclose(
        float(eos.alpha(t, s, p)), gsw.alpha(s, t, p), rtol=1e-15, atol=0.0
    )
    np.testing.assert_allclose(
        float(eos.beta(t, s, p)), gsw.beta(s, t, p), rtol=1e-15, atol=0.0
    )


@pytest.mark.parametrize("eos_id", [_PRACTICAL, _ABSOLUTE])
def test_labelled_dataarray_values_equal_bare_numpy_values(eos_id):
    """Attaching attrs must not perturb the numbers on the xarray path."""
    if eos_id == "teos10":
        pytest.importorskip("gsw")
    eos = xeos.equation_of_state(eos_id)
    t, s, p = 10.0, 35.0, 0.0
    for name, field in _fields(eos_id, t, s, p).items():
        bare = getattr(eos, name)(t, s, p)
        np.testing.assert_array_equal(field.values, np.array([float(bare)]))
