"""Input-variable conventions for seawater equations of state.

Different EOS expect different *kinds* of temperature, salinity, and pressure.
TEOS-10 and the Roquet polynomials take conservative temperature and absolute
salinity; every other scheme here takes potential temperature and practical
salinity.  ``xeos`` never silently converts between these — instead each backend
declares the kinds it expects (as metadata) so that mismatches can be detected
and documented, and explicit conversion helpers are provided for users who need
them.

Because nothing is converted, a mislabelled input is used exactly as given and
quietly produces wrong numbers.  :func:`check_input_units` is the detection half
of that bargain: it reads the ``units`` attribute of whatever the caller passed
and says so when it disagrees with what the kernels expect.  It compares units
for **equality, not convertibility** — a temperature in K and a salinity
labelled a plain ``"1"`` are both convertible to what is wanted and both wrong,
by 273 and by 1000.  Inputs with no units, or units UDUNITS-2 cannot read, are
left alone: an absent label is not evidence of a wrong one.  The check needs
``cf-units``; without it there is simply no check.

The conversion helpers label what they return, so a converted array can be
handed straight back in and pass that check.
"""

import warnings
from enum import Enum

from .xarray_utils import apply_eos

__all__ = [
    "TemperatureKind",
    "SalinityKind",
    "PressureUnit",
    "to_conservative_temperature",
    "to_absolute_salinity",
    "pressure_from_depth",
    "check_input_units",
]


class TemperatureKind(Enum):
    """Kind of temperature an EOS expects."""

    POTENTIAL = "potential temperature"  # theta, degC
    CONSERVATIVE = "conservative temperature"  # Theta (CT), degC
    INSITU = "in-situ temperature"  # t, degC


class SalinityKind(Enum):
    """Kind of salinity an EOS expects."""

    PRACTICAL = "practical salinity"  # Sp, PSU (PSS-78)
    ABSOLUTE = "absolute salinity"  # SA, g/kg (TEOS-10)


class PressureUnit(Enum):
    """Native pressure unit a backend's kernel expects."""

    DBAR = "dbar"  # sea pressure, decibar (oceanographic standard)
    PASCAL = "Pa"  # absolute/sea pressure, pascal


#: Multiplicative conversion factors into each native unit, from sea pressure in dbar.
_DBAR_TO = {PressureUnit.DBAR: 1.0, PressureUnit.PASCAL: 1.0e4}


def to_native_pressure(p_dbar, unit):
    """Convert sea pressure in dbar to a backend's native ``unit``."""
    return p_dbar * _DBAR_TO[PressureUnit(unit)]


def _require_gsw():
    try:
        import gsw  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without gsw
        raise ImportError(
            "This conversion requires the optional 'gsw' dependency. "
            "Install it with `pip install xeos[teos10]`."
        ) from exc
    return __import__("gsw")


def to_conservative_temperature(potential_temperature, practical_salinity, p_dbar,
                                lon=None, lat=None):
    """Convert potential temperature -> conservative temperature (requires ``gsw``).

    Absolute salinity is needed, so ``lon``/``lat`` are used when available to
    compute it from practical salinity; otherwise the practical value is used as
    a (slightly inexact) proxy.
    """
    gsw = _require_gsw()
    SA = to_absolute_salinity(practical_salinity, p_dbar, lon, lat)
    return gsw.CT_from_pt(SA, potential_temperature)


#: CF attributes for the two quantities :func:`to_absolute_salinity` can return.
#: These are *different quantities*, not two names for one: reference salinity is
#: what composition alone implies, without the spatially-varying anomaly that
#: makes absolute salinity absolute.  Describing them apart is what stops a
#: lon/lat-less conversion passing for the real thing downstream.
#:
#: Both are ``g kg-1`` -- the same unit as the dimensionless-but-1e-3-scaled
#: practical salinity that went in.  The conversion changes the quantity, not the
#: unit, which is why it needs a ``standard_name`` to be legible at all.
_SALINITY_OUTPUT_ATTRS = {
    "absolute": {
        "units": "g kg-1",
        "standard_name": "sea_water_absolute_salinity",
        "long_name": "absolute salinity",
    },
    "reference": {
        "units": "g kg-1",
        "standard_name": "sea_water_reference_salinity",
        "long_name": "reference salinity",
        "comment": (
            "Converted from practical salinity without lon/lat, so the "
            "spatially-varying absolute salinity anomaly is not included."
        ),
    },
}


def to_absolute_salinity(practical_salinity, p_dbar, lon=None, lat=None):
    """Convert practical salinity -> absolute salinity (requires ``gsw``).

    ``lon``/``lat`` are required for a geographically correct conversion; if
    omitted, falls back to ``gsw.SR_from_SP`` (reference salinity), which ignores
    the spatially-varying anomaly.

    The result is labelled ``g kg-1``, and its ``standard_name``/``long_name``
    say *which* of the two quantities it is, so a caller who omitted
    ``lon``/``lat`` cannot have reference salinity pass for absolute salinity
    downstream.  Attributes are attached only when the input is a ``DataArray``;
    a numpy input still returns numpy.
    """
    gsw = _require_gsw()
    if lon is None or lat is None:
        return apply_eos(
            gsw.SR_from_SP,
            practical_salinity,
            attrs=_SALINITY_OUTPUT_ATTRS["reference"],
        )
    return apply_eos(
        gsw.SA_from_SP,
        practical_salinity,
        p_dbar,
        lon,
        lat,
        attrs=_SALINITY_OUTPUT_ATTRS["absolute"],
    )


#: UDUNITS has no practical-salinity unit.  PSS-78 carries no dimension but it
#: carries a *scale*: it is a conductivity-ratio scale built so that a salinity
#: of 35 is numerically 35 grams of salt per kilogram of seawater.  So it aliases
#: to ``"0.001"``, which UDUNITS-2 reads as exactly ``g kg-1`` -- not to ``"1"``,
#: which would understate it by a thousand.
_UNIT_ALIASES = {
    "psu": "0.001",
    "PSU": "0.001",
    "practical_salinity_unit": "0.001",
    "practical_salinity_units": "0.001",
    "pss-78": "0.001",
    "PSS-78": "0.001",
}

#: What the kernels expect, whatever kind of temperature or salinity a backend
#: declares.  Both salinity kinds are the same unit -- see the note on
#: :data:`~xeos.eos._SALINITY_ATTRS` -- so one entry covers both.
_EXPECTED_INPUT_UNITS = {"temperature": "degC", "salinity": "g kg-1"}


def _parse_units(spec):
    """Parse a units string into a ``cf_units.Unit``, or None if not checkable.

    None covers every reason a check cannot be made: no units attribute, a blank
    or unparseable one, or ``cf-units`` not installed at all.  It never raises --
    an input whose units xeos cannot read is not an input xeos should complain
    about.
    """
    try:
        from cf_units import Unit
    except ImportError:  # pragma: no cover - core-only install
        return None
    if spec is None:
        return None
    text = str(spec).strip()
    if not text:
        return None
    try:
        unit = Unit(_UNIT_ALIASES.get(text, text))
    except ValueError:
        return None
    return None if (unit.is_unknown() or unit.is_no_unit()) else unit


def check_input_units(temperature=None, salinity=None, pressure=None,
                      pressure_unit=None):
    """Complaints about inputs whose ``units`` disagree with what xeos expects.

    Returns a list of human-readable strings, empty when everything checkable
    agrees.  Only ``DataArray`` inputs carrying a parseable ``units`` attribute
    are examined; anything else is skipped, since an unlabelled array is not
    evidence of a *wrong* label.

    The comparison is unit **equality, not convertibility**, and that is the
    whole point.  A temperature in K and a salinity labelled a plain ``"1"`` are
    both convertible to what the kernels want and both numerically wrong by a
    large factor -- 273 and 1000 respectively.  Since ``xeos`` never silently
    converts its inputs (see the module docstring), convertibility is not the
    question; being the same unit is.
    """
    complaints = []
    checks = [
        ("temperature", temperature, _EXPECTED_INPUT_UNITS["temperature"]),
        ("salinity", salinity, _EXPECTED_INPUT_UNITS["salinity"]),
    ]
    if pressure_unit is not None:
        checks.append(("pressure", pressure, PressureUnit(pressure_unit).value))
    for name, array, expected_spec in checks:
        found = _parse_units(getattr(array, "attrs", {}).get("units"))
        expected = _parse_units(expected_spec)
        if found is None or expected is None or found == expected:
            continue
        complaints.append(
            f"{name} is labelled {str(found)!r} but xeos expects "
            f"{expected_spec!r}; it does not convert its inputs, so the values "
            f"are used as given"
        )
    return complaints


def warn_on_input_units(temperature=None, salinity=None, pressure=None,
                        pressure_unit=None):
    """Emit one combined :class:`UserWarning` for whatever ``check_input_units`` finds."""
    complaints = check_input_units(temperature, salinity, pressure, pressure_unit)
    if complaints:
        warnings.warn("xeos input units: " + "; ".join(complaints), stacklevel=3)


def pressure_from_depth(depth_m, lat=None):
    """Convert geometric depth [m, positive down] to sea pressure [dbar].

    Uses ``gsw.p_from_z`` when ``gsw`` and ``lat`` are available; otherwise the
    common ~1 dbar/m approximation.
    """
    if lat is not None:
        try:
            gsw = _require_gsw()
            return gsw.p_from_z(-abs(depth_m), lat)
        except ImportError:
            pass
    return depth_m  # ~1 dbar per metre
