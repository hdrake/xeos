"""Input-variable conventions for seawater equations of state.

Different EOS expect different *kinds* of temperature, salinity, and pressure.
TEOS-10 and the Roquet polynomials take conservative temperature and absolute
salinity; every other scheme here takes potential temperature and practical
salinity.  ``xeos`` never silently converts between these — instead each backend
declares the kinds it expects (as metadata) so that mismatches can be detected
and documented, and explicit conversion helpers are provided for users who need
them.
"""

from enum import Enum

__all__ = [
    "TemperatureKind",
    "SalinityKind",
    "PressureUnit",
    "to_conservative_temperature",
    "to_absolute_salinity",
    "pressure_from_depth",
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


def to_absolute_salinity(practical_salinity, p_dbar, lon=None, lat=None):
    """Convert practical salinity -> absolute salinity (requires ``gsw``).

    ``lon``/``lat`` are required for a geographically correct conversion; if
    omitted, falls back to ``gsw.SR_from_SP`` (reference salinity), which ignores
    the spatially-varying anomaly.
    """
    gsw = _require_gsw()
    if lon is None or lat is None:
        return gsw.SR_from_SP(practical_salinity)
    return gsw.SA_from_SP(practical_salinity, p_dbar, lon, lat)


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
