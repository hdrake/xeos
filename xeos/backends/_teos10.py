"""Full TEOS-10 equation of state, delegated to the optional ``gsw`` library.

IOC, SCOR & IAPSO (2010); McDougall & Barker (2011), GSW Oceanographic Toolbox.

This is the only backend that is not vendored: the full Gibbs-function TEOS-10
standard is large and authoritatively implemented by ``gsw`` (whose only
dependency is numpy).  ``gsw`` is an optional extra — install ``xeos[teos10]``.

State variables: conservative temperature [degC], absolute salinity [g/kg],
sea pressure [dbar].
"""

from ..conventions import TemperatureKind, SalinityKind, PressureUnit
from ..registry import EOSBackend, register


def _gsw():
    try:
        import gsw
    except ImportError as exc:  # pragma: no cover - exercised only without gsw
        raise ImportError(
            "The 'teos10' EOS requires the optional 'gsw' dependency. "
            "Install it with `pip install xeos[teos10]`."
        ) from exc
    return gsw


def density(ct, sa, p_dbar):
    return _gsw().rho(sa, ct, p_dbar)


def drho_dt(ct, sa, p_dbar):
    _, rho_ct, _ = _gsw().rho_first_derivatives(sa, ct, p_dbar)
    return rho_ct


def drho_ds(ct, sa, p_dbar):
    rho_sa, _, _ = _gsw().rho_first_derivatives(sa, ct, p_dbar)
    return rho_sa


register(EOSBackend(
    id="teos10",
    density=density,
    drho_dt=drho_dt,
    drho_ds=drho_ds,
    temperature=TemperatureKind.CONSERVATIVE,
    salinity=SalinityKind.ABSOLUTE,
    pressure_unit=PressureUnit.DBAR,
    reference="IOC/SCOR/IAPSO (2010); McDougall & Barker (2011), GSW toolbox.",
    description="Full TEOS-10 via the gsw library (MOM6/MITgcm TEOS10).",
))
