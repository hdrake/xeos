"""TEOS-10 equation of state, delegated to the optional ``gsw`` library.

This calls ``gsw.rho`` / ``gsw.rho_first_derivatives``, which evaluate the
**75-term polynomial** approximation to TEOS-10 (Roquet et al., 2015, Ocean
Modelling 90) that GSW uses as its standard, computationally-efficient
implementation — NOT the exact Gibbs function (that is ``gsw.rho_t_exact``). It is
accurate to ~1e-3 kg m-3 within the oceanographic "funnel" and is what most
analysis code, and the MOM6/MITgcm ``TEOS10`` options, mean by "TEOS-10 density".
Relative to the vendored ``teos10-poly55`` (the 55-term Roquet *density* fit), this
is the 75-term *specific-volume* fit of the same TEOS-10 target.

Underlying standard: IOC/SCOR/IAPSO (2010); McDougall & Barker (2011), GSW toolbox.
``gsw`` is an optional extra (its only dependency is numpy) — install
``xeos[teos10]``.

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
    reference="gsw 75-term polynomial: Roquet et al. (2015), Ocean Modelling 90. "
              "TEOS-10 standard: IOC/SCOR/IAPSO (2010); McDougall & Barker (2011).",
    description="TEOS-10 via gsw's 75-term Roquet polynomial (MOM6/MITgcm TEOS10); "
                "not the exact Gibbs function.",
))
