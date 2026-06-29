"""Linear equation of state: rho = rho0 + drho_dT * T + drho_dS * S.

Used by all three target models (MOM6 ``LINEAR``, MITgcm ``LINEAR``,
Oceananigans ``LinearEquationOfState``).  The coefficients are configurable;
the registered default uses MITgcm-style values.
"""

from ..conventions import TemperatureKind, SalinityKind, PressureUnit
from ..registry import EOSBackend, register

# MITgcm-style defaults (rhoNil, tAlpha, sBeta).
DEFAULT_RHO0 = 999.8
DEFAULT_TALPHA = 2.0e-4  # thermal expansion [degC-1]
DEFAULT_SBETA = 7.4e-4   # haline contraction [PSU-1]


def make_linear(rho0=DEFAULT_RHO0, talpha=DEFAULT_TALPHA, sbeta=DEFAULT_SBETA,
                eos_id="linear"):
    """Build a linear :class:`EOSBackend` from expansion/contraction coefficients.

    ``rho = rho0 * (1 - talpha * T + sbeta * S)``, so ``drho/dT = -rho0*talpha``
    and ``drho/dS = rho0*sbeta`` are exact constants.
    """
    drdt = -rho0 * talpha
    drds = rho0 * sbeta

    def density(t, s, p):
        return rho0 + drdt * t + drds * s

    def drho_dt(t, s, p):
        return drdt + 0.0 * t

    def drho_ds(t, s, p):
        return drds + 0.0 * s

    return EOSBackend(
        id=eos_id,
        density=density,
        drho_dt=drho_dt,
        drho_ds=drho_ds,
        temperature=TemperatureKind.POTENTIAL,
        salinity=SalinityKind.PRACTICAL,
        pressure_unit=PressureUnit.DBAR,
        reference="Linear EOS (configurable thermal/haline coefficients).",
        description=(
            f"rho0={rho0}, talpha={talpha}, sbeta={sbeta}; "
            "pressure-independent (incompressible)."
        ),
    )


register(make_linear())
