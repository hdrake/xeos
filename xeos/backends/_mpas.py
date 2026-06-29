"""MPAS-Ocean (MPAS-O) equations of state.

MPAS-O (the ocean component of E3SM) selects its EOS with the namelist option
``config_eos_type``, which accepts exactly three values -- ``linear``, ``jm`` and
``wright`` -- implemented in
``components/mpas-ocean/src/shared/mpas_ocn_equation_of_state_{linear,jm,wright}.F``.
The ``jm`` (Jackett & McDougall 1995) and ``wright`` (Wright 1997) kernels are
byte-for-byte identical -- same coefficients and same algebraic form -- to xeos's
existing :mod:`._jmd95` and :mod:`._wright` (reduced-range) kernels, so this module
*reuses those kernel functions* rather than re-vendoring the coefficients; the
MPAS-O truth fixtures (``mpas-jm`` / ``mpas-wright``) validate that reuse against
MPAS-O's own compiled Fortran. ``mpas-linear`` is the linear form with MPAS-O's
namelist defaults baked in.

State variables (all three): potential temperature [degC], practical salinity
[PSU]; ``jm`` takes sea pressure [dbar], ``wright`` pressure [Pa], ``linear`` is
pressure-independent.

Two MPAS-O specifics are *documented here but intentionally not exposed* (the
facade takes pressure as an input, and the validation grid is in-range):

* **T/S clamping** before the polynomial -- ``jm`` clamps to T in [-2, 40] degC and
  S in [0, 42] PSU; ``wright`` to T in [-3, 30] degC and S in [28, 38] PSU.
* **depth -> pressure** -- ``jm`` derives a per-layer reference pressure in bars
  from depth via a POP/Levitus polynomial
  ``P_bar(z) = 0.059808*(exp(-0.025*z) - 1) + 0.100766*z + 2.28405e-7*z**2``;
  ``wright`` uses the Boussinesq linearisation ``p = -rho0*g*z``.
"""

from ..conventions import TemperatureKind, SalinityKind, PressureUnit
from ..registry import EOSBackend, register
from ._jmd95 import density as _jmd95_density
from ._wright import REDUCED as _WRIGHT_REDUCED, _make as _wright_make

# --- mpas-linear: rho = RhoRef - Alpha*(T - Tref) + Beta*(S - Sref) -----------
# MPAS-O Registry defaults (config_eos_linear_*): a *decrease* with temperature
# (-Alpha) and *increase* with salinity (+Beta); Alpha/Beta are dimensional
# (kg m-3 per degC / per PSU), not normalised by density.
_RHO_REF = 1000.0   # config_eos_linear_densityref [kg m-3]
_ALPHA = 0.2        # config_eos_linear_alpha      [kg m-3 degC-1]
_BETA = 0.8         # config_eos_linear_beta       [kg m-3 PSU-1]
_T_REF = 5.0        # config_eos_linear_Tref       [degC]
_S_REF = 35.0       # config_eos_linear_Sref       [PSU]


def _mpas_linear_density(t, s, p):
    return _RHO_REF - _ALPHA * (t - _T_REF) + _BETA * (s - _S_REF)


def _mpas_linear_drho_dt(t, s, p):
    return -_ALPHA + 0.0 * t


def _mpas_linear_drho_ds(t, s, p):
    return _BETA + 0.0 * s


# --- mpas-wright: reuse the reduced-range Wright (1997) kernel -----------------
_wright_density, _wright_drho_dt, _wright_drho_ds = _wright_make(_WRIGHT_REDUCED)


register(EOSBackend(
    id="mpas-linear",
    density=_mpas_linear_density,
    drho_dt=_mpas_linear_drho_dt,
    drho_ds=_mpas_linear_drho_ds,
    temperature=TemperatureKind.POTENTIAL,
    salinity=SalinityKind.PRACTICAL,
    pressure_unit=PressureUnit.DBAR,
    reference="MPAS-O config_eos_type='linear' (mpas_ocn_equation_of_state_linear.F).",
    description=(
        f"rho = {_RHO_REF} - {_ALPHA}*(T-{_T_REF}) + {_BETA}*(S-{_S_REF}); "
        "MPAS-O namelist defaults; pressure-independent (incompressible)."
    ),
))

register(EOSBackend(
    id="mpas-jm",
    density=_jmd95_density,
    temperature=TemperatureKind.POTENTIAL,
    salinity=SalinityKind.PRACTICAL,
    pressure_unit=PressureUnit.DBAR,
    reference=("MPAS-O config_eos_type='jm' (mpas_ocn_equation_of_state_jm.F); "
               "Jackett & McDougall (1995), J. Atmos. Oceanic Technol., 12, 381-389."),
    description=("Jackett-McDougall (1995) UNESCO/EOS-80 refit; identical kernel to "
                 "xeos 'jmd95'. MPAS-O clamps T,S and derives pressure from depth "
                 "(documented in this module; not applied here)."),
))

register(EOSBackend(
    id="mpas-wright",
    density=_wright_density,
    drho_dt=_wright_drho_dt,
    drho_ds=_wright_drho_ds,
    temperature=TemperatureKind.POTENTIAL,
    salinity=SalinityKind.PRACTICAL,
    pressure_unit=PressureUnit.PASCAL,
    reference=("MPAS-O config_eos_type='wright' (mpas_ocn_equation_of_state_wright.F); "
               "Wright (1997), J. Atmos. Oceanic Technol., 14, 735-740."),
    description=("Wright (1997) reduced-range coefficients (Table 1, last column); "
                 "identical kernel to xeos 'wright97-reduced'. MPAS-O clamps T,S and "
                 "uses Boussinesq pressure p=-rho0*g*z (documented; not applied here)."),
))
