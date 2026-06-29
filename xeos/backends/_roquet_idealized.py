"""Idealized second-order Roquet equations of state (Oceananigans).

Roquet, F., G. Madec, T.J. McDougall, P.M. Barker (2015): "Defining a Simplified
yet 'Realistic' Equation of State for Seawater." J. Phys. Oceanogr., 45, Table 3.

These are the ``RoquetSeawaterPolynomial(:Linear | :Cabbeling | ...)`` options in
SeawaterPolynomials.jl / Oceananigans.  Each is a second-order polynomial of the
density anomaly in conservative temperature, absolute salinity, and geopotential
height Z::

    rho = rho_ref + R100*S + R010*T + R020*T^2 - R011*T*Z + R200*S^2
                  - R101*S*Z + R110*S*T

Coefficients ported verbatim from SeawaterPolynomials.jl.  Geopotential height is
taken as ``Z = -p`` (sea pressure in dbar ~ depth in m, positive down), matching
the convention of the 55-term ``teos10-poly55`` backend.  There is no standalone
Python reference for these, so they are validated structurally (exact analytic
derivatives checked against finite differences; literal check values).
"""

from ..conventions import TemperatureKind, SalinityKind, PressureUnit
from ..registry import EOSBackend, register

DEFAULT_REFERENCE_DENSITY = 1024.6  # kg m-3 (Oceananigans / Roquet default)

# (R100, R010, R020, R011, R200, R101, R110) for each coefficient set.
COEFFICIENTS = {
    "roquet-linear": dict(R100=7.718e-1, R010=-1.775e-1),
    "roquet-cabbeling": dict(R100=7.718e-1, R010=-0.844e-1, R020=-4.561e-3),
    "roquet-cabbeling-thermobaricity": dict(
        R100=7.718e-1, R010=-0.651e-1, R020=-5.027e-3, R011=-2.5681e-5),
    "roquet-freezing": dict(
        R100=7.718e-1, R010=-0.491e-1, R020=-5.027e-3, R011=-2.5681e-5),
    "roquet-second-order": dict(
        R100=8.078e-1, R010=0.182e-1, R020=-4.937e-3, R011=-2.4677e-5,
        R200=-1.115e-4, R101=-8.241e-6, R110=-2.446e-3),
    # Simplest-realistic (Roquet 2015 eq. 17): Cb=0.011, Th=2.5e-5, b0=0.77, T0=-4.5.
    # The constant R000 = -Cb*T0^2/2 (~-0.11 kg m-3) is omitted (no dynamical effect),
    # matching SeawaterPolynomials.jl; this offsets the absolute density by that constant.
    "roquet-simplest-realistic": dict(
        R100=0.77, R010=0.011 * -4.5, R020=-0.011 / 2, R011=-2.5e-5),
}


def make_roquet_idealized(eos_id, reference_density=DEFAULT_REFERENCE_DENSITY, **coeffs):
    c = {k: 0.0 for k in ("R100", "R010", "R020", "R011", "R200", "R101", "R110")}
    c.update(coeffs)
    R100, R010, R020 = c["R100"], c["R010"], c["R020"]
    R011, R200, R101, R110 = c["R011"], c["R200"], c["R101"], c["R110"]

    def density(t, s, p):
        z = -p  # geopotential height [m] ~ -(sea pressure in dbar)
        anomaly = (R100 * s + R010 * t + R020 * t * t - R011 * t * z
                   + R200 * s * s - R101 * s * z + R110 * s * t)
        return reference_density + anomaly

    def drho_dt(t, s, p):
        z = -p
        return R010 + 2.0 * R020 * t - R011 * z + R110 * s

    def drho_ds(t, s, p):
        z = -p
        return R100 + 2.0 * R200 * s - R101 * z + R110 * t

    return EOSBackend(
        id=eos_id,
        density=density,
        drho_dt=drho_dt,
        drho_ds=drho_ds,
        temperature=TemperatureKind.CONSERVATIVE,
        salinity=SalinityKind.ABSOLUTE,
        pressure_unit=PressureUnit.DBAR,
        reference="Roquet et al. (2015), J. Phys. Oceanogr., 45, Table 3.",
        description=f"Idealized Roquet EOS '{eos_id.split('-', 1)[1]}' "
                    f"(Oceananigans; rho_ref={reference_density}).",
    )


for _eos_id, _coeffs in COEFFICIENTS.items():
    register(make_roquet_idealized(_eos_id, **_coeffs))
