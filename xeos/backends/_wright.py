"""Wright (1997) equation of state.

Wright, D.G. (1997): "An equation of state for use in ocean models: Eckart's
formula revisited." J. Atmos. Oceanic Technol., 14, 735-740.

All Wright variants share one functional form and differ only in their fitted
coefficients.  ``wright97-reduced`` uses the original (reduced-range) Wright
coefficients used by MOM6 ``WRIGHT``/``WRIGHT_RED`` (and by ``momlevel``);
``wright97-full`` uses the bug-fixed full-range refit, MOM6 ``WRIGHT_FULL``.

State variables: potential temperature [degC], practical salinity [PSU],
pressure in pascals.
"""

from ..conventions import TemperatureKind, SalinityKind, PressureUnit
from ..registry import EOSBackend, register

# Original Wright-1997 reduced-range coefficients (MOM6 WRIGHT / WRIGHT_RED).
REDUCED = dict(
    A0=7.057924e-4, A1=3.480336e-7, A2=-1.112733e-7,
    B0=5.790749e8, B1=3.516535e6, B2=-4.002714e4, B3=2.084372e2,
    B4=5.944068e5, B5=-9.643486e3,
    C0=1.704853e5, C1=7.904722e2, C2=-7.984422, C3=5.140652e-2,
    C4=-2.302158e2, C5=-3.079464,
)

# Full-range bug-fixed refit (MOM6 WRIGHT_FULL).
FULL = dict(
    A0=7.133718e-4, A1=2.724670e-7, A2=-1.646582e-7,
    B0=5.613770e8, B1=3.600337e6, B2=-3.727194e4, B3=1.660557e2,
    B4=6.844158e5, B5=-8.389457e3,
    C0=1.609893e5, C1=8.427815e2, C2=-6.931554, C3=3.869318e-2,
    C4=-1.664201e2, C5=-2.765195,
)


def _make(coeffs):
    c = coeffs
    A0, A1, A2 = c["A0"], c["A1"], c["A2"]
    B0, B1, B2, B3, B4, B5 = c["B0"], c["B1"], c["B2"], c["B3"], c["B4"], c["B5"]
    C0, C1, C2, C3, C4, C5 = c["C0"], c["C1"], c["C2"], c["C3"], c["C4"], c["C5"]

    def _terms(t, s, p):
        al0 = A0 + A1 * t + A2 * s
        p0 = B0 + B4 * s + t * (B1 + t * (B2 + B3 * t) + B5 * s)
        lam = C0 + C4 * s + t * (C1 + t * (C2 + C3 * t) + C5 * s)
        pp = p + p0
        return al0, pp, lam

    def density(t, s, p):
        al0, pp, lam = _terms(t, s, p)
        return pp / (lam + al0 * pp)

    def drho_dt(t, s, p):
        al0, pp, lam = _terms(t, s, p)
        i2 = 1.0 / (lam + al0 * pp) ** 2
        return i2 * (
            lam * (B1 + t * (2.0 * B2 + 3.0 * B3 * t) + B5 * s)
            - pp * (pp * A1 + (C1 + t * (2.0 * C2 + 3.0 * C3 * t) + C5 * s))
        )

    def drho_ds(t, s, p):
        al0, pp, lam = _terms(t, s, p)
        i2 = 1.0 / (lam + al0 * pp) ** 2
        return i2 * (lam * (B4 + B5 * t) - pp * (pp * A2 + (C4 + C5 * t)))

    return density, drho_dt, drho_ds


def _backend(eos_id, coeffs, label):
    density, drho_dt, drho_ds = _make(coeffs)
    return EOSBackend(
        id=eos_id,
        density=density,
        drho_dt=drho_dt,
        drho_ds=drho_ds,
        temperature=TemperatureKind.POTENTIAL,
        salinity=SalinityKind.PRACTICAL,
        pressure_unit=PressureUnit.PASCAL,
        reference="Wright (1997), J. Atmos. Oceanic Technol., 14, 735-740.",
        description=label,
    )


register(_backend("wright97-full", FULL, "Wright 1997, full range (MOM6 WRIGHT_FULL)."))
register(_backend("wright97-reduced", REDUCED,
                  "Wright 1997, reduced range (MOM6 WRIGHT/WRIGHT_RED)."))
