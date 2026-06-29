"""Jackett & McDougall (1995) equation of state (the MITgcm / ECCO standard).

Jackett, D.R. & T.J. McDougall (1995): "Minimal adjustment of hydrographic
profiles to achieve static stability." J. Atmos. Oceanic Technol., 12, 381-389.

Coefficients ported verbatim from the MITgcm reference implementation (also used
by ``fastjmd95``).  This same fit is what MOM6 calls ``UNESCO`` / ``JACKETT_MCD``
(``MOM_EOS_UNESCO.F90``), so those MOM6 selectors resolve here too.  State
variables: potential temperature [degC], practical
salinity [PSU], sea pressure [dbar].  Density derivatives are left to the
facade's centred finite-difference fallback (the reference analytic derivatives
carry a known typo); the fallback is accurate to ~1e-6 (O(h^2) truncation).
"""

import numpy as np

from ..conventions import TemperatureKind, SalinityKind, PressureUnit
from ..registry import EOSBackend, register

# Density of fresh / sea water at p = 0.
_CFW = np.array([999.842594, 6.793952e-02, -9.095290e-03,
                 1.001685e-04, -1.120083e-06, 6.536332e-09])
_CSW = np.array([8.244930e-01, -4.089900e-03, 7.643800e-05, -8.246700e-07,
                 5.387500e-09, -5.724660e-03, 1.022700e-04, -1.654600e-06,
                 4.831400e-04])
# Secant bulk modulus coefficients.
_CKFW = np.array([1.965933e04, 1.444304e02, -1.706103e00,
                  9.648704e-03, -4.190253e-05])
_CKSW = np.array([5.284855e01, -3.101089e-01, 6.283263e-03, -5.084188e-05,
                  3.886640e-01, 9.085835e-03, -4.619924e-04])
_CKP = np.array([3.186519e00, 2.212276e-02, -2.984642e-04, 1.956415e-06,
                 6.704388e-03, -1.847318e-04, 2.059331e-07, 1.480266e-04,
                 2.102898e-04, -1.202016e-05, 1.394680e-07, -2.040237e-06,
                 6.128773e-08, 6.207323e-10])


def _rho_surface(s, t):
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    s3o2 = s * np.sqrt(s)
    rho_fw = (_CFW[0] + _CFW[1] * t + _CFW[2] * t2 + _CFW[3] * t3
              + _CFW[4] * t4 + _CFW[5] * t4 * t)
    return (rho_fw
            + s * (_CSW[0] + _CSW[1] * t + _CSW[2] * t2 + _CSW[3] * t3 + _CSW[4] * t4)
            + s3o2 * (_CSW[5] + _CSW[6] * t + _CSW[7] * t2)
            + _CSW[8] * s * s)


def _bulk_modulus(s, t, p_bar):
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    s3o2 = s * np.sqrt(s)
    p = p_bar
    p2 = p * p
    bm = _CKFW[0] + _CKFW[1] * t + _CKFW[2] * t2 + _CKFW[3] * t3 + _CKFW[4] * t4
    bm = (bm
          + s * (_CKSW[0] + _CKSW[1] * t + _CKSW[2] * t2 + _CKSW[3] * t3)
          + s3o2 * (_CKSW[4] + _CKSW[5] * t + _CKSW[6] * t2))
    bm = (bm
          + p * (_CKP[0] + _CKP[1] * t + _CKP[2] * t2 + _CKP[3] * t3)
          + p * s * (_CKP[4] + _CKP[5] * t + _CKP[6] * t2)
          + p * s3o2 * _CKP[7]
          + p2 * (_CKP[8] + _CKP[9] * t + _CKP[10] * t2)
          + p2 * s * (_CKP[11] + _CKP[12] * t + _CKP[13] * t2))
    return bm


def density(t, s, p_dbar):
    """In-situ density [kg m-3] from potential temp, practical salinity, dbar."""
    p_bar = 0.1 * p_dbar
    rho_s = _rho_surface(s, t)
    bulk = _bulk_modulus(s, t, p_bar)
    return rho_s / (1.0 - p_bar / bulk)


register(EOSBackend(
    id="jmd95",
    density=density,
    temperature=TemperatureKind.POTENTIAL,
    salinity=SalinityKind.PRACTICAL,
    pressure_unit=PressureUnit.DBAR,
    reference="Jackett & McDougall (1995), J. Atmos. Oceanic Technol., 12, 381-389.",
    description="UNESCO/EOS-80 refit; MITgcm JMD95Z/JMD95P, ECCO standard.",
))
