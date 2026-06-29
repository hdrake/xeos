"""UNESCO / EOS-80 equation of state (Fofonoff & Millard, 1983).

The international (UNESCO 1981) EOS-80 polynomial, as used by MOM6 ``UNESCO`` /
``JACKETT_MCD`` and MITgcm ``eosType='UNESCO'``.  It shares the surface-density
polynomial with JMD95 but uses a distinct secant bulk modulus.  Coefficients
ported from the MITgcm reference implementation (``MITgcmutils.density.unesco``).

State variables: potential temperature [degC], practical salinity [PSU], sea
pressure [dbar].
"""

import numpy as np

from ..conventions import TemperatureKind, SalinityKind, PressureUnit
from ..registry import EOSBackend, register
from ._jmd95 import _rho_surface  # shared EOS-80 surface density polynomial

# Secant bulk modulus coefficients (UNESCO/EOS-80; differ from JMD95).
_KFW = np.array([1.965221e04, 1.484206e02, -2.327105e00, 1.360477e-02, -5.155288e-05])
_KSW = np.array([5.467460e01, -0.603459e00, 1.099870e-02, -6.167000e-05,
                 7.944000e-02, 1.648300e-02, -5.300900e-04])
_KP = np.array([3.239908e00, 1.437130e-03, 1.160920e-04, -5.779050e-07,
                2.283800e-03, -1.098100e-05, -1.607800e-06, 1.910750e-04,
                8.509350e-05, -6.122930e-06, 5.278700e-08, -9.934800e-07,
                2.081600e-08, 9.169700e-10])


def _bulk_modulus(s, t, p_bar):
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    s3o2 = s * np.sqrt(s)
    p = p_bar
    p2 = p * p
    bm = _KFW[0] + _KFW[1] * t + _KFW[2] * t2 + _KFW[3] * t3 + _KFW[4] * t4
    bm = (bm
          + s * (_KSW[0] + _KSW[1] * t + _KSW[2] * t2 + _KSW[3] * t3)
          + s3o2 * (_KSW[4] + _KSW[5] * t + _KSW[6] * t2))
    bm = (bm
          + p * (_KP[0] + _KP[1] * t + _KP[2] * t2 + _KP[3] * t3)
          + p * s * (_KP[4] + _KP[5] * t + _KP[6] * t2)
          + p * s3o2 * _KP[7]
          + p2 * (_KP[8] + _KP[9] * t + _KP[10] * t2)
          + p2 * s * (_KP[11] + _KP[12] * t + _KP[13] * t2))
    return bm


def density(t, s, p_dbar):
    """In-situ density [kg m-3] from potential temp, practical salinity, dbar."""
    p_bar = 0.1 * p_dbar
    rho_s = _rho_surface(s, t)
    return rho_s / (1.0 - p_bar / _bulk_modulus(s, t, p_bar))


register(EOSBackend(
    id="unesco",
    density=density,
    temperature=TemperatureKind.POTENTIAL,
    salinity=SalinityKind.PRACTICAL,
    pressure_unit=PressureUnit.DBAR,
    reference="Fofonoff & Millard (1983), UNESCO Tech. Papers Mar. Sci. 44 (EOS-80).",
    description="UNESCO/EOS-80; MOM6 UNESCO/JACKETT_MCD, MITgcm UNESCO.",
))
