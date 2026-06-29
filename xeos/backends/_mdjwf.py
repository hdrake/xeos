"""McDougall, Jackett, Wright & Feistel (2003) equation of state.

McDougall, T.J., D.R. Jackett, D.G. Wright & R. Feistel (2003): "Accurate and
computationally efficient algorithms for potential temperature and density of
seawater." J. Atmos. Oceanic Technol., 20, 730-741.

A rational-function (numerator / denominator) fit; MITgcm ``eosType='MDJWF'``.
Coefficients ported from the MITgcm reference (``MITgcmutils.density.mdjwf``).

State variables: potential temperature [degC], practical salinity [PSU], sea
pressure [dbar].
"""

import numpy as np

from ..conventions import TemperatureKind, SalinityKind, PressureUnit
from ..registry import EOSBackend, register

_NUM = np.array([7.35212840e+00, -5.45928211e-02, 3.98476704e-04, 2.96938239e+00,
                 -7.23268813e-03, 2.12382341e-03, 1.04004591e-02, 1.03970529e-07,
                 5.18761880e-06, -3.24041825e-08, -1.23869360e-11, 9.99843699e+02])
_DEN = np.array([7.28606739e-03, -4.60835542e-05, 3.68390573e-07, 1.80809186e-10,
                 2.14691708e-03, -9.27062484e-06, -1.78343643e-10, 4.76534122e-06,
                 1.63410736e-09, 5.30848875e-06, -3.03175128e-16, -1.27934137e-17,
                 1.00000000e+00])


def density(t, s, p_dbar):
    """In-situ density [kg m-3] from potential temp, practical salinity, dbar."""
    t1 = t
    t2 = t1 * t1
    s1 = s
    p1 = p_dbar
    sp5 = np.sqrt(s1)
    p1t1 = p1 * t1

    num = (_NUM[11]
           + t1 * (_NUM[0] + t1 * (_NUM[1] + _NUM[2] * t1))
           + s1 * (_NUM[3] + _NUM[4] * t1 + _NUM[5] * s1)
           + p1 * (_NUM[6] + _NUM[7] * t2 + _NUM[8] * s1
                   + p1 * (_NUM[9] + _NUM[10] * t2)))
    den = (_DEN[12]
           + t1 * (_DEN[0] + t1 * (_DEN[1] + t1 * (_DEN[2] + t1 * _DEN[3])))
           + s1 * (_DEN[4] + t1 * (_DEN[5] + _DEN[6] * t2)
                   + sp5 * (_DEN[7] + _DEN[8] * t2))
           + p1 * (_DEN[9] + p1t1 * (_DEN[10] * t2 + _DEN[11] * p1)))
    return num / den


register(EOSBackend(
    id="mdjwf",
    density=density,
    temperature=TemperatureKind.POTENTIAL,
    salinity=SalinityKind.PRACTICAL,
    pressure_unit=PressureUnit.DBAR,
    reference="McDougall, Jackett, Wright & Feistel (2003), "
              "J. Atmos. Oceanic Technol., 20, 730-741.",
    description="Rational-function EOS; MITgcm MDJWF.",
))
