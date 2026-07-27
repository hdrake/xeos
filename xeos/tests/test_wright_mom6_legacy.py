"""``wright97-reduced`` vs MOM6's *legacy* ``MOM_EOS_Wright.F90``.

MOM6 ships three Wright kernels. The reference-truth harness in ``reference/``
already checks ``wright97-reduced`` against the *re-associated* reduced kernel
``MOM_EOS_Wright_red.F90`` (MOM6 ``WRIGHT_RED``/``WRIGHT_REDUCED``). But the bare
``EQN_OF_STATE = "WRIGHT"`` selector runs the older ``MOM_EOS_Wright.F90``, whose
density expression uses the same coefficients but a different (left-to-right)
addition order.

This test pins the relationship the selector note relies on: ``wright97-reduced``
uses that legacy addition order, so it reproduces ``MOM_EOS_Wright.F90`` density
*bit-for-bit* in float64, and matches ``MOM_EOS_Wright_red.F90`` to round-off.
Both are far below the gsw/TEOS-10-vs-Wright difference (O(0.01-0.1 kg/m^3)) that
motivates matching a run's own EOS.
"""

import numpy as np

import xeos

# Reduced-range Wright (1997) coefficients (identical in MOM_EOS_Wright.F90 and
# MOM_EOS_Wright_red.F90).
_C = dict(
    A0=7.057924e-4, A1=3.480336e-7, A2=-1.112733e-7,
    B0=5.790749e8, B1=3.516535e6, B2=-4.002714e4, B3=2.084372e2,
    B4=5.944068e5, B5=-9.643486e3,
    C0=1.704853e5, C1=7.904722e2, C2=-7.984422, C3=5.140652e-2,
    C4=-2.302158e2, C5=-3.079464,
)


def _rho_legacy(t, s, p_pa):
    """density_elem_buggy_Wright grouping from MOM6 MOM_EOS_Wright.F90 (p in Pa)."""
    c = _C
    al0 = (c["A0"] + c["A1"] * t) + c["A2"] * s
    p0 = (c["B0"] + c["B4"] * s) + t * (c["B1"] + t * (c["B2"] + c["B3"] * t) + c["B5"] * s)
    lam = (c["C0"] + c["C4"] * s) + t * (c["C1"] + t * (c["C2"] + c["C3"] * t) + c["C5"] * s)
    return (p_pa + p0) / (lam + al0 * (p_pa + p0))


def _rho_reassociated(t, s, p_pa):
    """density_elem_Wright_red grouping from MOM6 MOM_EOS_Wright_red.F90 (p in Pa)."""
    c = _C
    al0 = c["A0"] + (c["A1"] * t + c["A2"] * s)
    p0 = c["B0"] + (c["B4"] * s + t * (c["B1"] + (t * (c["B2"] + c["B3"] * t) + c["B5"] * s)))
    lam = c["C0"] + (c["C4"] * s + t * (c["C1"] + (t * (c["C2"] + c["C3"] * t) + c["C5"] * s)))
    return (p_pa + p0) / (lam + al0 * (p_pa + p0))


def _grid():
    t = np.linspace(-2.0, 35.0, 50)[:, None, None]
    s = np.linspace(2.0, 38.0, 40)[None, :, None]
    p_dbar = np.linspace(0.0, 6000.0, 30)[None, None, :]
    t, s, p_dbar = np.broadcast_arrays(t, s, p_dbar)
    return t, s, p_dbar


def test_reduced_matches_mom6_legacy_bit_for_bit():
    eos = xeos.from_model("MOM6", "WRIGHT_REDUCED")  # warning-free alias
    t, s, p_dbar = _grid()
    xe = np.asarray(eos.rho(t, s, p_dbar))
    legacy = _rho_legacy(t, s, p_dbar * 1e4)  # kernel is native Pa
    assert np.array_equal(xe, legacy)  # identical coefficients + addition order


def test_reduced_matches_mom6_reassociated_to_roundoff():
    eos = xeos.from_model("MOM6", "WRIGHT_REDUCED")
    t, s, p_dbar = _grid()
    xe = np.asarray(eos.rho(t, s, p_dbar))
    reassoc = _rho_reassociated(t, s, p_dbar * 1e4)
    assert np.max(np.abs(xe - reassoc)) < 1e-11  # observed ~4.5e-13 kg/m^3
