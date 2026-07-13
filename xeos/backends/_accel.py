"""Optional numba-accelerated scalar density kernels (off by default).

The vendored numpy density kernels already run at C speed for large arrays, but a
``numba.vectorize`` scalar ufunc fuses the whole polynomial evaluation into a
single pass over memory (fewer temporaries, lower peak RAM, optional threading)
and enables point-wise inner loops.  This module provides a *scalar* re-expression
of each hot density kernel (jmd95, unesco, mdjwf, wright full/reduced, the 55-term
Roquet density, and roquet-spv) and wires them onto the corresponding backends as
:attr:`~xeos.registry.EOSBackend.density_fast`.

**No hard numba import happens at module load.**  Each scalar kernel is a plain
Python function that *reuses the exact coefficient constants already defined in its
backend module* (referenced by attribute, so only the arithmetic is re-expressed,
not the numbers).  The ``numba.vectorize`` ufunc is built lazily — the first time
an accelerated density is actually evaluated — by :class:`_LazyUfunc`.  Importing
this module therefore never imports numba, preserving the lightweight-core
invariant; numba is imported only when a facade built with ``accelerate=True`` is
first *called*.  ``mpas-jm`` / ``mpas-wright`` reuse the jmd95 / wright scalar
kernels, so they inherit acceleration exactly.
"""

import math

from . import _jmd95 as _jm
from . import _unesco as _un
from . import _mdjwf as _md
from . import _wright as _wr
from . import _roquet as _rq
from . import _roquet_spv as _spv
from ..registry import get_backend

__all__ = ["attach_fast_kernels", "numba_available", "FAST_BACKENDS"]

#: signature every scalar kernel is compiled against.
_SIG = "float64(float64, float64, float64)"


def numba_available() -> bool:
    """True if numba can be imported (without importing it here when False)."""
    import importlib.util
    return importlib.util.find_spec("numba") is not None


class _LazyUfunc:
    """Wrap a scalar kernel; compile a ``numba.vectorize`` ufunc on first call.

    Keeps numba out of import time entirely: numba is imported (and the JIT runs)
    only when the accelerated density is first evaluated.  The compiled ufunc is
    cached on the instance thereafter.
    """

    def __init__(self, scalar_kernel, target="cpu"):
        self._scalar = scalar_kernel
        self._target = target
        self._ufunc = None

    def __call__(self, *args):
        if self._ufunc is None:
            import numba  # imported lazily; ImportError propagates if absent
            self._ufunc = numba.vectorize([_SIG], target=self._target)(self._scalar)
        return self._ufunc(*args)


# --------------------------------------------------------------------------- #
# Scalar density kernels.  Each mirrors, term-for-term, the numpy ``density``
# in its backend module, referencing the same coefficient arrays/constants by
# attribute (so the numbers are reused verbatim) and using ``math.sqrt`` in place
# of ``np.sqrt``.
# --------------------------------------------------------------------------- #

def _jmd95_scalar(t, s, p_dbar):
    p_bar = 0.1 * p_dbar
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    s3o2 = s * math.sqrt(s)
    rho_fw = (_jm._CFW[0] + _jm._CFW[1] * t + _jm._CFW[2] * t2 + _jm._CFW[3] * t3
              + _jm._CFW[4] * t4 + _jm._CFW[5] * t4 * t)
    rho_s = (rho_fw
             + s * (_jm._CSW[0] + _jm._CSW[1] * t + _jm._CSW[2] * t2
                    + _jm._CSW[3] * t3 + _jm._CSW[4] * t4)
             + s3o2 * (_jm._CSW[5] + _jm._CSW[6] * t + _jm._CSW[7] * t2)
             + _jm._CSW[8] * s * s)
    p = p_bar
    p2 = p * p
    bm = (_jm._CKFW[0] + _jm._CKFW[1] * t + _jm._CKFW[2] * t2
          + _jm._CKFW[3] * t3 + _jm._CKFW[4] * t4)
    bm = (bm
          + s * (_jm._CKSW[0] + _jm._CKSW[1] * t + _jm._CKSW[2] * t2 + _jm._CKSW[3] * t3)
          + s3o2 * (_jm._CKSW[4] + _jm._CKSW[5] * t + _jm._CKSW[6] * t2))
    bm = (bm
          + p * (_jm._CKP[0] + _jm._CKP[1] * t + _jm._CKP[2] * t2 + _jm._CKP[3] * t3)
          + p * s * (_jm._CKP[4] + _jm._CKP[5] * t + _jm._CKP[6] * t2)
          + p * s3o2 * _jm._CKP[7]
          + p2 * (_jm._CKP[8] + _jm._CKP[9] * t + _jm._CKP[10] * t2)
          + p2 * s * (_jm._CKP[11] + _jm._CKP[12] * t + _jm._CKP[13] * t2))
    return rho_s / (1.0 - p_bar / bm)


def _unesco_scalar(t, s, p_dbar):
    # Shares the JMD95 surface-density polynomial (_jm._CFW/_CSW), own bulk modulus.
    p_bar = 0.1 * p_dbar
    t2 = t * t
    t3 = t2 * t
    t4 = t3 * t
    s3o2 = s * math.sqrt(s)
    rho_fw = (_jm._CFW[0] + _jm._CFW[1] * t + _jm._CFW[2] * t2 + _jm._CFW[3] * t3
              + _jm._CFW[4] * t4 + _jm._CFW[5] * t4 * t)
    rho_s = (rho_fw
             + s * (_jm._CSW[0] + _jm._CSW[1] * t + _jm._CSW[2] * t2
                    + _jm._CSW[3] * t3 + _jm._CSW[4] * t4)
             + s3o2 * (_jm._CSW[5] + _jm._CSW[6] * t + _jm._CSW[7] * t2)
             + _jm._CSW[8] * s * s)
    p = p_bar
    p2 = p * p
    bm = (_un._KFW[0] + _un._KFW[1] * t + _un._KFW[2] * t2
          + _un._KFW[3] * t3 + _un._KFW[4] * t4)
    bm = (bm
          + s * (_un._KSW[0] + _un._KSW[1] * t + _un._KSW[2] * t2 + _un._KSW[3] * t3)
          + s3o2 * (_un._KSW[4] + _un._KSW[5] * t + _un._KSW[6] * t2))
    bm = (bm
          + p * (_un._KP[0] + _un._KP[1] * t + _un._KP[2] * t2 + _un._KP[3] * t3)
          + p * s * (_un._KP[4] + _un._KP[5] * t + _un._KP[6] * t2)
          + p * s3o2 * _un._KP[7]
          + p2 * (_un._KP[8] + _un._KP[9] * t + _un._KP[10] * t2)
          + p2 * s * (_un._KP[11] + _un._KP[12] * t + _un._KP[13] * t2))
    return rho_s / (1.0 - p_bar / bm)


def _mdjwf_scalar(t, s, p_dbar):
    t1 = t
    t2 = t1 * t1
    s1 = s
    p1 = p_dbar
    sp5 = math.sqrt(s1)
    p1t1 = p1 * t1
    num = (_md._NUM[11]
           + t1 * (_md._NUM[0] + t1 * (_md._NUM[1] + _md._NUM[2] * t1))
           + s1 * (_md._NUM[3] + _md._NUM[4] * t1 + _md._NUM[5] * s1)
           + p1 * (_md._NUM[6] + _md._NUM[7] * t2 + _md._NUM[8] * s1
                   + p1 * (_md._NUM[9] + _md._NUM[10] * t2)))
    den = (_md._DEN[12]
           + t1 * (_md._DEN[0] + t1 * (_md._DEN[1] + t1 * (_md._DEN[2] + t1 * _md._DEN[3])))
           + s1 * (_md._DEN[4] + t1 * (_md._DEN[5] + _md._DEN[6] * t2)
                   + sp5 * (_md._DEN[7] + _md._DEN[8] * t2))
           + p1 * (_md._DEN[9] + p1t1 * (_md._DEN[10] * t2 + _md._DEN[11] * p1)))
    return num / den


def _make_wright_scalar(coeffs):
    """Build a scalar Wright kernel; ``coeffs`` is the backend's REDUCED/FULL dict.

    Captures the (reused) coefficients as closure constants so numba freezes them.
    """
    c = coeffs
    A0, A1, A2 = c["A0"], c["A1"], c["A2"]
    B0, B1, B2, B3, B4, B5 = c["B0"], c["B1"], c["B2"], c["B3"], c["B4"], c["B5"]
    C0, C1, C2, C3, C4, C5 = c["C0"], c["C1"], c["C2"], c["C3"], c["C4"], c["C5"]

    def _wright_scalar(t, s, p):
        al0 = A0 + A1 * t + A2 * s
        p0 = B0 + B4 * s + t * (B1 + t * (B2 + B3 * t) + B5 * s)
        lam = C0 + C4 * s + t * (C1 + t * (C2 + C3 * t) + C5 * s)
        pp = p + p0
        return pp / (lam + al0 * pp)

    return _wright_scalar


def _roquet_scalar(ct, sa, p_dbar):
    ss = math.sqrt((sa + _rq._dS) / _rq._SAu)
    tt = ct / _rq._CTu
    pp = p_dbar / _rq._Zu
    R00 = _rq._R0[0]
    R01 = _rq._R0[1]
    R02 = _rq._R0[2]
    R03 = _rq._R0[3]
    R04 = _rq._R0[4]
    R05 = _rq._R0[5]
    r0 = (((((R05 * pp + R04) * pp + R03) * pp + R02) * pp + R01) * pp + R00) * pp
    rz3 = _rq.R013 * tt + _rq.R103 * ss + _rq.R003
    rz2 = ((_rq.R022 * tt + _rq.R112 * ss + _rq.R012) * tt
           + (_rq.R202 * ss + _rq.R102) * ss + _rq.R002)
    rz1 = ((((_rq.R041 * tt + _rq.R131 * ss + _rq.R031) * tt
             + (_rq.R221 * ss + _rq.R121) * ss + _rq.R021) * tt
            + ((_rq.R311 * ss + _rq.R211) * ss + _rq.R111) * ss + _rq.R011) * tt
           + (((_rq.R401 * ss + _rq.R301) * ss + _rq.R201) * ss + _rq.R101) * ss + _rq.R001)
    rz0 = ((((((_rq.R060 * tt + _rq.R150 * ss + _rq.R050) * tt
               + (_rq.R240 * ss + _rq.R140) * ss + _rq.R040) * tt
              + ((_rq.R330 * ss + _rq.R230) * ss + _rq.R130) * ss + _rq.R030) * tt
             + (((_rq.R420 * ss + _rq.R320) * ss + _rq.R220) * ss + _rq.R120) * ss
             + _rq.R020) * tt
            + ((((_rq.R510 * ss + _rq.R410) * ss + _rq.R310) * ss + _rq.R210) * ss
               + _rq.R110) * ss + _rq.R010) * tt
           + (((((_rq.R600 * ss + _rq.R500) * ss + _rq.R400) * ss + _rq.R300) * ss
               + _rq.R200) * ss + _rq.R100) * ss + _rq.R000)
    anomaly = ((rz3 * pp + rz2) * pp + rz1) * pp + rz0
    return r0 + anomaly


def _roquet_spv_scalar(ct, sa, p_dbar):
    ss = math.sqrt((sa + _spv._dS) / _spv._SAu)
    tt = ct / _spv._CTu
    pp = p_dbar / _spv._Zu
    V00 = _spv._V0[0]
    V01 = _spv._V0[1]
    V02 = _spv._V0[2]
    V03 = _spv._V0[3]
    V04 = _spv._V0[4]
    V05 = _spv._V0[5]
    v0 = (((((V05 * pp + V04) * pp + V03) * pp + V02) * pp + V01) * pp + V00) * pp
    vp3 = _spv.V013 * tt + _spv.V103 * ss + _spv.V003
    vp2 = ((_spv.V022 * tt + _spv.V112 * ss + _spv.V012) * tt
           + (_spv.V202 * ss + _spv.V102) * ss + _spv.V002)
    vp1 = ((((_spv.V041 * tt + _spv.V131 * ss + _spv.V031) * tt
             + (_spv.V221 * ss + _spv.V121) * ss + _spv.V021) * tt
            + ((_spv.V311 * ss + _spv.V211) * ss + _spv.V111) * ss + _spv.V011) * tt
           + (((_spv.V401 * ss + _spv.V301) * ss + _spv.V201) * ss + _spv.V101) * ss
           + _spv.V001)
    vp0 = ((((((_spv.V060 * tt + _spv.V150 * ss + _spv.V050) * tt
               + (_spv.V240 * ss + _spv.V140) * ss + _spv.V040) * tt
              + ((_spv.V330 * ss + _spv.V230) * ss + _spv.V130) * ss + _spv.V030) * tt
             + (((_spv.V420 * ss + _spv.V320) * ss + _spv.V220) * ss + _spv.V120) * ss
             + _spv.V020) * tt
            + ((((_spv.V510 * ss + _spv.V410) * ss + _spv.V310) * ss + _spv.V210) * ss
               + _spv.V110) * ss + _spv.V010) * tt
           + (((((_spv.V600 * ss + _spv.V500) * ss + _spv.V400) * ss + _spv.V300) * ss
               + _spv.V200) * ss + _spv.V100) * ss + _spv.V000)
    delta = ((vp3 * pp + vp2) * pp + vp1) * pp + vp0
    return 1.0 / (v0 + delta)


# eos_id -> scalar kernel.  mpas-jm / mpas-wright reuse the jmd95 / wright kernels.
_wright_full_scalar = _make_wright_scalar(_wr.FULL)
_wright_reduced_scalar = _make_wright_scalar(_wr.REDUCED)

FAST_KERNELS = {
    "jmd95": _jmd95_scalar,
    "unesco": _unesco_scalar,
    "mdjwf": _mdjwf_scalar,
    "wright97-full": _wright_full_scalar,
    "wright97-reduced": _wright_reduced_scalar,
    "teos10-poly55": _roquet_scalar,
    "roquet-spv": _roquet_spv_scalar,
    "mpas-jm": _jmd95_scalar,             # byte-identical to jmd95
    "mpas-wright": _wright_reduced_scalar,  # byte-identical to wright97-reduced
}

#: ids that gain a fast path (public, e.g. for tests/introspection).
FAST_BACKENDS = tuple(FAST_KERNELS)


def attach_fast_kernels():
    """Populate ``density_fast`` (a lazy numba ufunc) on each accelerable backend.

    Uses ``object.__setattr__`` because :class:`EOSBackend` is a frozen dataclass.
    Never imports numba (the ufunc is compiled lazily on first evaluation).
    """
    for eos_id, scalar in FAST_KERNELS.items():
        backend = get_backend(eos_id)
        object.__setattr__(backend, "density_fast", _LazyUfunc(scalar))
