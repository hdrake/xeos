"""Build & run standalone drivers from MITgcm's own EOS Fortran (jmd95/unesco/mdjwf).

xeos's ``jmd95``, ``unesco`` and ``mdjwf`` kernels were ported from the MITgcm
reference. This generator validates them against MITgcm's *source* rather than the
``fastjmd95`` / ``MITgcmutils`` Python ports: it parses the authoritative
coefficients from MITgcm ``model/src/ini_eos.F`` (verbatim, including the JMD95 vs
UNESCO secant-bulk-modulus split and the MDJWF rational coefficients) and evaluates
the density with the formulas from ``model/src/find_rho.F``
(``FIND_RHOP0`` + ``FIND_BULKMOD`` for JMD95/UNESCO, ``FIND_RHONUM`` /
``FIND_RHODEN`` for MDJWF), compiled with gfortran.

The coefficients -- where cross-model drift and porting typos live -- come straight
from ``ini_eos.F`` (downloaded on demand, not committed; only the numbers go into
``truth.json``). The polynomial/rational forms are transcribed line-for-line from
``find_rho.F`` (cited inline). Before any output is trusted, each driver self-checks
its density at S=35, T=25, p=2000 dbar against a value computed independently, in
numpy, from the *same parsed coefficients* -- an independent code path, so a mangled
parse or a transcription slip fails loudly.

alpha/beta are produced by centred finite difference of the compiled density (the
same h=1e-3 the facade uses for these FD-fallback backends), validating xeos's own
FD path against an independent compiled implementation.

Notes from the MITgcm source:
  * MITgcm's UNESCO uses the *original* Fofonoff & Millard EOS-80 secant bulk
    modulus (distinct from JMD95) and shares the EOS-80 surface polynomial; ini_eos.F
    even prints a runtime WARNING that feeding it potential temperature (as MITgcm
    and xeos do) "can result in density errors of up to 5%". This is the EOS-80
    `unesco` lineage -- NOT MOM6's "UNESCO", which is the JMD95 fit (see
    _build_unesco_fortran.py).
  * JMD95 / UNESCO native pressure is bar (find_rho.F: p = locPres*SItoBar, 1e-5);
    MDJWF is dbar (p1 = locPres*SItodBar, 1e-4). We feed locPres in Pa = p_dbar*1e4.
"""

import math
import os
import re
import subprocess
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
INI_EOS_URL = "https://raw.githubusercontent.com/MITgcm/MITgcm/master/model/src/ini_eos.F"
_F = os.path.join(HERE, "ini_eos.F")  # gitignored

_H = 1.0e-3  # FD step; matches the facade's _DT/_DS

# Check point for the self-check gate (independent numpy eval, below).
_CHK_S, _CHK_T, _CHK_P = 35.0, 25.0, 2000.0


def _parse_array(region, name):
    """Parse ``name(i) = [-] mantissa _d exp`` assignments from a source region
    into {index: float}. Handles MITgcm's ``_d`` double macro and space-separated
    leading minus (e.g. ``- 9.095290 _d -03``)."""
    out = {}
    pat = re.compile(
        rf"{re.escape(name)}\(\s*(\d+)\s*\)\s*=\s*(-?)\s*([0-9.]+)\s*_d\s*([+-]?\s*\d+)")
    for idx, sign, mant, exp in pat.findall(region):
        out[int(idx)] = float(f"{sign}{mant}e{int(exp.replace(' ', ''))}")
    return out


def _slice(src, start_marker, end_marker):
    i = src.index(start_marker)
    j = src.index(end_marker, i + len(start_marker))
    return src[i:j]


def _coefficients():
    """Return parsed coefficient arrays for each eosType, straight from ini_eos.F."""
    if not os.path.exists(_F):
        urllib.request.urlretrieve(INI_EOS_URL, _F)
    src = open(_F).read()

    surf = _slice(src, "equationOfState .EQ. 'JMD95Z'",
                  "equationOfState(1:5) .EQ. 'JMD95'")
    jmd_bulk = _slice(src, "equationOfState(1:5) .EQ. 'JMD95'",
                      "equationOfState .EQ. 'UNESCO'")
    uns_bulk = _slice(src, "equationOfState .EQ. 'UNESCO'",
                      "INI_EOS: We should never")
    mdjwf = _slice(src, "equationOfState .EQ. 'MDJWF'",
                   "equationOfState .EQ. 'TEOS10'")

    cfw = _parse_array(surf, "eosJMDCFw")   # 1..6  surface fresh
    csw = _parse_array(surf, "eosJMDCSw")   # 1..9  surface salt
    common = {"CFw": [cfw[i] for i in range(1, 7)],
              "CSw": [csw[i] for i in range(1, 10)]}

    def bulk(region):
        kfw = _parse_array(region, "eosJMDCKFw")  # 1..5
        ksw = _parse_array(region, "eosJMDCKSw")  # 1..7
        kp = _parse_array(region, "eosJMDCKP")    # 1..14
        return {"CKFw": [kfw[i] for i in range(1, 6)],
                "CKSw": [ksw[i] for i in range(1, 8)],
                "CKP": [kp[i] for i in range(1, 15)]}

    num = _parse_array(mdjwf, "eosMDJWFnum")  # 0..11
    den = _parse_array(mdjwf, "eosMDJWFden")  # 0..12
    return {
        "jmd95": {**common, **bulk(jmd_bulk)},
        "unesco": {**common, **bulk(uns_bulk)},
        "mdjwf": {"num": [num[i] for i in range(0, 12)],
                  "den": [den[i] for i in range(0, 13)]},
    }


# ---- independent numpy reference (self-check gate only) ---------------------

def _np_secant(c, t, s, p_dbar):
    t2, t3, t4 = t * t, t * t * t, t * t * t * t
    s = max(s, 0.0)
    s3o2 = s * math.sqrt(s)
    p = p_dbar * 0.1            # bar  (= p_Pa * SItoBar = p_dbar*1e4*1e-5)
    p2 = p * p
    F, Sc, KF, KS, KP = c["CFw"], c["CSw"], c["CKFw"], c["CKSw"], c["CKP"]
    rho0 = (F[0] + F[1] * t + F[2] * t2 + F[3] * t3 + F[4] * t4 + F[5] * t4 * t
            + s * (Sc[0] + Sc[1] * t + Sc[2] * t2 + Sc[3] * t3 + Sc[4] * t4)
            + s3o2 * (Sc[5] + Sc[6] * t + Sc[7] * t2) + Sc[8] * s * s)
    bm = (KF[0] + KF[1] * t + KF[2] * t2 + KF[3] * t3 + KF[4] * t4
          + s * (KS[0] + KS[1] * t + KS[2] * t2 + KS[3] * t3)
          + s3o2 * (KS[4] + KS[5] * t + KS[6] * t2)
          + p * (KP[0] + KP[1] * t + KP[2] * t2 + KP[3] * t3)
          + p * s * (KP[4] + KP[5] * t + KP[6] * t2) + p * s3o2 * KP[7]
          + p2 * (KP[8] + KP[9] * t + KP[10] * t2)
          + p2 * s * (KP[11] + KP[12] * t + KP[13] * t2))
    return rho0 / (1.0 - p / bm)


def _np_mdjwf(c, t1, s1, p_dbar):
    t2 = t1 * t1
    s1 = max(s1, 0.0)
    sp5 = math.sqrt(s1)
    p1 = p_dbar                 # dbar  (= p_Pa * SItodBar = p_dbar*1e4*1e-4)
    p1t1 = p1 * t1
    n, d = c["num"], c["den"]
    num = (n[0] + t1 * (n[1] + t1 * (n[2] + n[3] * t1))
           + s1 * (n[4] + n[5] * t1 + n[6] * s1)
           + p1 * (n[7] + n[8] * t2 + n[9] * s1 + p1 * (n[10] + n[11] * t2)))
    den = (d[0] + t1 * (d[1] + t1 * (d[2] + t1 * (d[3] + t1 * d[4])))
           + s1 * (d[5] + t1 * (d[6] + d[7] * t2) + sp5 * (d[8] + d[9] * t2))
           + p1 * (d[10] + p1t1 * (d[11] * t2 + d[12] * p1)))
    return num / den


# ---- Fortran driver generation ---------------------------------------------

def _fmt(vals):
    # Python float repr -> Fortran real*8 literal: '6.5e-09'->'6.5d-09', '999.8'->'999.8d0'.
    def lit(v):
        r = repr(float(v))
        return r.replace("e", "d") if "e" in r else r + "d0"
    return ", ".join(lit(v) for v in vals)


def _secant_driver(c):
    """JMD95/UNESCO: FIND_RHOP0 + FIND_BULKMOD + rho=rhoP0/(1-p/bulk) (find_rho.F)."""
    return f"""program mitgcm_rho
  implicit none
  real*8 :: CFw(6)=(/{_fmt(c['CFw'])}/)
  real*8 :: CSw(9)=(/{_fmt(c['CSw'])}/)
  real*8 :: CKFw(5)=(/{_fmt(c['CKFw'])}/)
  real*8 :: CKSw(7)=(/{_fmt(c['CKSw'])}/)
  real*8 :: CKP(14)=(/{_fmt(c['CKP'])}/)
  real*8 :: S, T, pdbar, rho
  integer :: ios
  do
    read(*,*,iostat=ios) S, T, pdbar
    if (ios /= 0) exit
    call rho_secant(T, S, pdbar*1.0d4, rho)
    write(*,'(ES25.16)') rho
  end do
contains
  subroutine rho_secant(t_in, s_in, p_pa, rho)
    real*8, intent(in) :: t_in, s_in, p_pa
    real*8, intent(out) :: rho
    real*8 :: t, t2, t3, t4, s, s3o2, p, p2
    real*8 :: rfresh, rsalt, rhoP0, bMfresh, bMsalt, bMpres, bulk
    real*8, parameter :: SItoBar = 1.0d-5
    t = t_in; t2 = t*t; t3 = t2*t; t4 = t3*t
    s = s_in
    if (s .gt. 0.0d0) then
      s3o2 = s*sqrt(s)
    else
      s = 0.0d0; s3o2 = 0.0d0
    end if
    ! FIND_RHOP0 (find_rho.F)
    rfresh = CFw(1) + CFw(2)*t + CFw(3)*t2 + CFw(4)*t3 + CFw(5)*t4 + CFw(6)*t4*t
    rsalt = s*(CSw(1) + CSw(2)*t + CSw(3)*t2 + CSw(4)*t3 + CSw(5)*t4) &
          + s3o2*(CSw(6) + CSw(7)*t + CSw(8)*t2) + CSw(9)*s*s
    rhoP0 = rfresh + rsalt
    ! FIND_BULKMOD (find_rho.F)
    p = p_pa*SItoBar; p2 = p*p
    bMfresh = CKFw(1) + CKFw(2)*t + CKFw(3)*t2 + CKFw(4)*t3 + CKFw(5)*t4
    bMsalt = s*(CKSw(1) + CKSw(2)*t + CKSw(3)*t2 + CKSw(4)*t3) &
           + s3o2*(CKSw(5) + CKSw(6)*t + CKSw(7)*t2)
    bMpres = p*(CKP(1) + CKP(2)*t + CKP(3)*t2 + CKP(4)*t3) &
           + p*s*(CKP(5) + CKP(6)*t + CKP(7)*t2) + p*s3o2*CKP(8) &
           + p2*(CKP(9) + CKP(10)*t + CKP(11)*t2) &
           + p2*s*(CKP(12) + CKP(13)*t + CKP(14)*t2)
    bulk = bMfresh + bMsalt + bMpres
    ! density of sea water at pressure p (find_rho.F)
    rho = rhoP0/(1.0d0 - p/bulk)
  end subroutine
end program
"""


def _mdjwf_driver(c):
    """MDJWF: FIND_RHONUM / FIND_RHODEN, rho = num/den (find_rho.F)."""
    return f"""program mitgcm_rho
  implicit none
  real*8 :: num(0:11)=(/{_fmt(c['num'])}/)
  real*8 :: den(0:12)=(/{_fmt(c['den'])}/)
  real*8 :: S, T, pdbar, rho
  integer :: ios
  do
    read(*,*,iostat=ios) S, T, pdbar
    if (ios /= 0) exit
    call rho_mdjwf(T, S, pdbar*1.0d4, rho)
    write(*,'(ES25.16)') rho
  end do
contains
  subroutine rho_mdjwf(t_in, s_in, p_pa, rho)
    real*8, intent(in) :: t_in, s_in, p_pa
    real*8, intent(out) :: rho
    real*8 :: t1, t2, s1, sp5, p1, p1t1, rnum, rden
    real*8, parameter :: SItodBar = 1.0d-4
    t1 = t_in; t2 = t1*t1; s1 = s_in
    if (s1 .gt. 0.0d0) then
      sp5 = sqrt(s1)
    else
      s1 = 0.0d0; sp5 = 0.0d0
    end if
    p1 = p_pa*SItodBar; p1t1 = p1*t1
    ! FIND_RHONUM (find_rho.F)
    rnum = num(0) + t1*(num(1) + t1*(num(2) + num(3)*t1)) &
         + s1*(num(4) + num(5)*t1 + num(6)*s1) &
         + p1*(num(7) + num(8)*t2 + num(9)*s1 + p1*(num(10) + num(11)*t2))
    ! FIND_RHODEN (find_rho.F)
    rden = den(0) + t1*(den(1) + t1*(den(2) + t1*(den(3) + t1*den(4)))) &
         + s1*(den(5) + t1*(den(6) + den(7)*t2) + sp5*(den(8) + den(9)*t2)) &
         + p1*(den(10) + p1t1*(den(11)*t2 + den(12)*p1))
    rho = rnum/rden
  end subroutine
end program
"""


def gfortran_version():
    try:
        v = subprocess.run(["gfortran", "--version"], capture_output=True, text=True)
        return v.stdout.splitlines()[0]
    except (OSError, IndexError):
        return None


def _build(eos_id, source):
    driver = os.path.join(HERE, f"_mitgcm_{eos_id}_driver.f90")  # gitignored
    binary = os.path.join(HERE, f"_mitgcm_{eos_id}_driver")      # gitignored
    open(driver, "w").write(source)
    subprocess.run(["gfortran", "-O2", "-o", binary, driver], check=True)
    return binary


def _run(binary, s, t, p_dbar):
    inp = "".join(f"{si} {ti} {pi}\n" for si, ti, pi in zip(s, t, p_dbar))
    out = subprocess.run([binary], input=inp, capture_output=True, text=True, check=True)
    return [float(line) for line in out.stdout.strip().splitlines()]


def mitgcm_truth(s, t, p_dbar):
    """Return ``{eos_id: {rho, alpha, beta}}`` for jmd95/unesco/mdjwf from MITgcm's
    Fortran, or None if gfortran is unavailable. Each driver is self-checked against
    an independent numpy eval of the same parsed coefficients before being trusted."""
    if gfortran_version() is None:
        return None
    coeffs = _coefficients()
    drivers = {
        "jmd95": (_secant_driver(coeffs["jmd95"]),
                  lambda S, T, P: _np_secant(coeffs["jmd95"], T, S, P)),
        "unesco": (_secant_driver(coeffs["unesco"]),
                   lambda S, T, P: _np_secant(coeffs["unesco"], T, S, P)),
        "mdjwf": (_mdjwf_driver(coeffs["mdjwf"]),
                  lambda S, T, P: _np_mdjwf(coeffs["mdjwf"], T, S, P)),
    }
    s, t, p_dbar = list(s), list(t), list(p_dbar)
    n = len(s)
    out = {}
    for eos_id, (source, npref) in drivers.items():
        binary = _build(eos_id, source)
        check = _run(binary, [_CHK_S], [_CHK_T], [_CHK_P])[0]
        expect = npref(_CHK_S, _CHK_T, _CHK_P)
        assert abs(check - expect) < 1e-9 * expect, (
            f"MITgcm {eos_id} driver failed its check: rho={check!r} "
            f"(numpy ref {expect!r})")
        rho = _run(binary, s, t, p_dbar)
        rt_p = _run(binary, s, [ti + _H for ti in t], p_dbar)
        rt_m = _run(binary, s, [ti - _H for ti in t], p_dbar)
        rs_p = _run(binary, [si + _H for si in s], t, p_dbar)
        rs_m = _run(binary, [si - _H for si in s], t, p_dbar)
        out[eos_id] = {
            "rho": rho,
            "alpha": [-(rt_p[i] - rt_m[i]) / (2.0 * _H) / rho[i] for i in range(n)],
            "beta": [(rs_p[i] - rs_m[i]) / (2.0 * _H) / rho[i] for i in range(n)],
        }
    return out
