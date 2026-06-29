"""Build & run standalone drivers from MPAS-Ocean's authoritative EOS Fortran.

MPAS-O selects its equation of state with the namelist option ``config_eos_type``,
which accepts exactly three values -- ``linear``, ``jm`` (Jackett & McDougall 1995)
and ``wright`` (Wright 1997) -- implemented in
``components/mpas-ocean/src/shared/mpas_ocn_equation_of_state_{linear,jm,wright}.F``.
xeos's ``mpas-jm`` / ``mpas-wright`` backends *reuse* the existing ``jmd95`` /
``wright97-reduced`` kernels (the coefficients are byte-for-byte identical); this
module validates that reuse against MPAS-O's own compiled Fortran, the same way
``_build_roquet_spv_fortran.py`` validates ``roquet-spv`` against MOM6's Fortran.

This module downloads the three ``.F`` files (not committed -- only the resulting
numbers go into ``truth.json``), extracts each file's ``real (kind=RKIND),
parameter ::`` coefficient blocks and its core density arithmetic into a
self-contained driver program, compiles each with gfortran, and evaluates them on
a grid. ``alpha``/``beta`` are produced by centred finite difference of the
MPAS-source density (mirroring ``generate_truth.py``'s ``fd_alpha_beta``), so xeos's
own derivative path is checked against an independent implementation.

Two MPAS-O specifics are deliberately *not* reproduced here, matching xeos's
backends (which take pressure as an input and do not clamp): the T/S clamping and
the depth->pressure parameterisations. The validation grid is in-range, so clamping
would be a no-op for density anyway; omitting it keeps the finite-difference
derivatives one-to-one with xeos's unclamped kernels at the grid's range boundaries.

A documented check value is asserted for each driver after the build (e.g. jm gives
the standard UNESCO/EOS-80 surface density rho(theta=25, S=35, p=0) = 1023.343...
kg m-3), so silent breakage from upstream reformatting fails loudly.
"""

import os
import re
import subprocess
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = ("https://raw.githubusercontent.com/E3SM-Project/E3SM/master/"
         "components/mpas-ocean/src/shared/mpas_ocn_equation_of_state_")

# (driver tag -> EOS .F basename); files are gitignored, downloaded on demand.
_SRC = {"jm": "mpas_ocn_equation_of_state_jm.F",
        "wright": "mpas_ocn_equation_of_state_wright.F",
        "linear": "mpas_ocn_equation_of_state_linear.F"}


def _path(name):
    return os.path.join(HERE, _SRC[name])


def _nocomment(line):
    """Strip a Fortran inline comment; the continuation '&' precedes any '!'."""
    return re.sub(r"!.*$", "", line).rstrip()


def _params(src):
    """Join every ``real (kind=RKIND), parameter ::`` block (with its '&'
    continuation lines), comments stripped -- the verbatim EOS coefficients."""
    lines = [_nocomment(l) for l in src.splitlines()]
    out, i = [], 0
    while i < len(lines):
        if re.match(r"\s*real\s*\(kind=RKIND\)\s*,\s*parameter\s*::", lines[i]):
            block = [lines[i]]
            while block[-1].endswith("&"):
                i += 1
                block.append(lines[i])
            out.append("\n".join(block))
        i += 1
    return "\n".join("  " + l for l in "\n".join(out).splitlines())


def _slice(src, start_re, end_re):
    """Return the comment-stripped lines from the first match of ``start_re`` to the
    end of the statement that first matches ``end_re`` (following '&' continuations)."""
    lines = [_nocomment(l) for l in src.splitlines()]
    s = next(i for i, l in enumerate(lines) if re.search(start_re, l))
    e = next(i for i, l in enumerate(lines) if i >= s and re.search(end_re, l))
    while lines[e].endswith("&"):
        e += 1
    return "\n".join(lines[s:e + 1])


def _driver_jm(src):
    params = _params(src)
    arith = _slice(src, r"sqr\s*=\s*sqrt\(sq\)", r"density\(k,iCell\)")
    arith = arith.replace("p2(k)", "(pbar*pbar)").replace("p(k)", "pbar")
    arith = re.sub(r"density\(k,iCell\)", "rho", arith)
    return f"""program mpas_jm_truth
  implicit none
  integer, parameter :: RKIND = 8
{params}
  real(kind=RKIND) :: T, S, pdbar, pbar, rho, tq, sq, sqr, t2
  real(kind=RKIND) :: work1, work2, work3, work4, rhosfc, bulkMod
  integer :: ios
  do
    read(*,*,iostat=ios) T, S, pdbar
    if (ios /= 0) exit
    tq = T            ! MPAS clamps tq,sq to its valid range; the grid is in-range,
    sq = S            ! so density is unchanged and FD derivatives stay two-sided.
    pbar = 0.1_RKIND * pdbar          ! jm bulk modulus takes pressure in bars
{arith}
    write(*,'(ES25.16)') rho
  end do
end program
"""


def _driver_wright(src):
    params = _params(src)
    arith = _slice(src, r"T2\s*=\s*T\*T", r"density\(k, ?iCell\)\s*=\s*\(p \+ p0\)")
    arith = re.sub(r"density\(k, ?iCell\)", "rho", arith)
    return f"""program mpas_wright_truth
  implicit none
  integer, parameter :: RKIND = 8
{params}
  real(kind=RKIND) :: T, S, pdbar, p, rho, T2, T3, p0, lambda0, alpha0
  integer :: ios
  do
    read(*,*,iostat=ios) T, S, pdbar
    if (ios /= 0) exit
    p = 1.0e4_RKIND * pdbar           ! Wright (1997) coefficients take pressure in Pa
{arith}
    write(*,'(ES25.16)') rho
  end do
end program
"""


def _driver_linear(src):
    # The linear coefficients are namelist values (MPAS-O Registry.xml defaults:
    # config_eos_linear_densityref/alpha/beta/Tref/Sref), not Fortran parameters;
    # only the algebraic form lives in the .F, which we extract verbatim.
    arith = _slice(src,
                   r"density\(k,iCell\) = ocnEqStateLinearRhoRef",
                   r"ocnEqStateLinearSref\)")
    arith = (arith
             .replace("tracers(indexT,k,iCell)", "T")
             .replace("tracers(indexS,k,iCell)", "S")
             .replace("density(k,iCell)", "rho"))
    return f"""program mpas_linear_truth
  implicit none
  integer, parameter :: RKIND = 8
  real(kind=RKIND), parameter :: ocnEqStateLinearRhoRef = 1000.0_RKIND
  real(kind=RKIND), parameter :: ocnEqStateLinearAlpha  = 0.2_RKIND
  real(kind=RKIND), parameter :: ocnEqStateLinearBeta   = 0.8_RKIND
  real(kind=RKIND), parameter :: ocnEqStateLinearTref   = 5.0_RKIND
  real(kind=RKIND), parameter :: ocnEqStateLinearSref   = 35.0_RKIND
  real(kind=RKIND) :: T, S, pdbar, rho
  integer :: ios
  do
    read(*,*,iostat=ios) T, S, pdbar
    if (ios /= 0) exit
{arith}
    write(*,'(ES25.16)') rho
  end do
end program
"""


_DRIVERS = {"jm": _driver_jm, "wright": _driver_wright, "linear": _driver_linear}
# Published / cross-validated check value: rho at (theta=25 degC, S=35 PSU, p=0 dbar).
# jm reproduces the standard UNESCO/EOS-80 surface density 1023.343 kg m-3.
_CHECK = {"jm": 1023.3430584772268,
          "wright": 1023.3439704578486,
          "linear": 996.0}


def _bin(name):
    return os.path.join(HERE, f"_mpas_{name}_driver")


def _build():
    for name in _SRC:
        if not os.path.exists(_path(name)):
            urllib.request.urlretrieve(_BASE + name + ".F", _path(name))
        src = open(_path(name)).read()
        driver = os.path.join(HERE, f"_mpas_{name}_driver.f90")
        open(driver, "w").write(_DRIVERS[name](src))
        subprocess.run(["gfortran", "-fdefault-real-8", "-fdefault-double-8", "-O2",
                        "-o", _bin(name), driver], check=True)


def _run(name, t, s, p_dbar):
    inp = "".join(f"{a} {b} {c}\n" for a, b, c in zip(t, s, p_dbar))
    out = subprocess.run([_bin(name)], input=inp, capture_output=True, text=True,
                         check=True)
    return [float(x) for x in out.stdout.split()]


def gfortran_version():
    try:
        v = subprocess.run(["gfortran", "--version"], capture_output=True, text=True)
        return v.stdout.splitlines()[0]
    except (OSError, IndexError):
        return None


def mpas_eos_truth(t, s, p_dbar, h=1.0e-3):
    """Return ``{eos_id: {"rho", "alpha", "beta"}}`` from MPAS-O's compiled Fortran
    for the ``mpas-linear`` / ``mpas-jm`` / ``mpas-wright`` backends, or ``None`` if
    gfortran is unavailable. Each driver's published check value is asserted first.
    """
    if gfortran_version() is None:
        return None
    _build()
    t, s, p_dbar = np.asarray(t), np.asarray(s), np.asarray(p_dbar)
    cases = {}
    for name, eos_id in [("linear", "mpas-linear"), ("jm", "mpas-jm"),
                         ("wright", "mpas-wright")]:
        check = _run(name, [25.0], [35.0], [0.0])[0]
        assert abs(check - _CHECK[name]) < 1.0e-9, (
            f"MPAS {name} driver failed its check value: rho={check!r} "
            f"(expected {_CHECK[name]})")
        rho0 = np.array(_run(name, t, s, p_dbar))
        # centred finite difference of the MPAS-source density (same step the facade
        # uses), validating xeos's alpha/beta against an independent implementation.
        alpha = -(np.array(_run(name, t + h, s, p_dbar))
                  - np.array(_run(name, t - h, s, p_dbar))) / (2.0 * h) / rho0
        beta = (np.array(_run(name, t, s + h, p_dbar))
                - np.array(_run(name, t, s - h, p_dbar))) / (2.0 * h) / rho0
        cases[eos_id] = {"rho": rho0.tolist(),
                         "alpha": alpha.tolist(), "beta": beta.tolist()}
    return cases
