"""Build & run a standalone driver from MOM6's authoritative ROQUET_SPV Fortran.

The Roquet specific-volume EOS (MOM6 ``EQN_OF_STATE='ROQUET_SPV'``) has no
trustworthy Python reference: the widely-used ``polyTEOS10.py`` has a typo in its
55-term specific-volume routine (``deltaS=32`` instead of ``24``) that makes its
output disagree with its own published check values. So we validate xeos's vendored
``roquet-spv`` against the actual MOM6 Fortran instead.

This module downloads ``MOM_EOS_Roquet_SpV.F90`` (not committed — MOM6 is LGPL;
only the resulting numbers go into truth.json), extracts its module parameters and
the ``spec_vol_elem`` + ``calculate_specvol_derivs`` elemental routines into a
self-contained driver, compiles it with gfortran, and evaluates it on a grid.

A documented check value (specvol = 9.732820466e-04 at SA=30, CT=10, p=1000 dbar)
is asserted after the build, so silent breakage from upstream reformatting fails
loudly rather than producing wrong "truth".
"""

import os
import re
import subprocess
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MOM6_URL = ("https://raw.githubusercontent.com/mom-ocean/MOM6/main/"
            "src/equation_of_state/MOM_EOS_Roquet_SpV.F90")
_F90 = os.path.join(HERE, "MOM_EOS_Roquet_SpV.F90")   # gitignored
_DRIVER = os.path.join(HERE, "_roquet_spv_driver.f90")  # gitignored
_BIN = os.path.join(HERE, "_roquet_spv_driver")          # gitignored


def _function_body(src, name):
    """Return the executable lines of an elemental routine `name` (function or
    subroutine), from just after its `zp = pressure` line up to (not incl.) `end`."""
    m = re.search(rf"(?:function|subroutine) {re.escape(name)}\b", src)
    body = src[m.start():]
    lines = body.splitlines()
    out, started = [], False
    for ln in lines:
        if "zp = pressure" in ln:
            started = True
            continue
        if started:
            if ln.strip().startswith("end function") or ln.strip().startswith("end subroutine"):
                break
            # drop the commented gsw-conversion hints
            if ln.strip().startswith("!"):
                continue
            out.append(ln)
    return "\n".join(out)


def _build():
    if not os.path.exists(_F90):
        urllib.request.urlretrieve(MOM6_URL, _F90)
    src = open(_F90).read()

    params = [ln.split("!", 1)[0].rstrip() for ln in src.splitlines()
              if ln.strip().startswith("real, parameter ::")]
    params_block = "\n".join("  " + p.strip() for p in params)

    spec = _function_body(src, "spec_vol_elem_Roquet_SpV")
    spec = spec.replace("spec_vol_elem_Roquet_SpV", "spv")
    derivs = _function_body(src, "calculate_specvol_derivs_elem_Roquet_SpV")

    driver = f"""program roquet_spv_truth
  implicit none
{params_block}
  real :: SA, CT, pdbar, sv, dT, dS, alpha, beta
  integer :: ios
  do
    read(*,*,iostat=ios) SA, CT, pdbar
    if (ios /= 0) exit
    call compute(CT, SA, pdbar*1.0e4, sv, dT, dS)
    alpha = dT / sv
    beta  = -dS / sv
    write(*,'(4ES25.16)') sv, 1.0/sv, alpha, beta
  end do
contains
  subroutine compute(T, S, pressure, spv, dSV_dT, dSV_dS)
    real, intent(in) :: T, S, pressure
    real, intent(out) :: spv, dSV_dT, dSV_dS
    real :: zp, zt, zs
    real :: SV_00p, SV_TS, SV_TS0, SV_TS1, SV_TS2, SV_TS3, SV_0S0
    real :: dSVdzt0, dSVdzt1, dSVdzt2, dSVdzt3
    real :: dSVdzs0, dSVdzs1, dSVdzs2, dSVdzs3
    zt = T
    zs = SQRT( ABS( S + rdeltaS ) * r1_S0 )
    zp = pressure
{spec}
{derivs}
  end subroutine
end program
"""
    open(_DRIVER, "w").write(driver)
    subprocess.run(["gfortran", "-fdefault-real-8", "-fdefault-double-8", "-O2",
                    "-o", _BIN, _DRIVER], check=True)


def _run(sa, ct, p_dbar):
    inp = "".join(f"{s} {t} {p}\n" for s, t, p in zip(sa, ct, p_dbar))
    out = subprocess.run([_BIN], input=inp, capture_output=True, text=True, check=True)
    rows = [[float(x) for x in line.split()] for line in out.stdout.strip().splitlines()]
    return rows  # each row: [specvol, rho, alpha, beta]


def gfortran_version():
    try:
        v = subprocess.run(["gfortran", "--version"], capture_output=True, text=True)
        return v.stdout.splitlines()[0]
    except (OSError, IndexError):
        return None


def roquet_spv_truth(sa, ct, p_dbar):
    """Return dict of rho/alpha/beta lists from the MOM6 Fortran, or None if
    gfortran is unavailable. Asserts the published specvol check value first."""
    if gfortran_version() is None:
        return None
    _build()
    # self-check against the published check value before trusting any output
    check = _run([30.0], [10.0], [1000.0])[0]
    assert abs(check[0] - 9.732820466e-04) < 1e-12, (
        f"Fortran driver failed its check value: specvol={check[0]!r}")
    rows = _run(list(sa), list(ct), list(p_dbar))
    return {
        "rho": [r[1] for r in rows],
        "alpha": [r[2] for r in rows],
        "beta": [r[3] for r in rows],
    }
