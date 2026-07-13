"""Build & run a standalone driver from MOM6's authoritative Roquet-rho Fortran.

xeos's ``teos10-poly55`` backend is the Roquet et al. (2015) 55-term *density*
polynomial (the ``polyTEOS10_bsq`` form). Its only other reference here is
``polyTEOS10.py`` (Roquet's own Python), a third-party port that could silently
drift. This generator gives it a *second, independent* model source: MOM6's
``MOM_EOS_Roquet_rho.F90`` (the same polynomial NEMO/MOM6 use), compiled with
gfortran, so ``teos10-poly55`` is no longer validated against a single Python file.

Same mechanics as ``_build_roquet_spv_fortran.py``: download the source (not
committed -- MOM6 is Apache-2.0; only the resulting numbers go into ``truth.json``),
extract the module ``real, parameter`` coefficients and the executable bodies of the
``density_elem_Roquet_rho`` function and ``calculate_density_derivs_elem_Roquet_rho``
subroutine into a self-contained driver, compile, and evaluate on the grid.

alpha/beta come from the model's *analytic* derivative routine
(``calculate_density_derivs_elem_Roquet_rho``): alpha = -drho_dT/rho,
beta = drho_dS/rho, matching how xeos and the polyTEOS10 path derive them.

Native pressure unit is Pa (the coefficients carry Pa2kb factors); the grid, given
in dbar, is multiplied by 1e4. Before any output is trusted, the driver self-checks
its density at SA=30, CT=10, p=1000 dbar against the published Roquet check value
(rho = 1027.45140 kg m-3, from polyTEOS10.py's ``polyTEOS10_bsq`` header), so a
mangled extraction or an upstream change fails loudly rather than producing wrong
"truth".
"""

import os
import re
import subprocess
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
# Pinned to a specific MOM6 commit (see generate_truth.py provenance) so the truth
# is reproducible; bump the SHA there and here together when refreshing.
MOM6_SHA = "d4d0437f0366e097ff90fe896ae6307cb3b67c56"
MOM6_URL = (f"https://raw.githubusercontent.com/mom-ocean/MOM6/{MOM6_SHA}/"
            "src/equation_of_state/MOM_EOS_Roquet_rho.F90")
_F90 = os.path.join(HERE, "MOM_EOS_Roquet_rho.F90")   # gitignored
_DRIVER = os.path.join(HERE, "_roquet_rho_driver.f90")  # gitignored
_BIN = os.path.join(HERE, "_roquet_rho_driver")          # gitignored

# Published check: in-situ density at SA=30, CT=10, p=1000 dbar for the 55-term
# density polynomial (polyTEOS10.py polyTEOS10_bsq header CHECK VALUES).
_CHECK = 1027.45140


def _function_body(src, name):
    """Return the executable lines of an elemental routine `name` (function or
    subroutine), from just after its `zp = pressure` line up to (not incl.) `end`."""
    m = re.search(rf"(?:function|subroutine) {re.escape(name)}\b", src)
    body = src[m.start():]
    out, started = [], False
    for ln in body.splitlines():
        if "zp = pressure" in ln:
            started = True
            continue
        if started:
            if ln.strip().startswith("end function") or ln.strip().startswith("end subroutine"):
                break
            # drop the commented gsw-conversion hints and inline notes
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

    dens = _function_body(src, "density_elem_Roquet_rho")
    dens = dens.replace("density_elem_Roquet_rho", "rho")
    derivs = _function_body(src, "calculate_density_derivs_elem_Roquet_rho")

    driver = f"""program roquet_rho_truth
  implicit none
{params_block}
  real :: SA, CT, pdbar, rho, drho_dT, drho_dS, alpha, beta
  integer :: ios
  do
    read(*,*,iostat=ios) SA, CT, pdbar
    if (ios /= 0) exit
    call compute(CT, SA, pdbar*1.0e4, rho, drho_dT, drho_dS)
    alpha = -drho_dT / rho
    beta  =  drho_dS / rho
    write(*,'(3ES25.16)') rho, alpha, beta
  end do
contains
  subroutine compute(T, S, pressure, rho, drho_dT, drho_dS)
    real, intent(in) :: T, S, pressure
    real, intent(out) :: rho, drho_dT, drho_dS
    real :: zp, zt, zs
    real :: rho00p, rhoTS, rhoTS0, rhoTS1, rhoTS2, rhoTS3, rho0S0
    real :: dRdzt0, dRdzt1, dRdzt2, dRdzt3
    real :: dRdzs0, dRdzs1, dRdzs2, dRdzs3
    zt = T
    zs = SQRT( ABS( S + rdeltaS ) * r1_S0 )
    zp = pressure
{dens}
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
    return rows  # each row: [rho, alpha, beta]


def gfortran_version():
    try:
        v = subprocess.run(["gfortran", "--version"], capture_output=True, text=True)
        return v.stdout.splitlines()[0]
    except (OSError, IndexError):
        return None


def roquet_rho_truth(sa, ct, p_dbar):
    """Return dict of rho/alpha/beta lists from the MOM6 Roquet-rho Fortran, or None
    if gfortran is unavailable. Asserts the published density check value first."""
    if gfortran_version() is None:
        return None
    _build()
    # self-check against the published check value before trusting any output
    check = _run([30.0], [10.0], [1000.0])[0][0]
    assert abs(check - _CHECK) < 1e-4, (
        f"MOM6 Roquet_rho driver failed its check value: rho={check!r} "
        f"(expected {_CHECK!r})")
    rows = _run(list(sa), list(ct), list(p_dbar))
    return {
        "rho": [r[0] for r in rows],
        "alpha": [r[1] for r in rows],
        "beta": [r[2] for r in rows],
    }
