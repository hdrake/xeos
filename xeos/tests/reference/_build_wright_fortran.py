"""Build & run standalone drivers from MOM6's authoritative Wright-1997 Fortran.

xeos's ``wright97-reduced`` and ``wright97-full`` kernels are transcriptions of
MOM6's reduced- and full-range Wright equations of state. Rather than trust the
third-party ``momlevel`` Python port (which only implements the reduced-range
coefficients), this module validates them against the *actual* MOM6 Fortran:
``MOM_EOS_Wright_red.F90`` and ``MOM_EOS_Wright_full.F90``.

For each variant it downloads the source (not committed -- MOM6 is LGPL; only the
resulting numbers go into ``truth.json``), extracts the module ``real, parameter``
coefficients and the executable bodies of the ``density_elem_Wright_*`` and
``calculate_density_derivs_elem_Wright_*`` elemental routines into a self-contained
driver, compiles it with gfortran, and evaluates it on a grid.

Before any output is trusted, the driver self-checks its density at S=35, T=10,
p=2000 dbar against a value computed independently from the *published* Wright
(1997) coefficients (``_CHECK`` below), so a mangled extraction or an upstream
coefficient change fails loudly rather than producing wrong "truth".

The native pressure unit is Pa; the grid (given in dbar) is multiplied by 1e4.
"""

import os
import re
import subprocess
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
_BASE = (
    "https://raw.githubusercontent.com/mom-ocean/MOM6/main/" "src/equation_of_state/"
)

# Density at S=35, T=10, p=2000 dbar, computed independently from the published
# Wright-1997 coefficients (see this folder's README / _build script header). Any
# faithful build of the MOM6 source must reproduce these to ~1e-9 relative.
_CHECK = {"red": 1035.7639223341512, "full": 1035.7651718698569}


def _variant_files(variant):
    f90 = os.path.join(HERE, f"MOM_EOS_Wright_{variant}.F90")  # gitignored
    driver = os.path.join(HERE, f"_wright_{variant}_driver.f90")  # gitignored
    binary = os.path.join(HERE, f"_wright_{variant}_driver")  # gitignored
    return f90, driver, binary


def _exec_lines(src, name):
    """Return the executable (assignment / continuation) lines of an elemental
    routine `name`, dropping its signature, declarations (any line with ``::``)
    and comments. Continuation lines (trailing ``&``) are preserved verbatim."""
    m = re.search(rf"(?:function|subroutine) {re.escape(name)}\b", src)
    lines = src[m.start() :].splitlines()[1:]  # skip the signature line
    out = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("end function") or s.startswith("end subroutine"):
            break
        if not s or s.startswith("!") or "::" in ln:
            continue
        out.append(ln)
    return "\n".join(out)


def _build(variant):
    f90, driver_path, binary = _variant_files(variant)
    if not os.path.exists(f90):
        urllib.request.urlretrieve(f"{_BASE}MOM_EOS_Wright_{variant}.F90", f90)
    src = open(f90).read()

    # Module-level coefficients only: the routines below `contains` declare their
    # own local `real, parameter` rational constants (C1_3, ...) we must not grab.
    module_head = re.split(r"(?m)^contains\b", src, maxsplit=1)[0]
    params = [
        ln.split("!", 1)[0].rstrip()
        for ln in module_head.splitlines()
        if ln.strip().startswith("real, parameter ::")
    ]
    params_block = "\n".join("  " + p.strip() for p in params)

    dens = _exec_lines(src, f"density_elem_Wright_{variant}")
    dens = dens.replace(f"density_elem_Wright_{variant}", "rho")
    derivs = _exec_lines(src, f"calculate_density_derivs_elem_Wright_{variant}")

    driver = f"""program wright_truth
  implicit none
{params_block}
  real :: T, S, pdbar, rho, drho_dT, drho_dS, alpha, beta
  integer :: ios
  do
    read(*,*,iostat=ios) S, T, pdbar
    if (ios /= 0) exit
    call compute(T, S, pdbar*1.0e4, rho, drho_dT, drho_dS)
    alpha = -drho_dT / rho
    beta  =  drho_dS / rho
    write(*,'(3ES25.16)') rho, alpha, beta
  end do
contains
  subroutine compute(T, S, pressure, rho, drho_dT, drho_dS)
    real, intent(in) :: T, S, pressure
    real, intent(out) :: rho, drho_dT, drho_dS
    real :: al0, p0, lambda, I_denom2
{dens}
{derivs}
  end subroutine
end program
"""
    open(driver_path, "w").write(driver)
    subprocess.run(
        [
            "gfortran",
            "-fdefault-real-8",
            "-fdefault-double-8",
            "-O2",
            "-o",
            binary,
            driver_path,
        ],
        check=True,
    )
    return binary


def _run(binary, s, t, p_dbar):
    inp = "".join(f"{si} {ti} {pi}\n" for si, ti, pi in zip(s, t, p_dbar))
    out = subprocess.run(
        [binary], input=inp, capture_output=True, text=True, check=True
    )
    return [
        [float(x) for x in line.split()] for line in out.stdout.strip().splitlines()
    ]


def gfortran_version():
    try:
        v = subprocess.run(["gfortran", "--version"], capture_output=True, text=True)
        return v.stdout.splitlines()[0]
    except (OSError, IndexError):
        return None


def wright_truth(variant, s, t, p_dbar):
    """Return dict of rho/alpha/beta lists from the MOM6 Wright Fortran for
    `variant` in {"red", "full"}, or None if gfortran is unavailable. Asserts the
    published-coefficient check value first."""
    if gfortran_version() is None:
        return None
    binary = _build(variant)
    check = _run(binary, [35.0], [10.0], [2000.0])[0][0]
    assert abs(check - _CHECK[variant]) < 1e-6 * _CHECK[variant], (
        f"Wright {variant} driver failed its check value: rho={check!r} "
        f"(expected {_CHECK[variant]!r})"
    )
    rows = _run(binary, list(s), list(t), list(p_dbar))
    return {
        "rho": [r[0] for r in rows],
        "alpha": [r[1] for r in rows],
        "beta": [r[2] for r in rows],
    }
