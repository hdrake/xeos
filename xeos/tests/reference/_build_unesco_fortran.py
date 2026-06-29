"""Build & run a standalone driver from MOM6's authoritative UNESCO Fortran.

MOM6's ``MOM_EOS_UNESCO.F90`` is, despite its name, the **Jackett & McDougall
(1995)** potential-temperature fit ("the equation of state using the Jackett and
McDougall fits to the UNESCO EOS"), *not* the original Fofonoff & Millard (1983)
EOS-80. Its coefficients and ``rho0*ks/(ks-p)`` form are mathematically identical
to xeos's ``jmd95`` backend -- so this generator validates ``jmd95`` against MOM6
source, and confirms that MOM6 ``EQN_OF_STATE='UNESCO'`` / ``'JACKETT_MCD'`` should
resolve to ``jmd95``, not to xeos's ``unesco`` (which is the genuinely-different
EOS-80, the MITgcm ``UNESCO`` lineage).

Same mechanics as ``_build_wright_fortran.py``: download the LGPL source (not
committed -- only the numbers go into ``truth.json``), extract the module
``real, parameter`` coefficients and the executable body of the
``density_elem_UNESCO`` elemental function into a self-contained driver, compile
with gfortran, evaluate on a grid. Native pressure unit is Pa (the routine
converts to bar internally), so the dbar grid is multiplied by 1e4.

alpha/beta are produced by centred finite difference of the MOM6 density (the same
h=1e-3 the facade uses for ``jmd95``), validating xeos's own FD path against an
independent compiled implementation rather than against analytic derivatives.

A published-coefficient check value (rho = 1031.6521274959705 at S=35, T=25,
p=2000 dbar, the JMD95 fit) gates the output before it is trusted.
"""

import os
import re
import subprocess
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MOM6_URL = (
    "https://raw.githubusercontent.com/mom-ocean/MOM6/main/"
    "src/equation_of_state/MOM_EOS_UNESCO.F90"
)
_F90 = os.path.join(HERE, "MOM_EOS_UNESCO.F90")  # gitignored
_DRIVER = os.path.join(HERE, "_unesco_mom6_driver.f90")  # gitignored
_BIN = os.path.join(HERE, "_unesco_mom6_driver")  # gitignored

_CHECK = 1031.6521274959705  # JMD95 density at S=35, T=25, p=2000 dbar
_H = 1.0e-3  # FD step; matches the facade's _DT/_DS


def _exec_lines(src, name):
    """Executable (assignment / continuation) lines of elemental routine `name`,
    dropping signature, declarations (any line with ``::``) and comments."""
    m = re.search(rf"(?:function|subroutine) {re.escape(name)}\b", src)
    lines = src[m.start() :].splitlines()[1:]
    out = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("end function") or s.startswith("end subroutine"):
            break
        if not s or s.startswith("!") or "::" in ln:
            continue
        out.append(ln)
    return "\n".join(out)


def _build():
    if not os.path.exists(_F90):
        urllib.request.urlretrieve(MOM6_URL, _F90)
    src = open(_F90).read()

    module_head = re.split(r"(?m)^contains\b", src, maxsplit=1)[0]
    params = [
        ln.split("!", 1)[0].rstrip()
        for ln in module_head.splitlines()
        if ln.strip().startswith("real, parameter ::")
    ]
    params_block = "\n".join("  " + p.strip() for p in params)

    dens = _exec_lines(src, "density_elem_UNESCO").replace("density_elem_UNESCO", "rho")

    driver = f"""program unesco_truth
  implicit none
{params_block}
  real :: T, S, pdbar, rho
  integer :: ios
  do
    read(*,*,iostat=ios) S, T, pdbar
    if (ios /= 0) exit
    call compute(T, S, pdbar*1.0e4, rho)
    write(*,'(ES25.16)') rho
  end do
contains
  subroutine compute(T, S, pressure, rho)
    real, intent(in) :: T, S, pressure
    real, intent(out) :: rho
    real :: t1, s1, p1, s12, rho0, sig0, ks
{dens}
  end subroutine
end program
"""
    open(_DRIVER, "w").write(driver)
    subprocess.run(
        [
            "gfortran",
            "-fdefault-real-8",
            "-fdefault-double-8",
            "-O2",
            "-o",
            _BIN,
            _DRIVER,
        ],
        check=True,
    )


def _run(s, t, p_dbar):
    inp = "".join(f"{si} {ti} {pi}\n" for si, ti, pi in zip(s, t, p_dbar))
    out = subprocess.run([_BIN], input=inp, capture_output=True, text=True, check=True)
    return [float(line) for line in out.stdout.strip().splitlines()]


def gfortran_version():
    try:
        v = subprocess.run(["gfortran", "--version"], capture_output=True, text=True)
        return v.stdout.splitlines()[0]
    except (OSError, IndexError):
        return None


def unesco_mom6_truth(s, t, p_dbar):
    """Return dict of rho/alpha/beta lists from the MOM6 UNESCO (= JMD95) Fortran,
    or None if gfortran is unavailable. alpha/beta by centred FD of the compiled
    density. Asserts the published JMD95 check value first."""
    if gfortran_version() is None:
        return None
    _build()
    check = _run([35.0], [25.0], [2000.0])[0]
    assert abs(check - _CHECK) < 1e-6 * _CHECK, (
        f"MOM6 UNESCO driver failed its check value: rho={check!r} "
        f"(expected {_CHECK!r})"
    )
    s, t, p_dbar = list(s), list(t), list(p_dbar)
    n = len(s)
    rho = _run(s, t, p_dbar)
    # centred differences: stack +/- perturbations into single batched runs
    rt_p = _run(s, [ti + _H for ti in t], p_dbar)
    rt_m = _run(s, [ti - _H for ti in t], p_dbar)
    rs_p = _run([si + _H for si in s], t, p_dbar)
    rs_m = _run([si - _H for si in s], t, p_dbar)
    alpha = [-(rt_p[i] - rt_m[i]) / (2.0 * _H) / rho[i] for i in range(n)]
    beta = [(rs_p[i] - rs_m[i]) / (2.0 * _H) / rho[i] for i in range(n)]
    return {"rho": rho, "alpha": alpha, "beta": beta}
