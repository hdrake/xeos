"""Density-derivative correctness (Q3 Gap 1 & 2).

Two independent checks:

* **analytic vs finite-difference** -- every backend that ships analytic
  ``drho_dt`` / ``drho_ds`` must agree with a centred finite difference of *its own*
  density kernel to O(h^2);
* **step-halving convergence** -- the FD-only backends (jmd95, unesco, mdjwf,
  mpas-jm) must show the ~4x error reduction of a centred second-order stencil when
  the step is halved (measured against a complex-step "exact" derivative, which is
  free of subtractive cancellation), with no float64 blow-up.

``gsw`` (TEOS-10) is used as a loose, independent analytic anchor for jmd95's
alpha/beta -- a sanity bound, not an equality (the two EOS differ physically).
"""

import numpy as np
import pytest

import xeos
from xeos.registry import get_backend

# Points inside the oceanographic funnel; the same numeric (t, s, p) works for
# both potential-temp/practical-salinity and conservative-temp/absolute-salinity
# backends (SA ~ SP numerically near 35).
_POINTS = [(10.0, 35.0, 0.0), (5.0, 34.0, 1000.0),
           (18.0, 36.0, 3000.0), (2.0, 33.0, 500.0)]

# Backends that expose analytic derivatives (facade returns them directly).
_ANALYTIC = [b for b in xeos.list_eos() if get_backend(b).drho_dt is not None
             and get_backend(b).drho_ds is not None]

# FD-only backends (no analytic derivatives -> facade uses centred differences).
_FD_ONLY = ["jmd95", "unesco", "mdjwf", "mpas-jm"]


def _needs_gsw(eos_id):
    return eos_id in ("teos10",)


@pytest.mark.parametrize("eos_id", _ANALYTIC)
@pytest.mark.parametrize("t,s,p", _POINTS)
def test_analytic_matches_finite_difference(eos_id, t, s, p):
    """Analytic drho_dt/drho_ds ~ centred FD of the same backend's density."""
    if _needs_gsw(eos_id):
        pytest.importorskip("gsw")
    eos = xeos.equation_of_state(eos_id)
    h = 1.0e-4
    fd_dt = (float(eos.rho(t + h, s, p)) - float(eos.rho(t - h, s, p))) / (2.0 * h)
    fd_ds = (float(eos.rho(t, s + h, p)) - float(eos.rho(t, s - h, p))) / (2.0 * h)
    np.testing.assert_allclose(float(eos.drho_dt(t, s, p)), fd_dt,
                               rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(float(eos.drho_ds(t, s, p)), fd_ds,
                               rtol=1e-4, atol=1e-6)


def _cstep(kernel, var, t, s, p, hcs=1e-20):
    """Complex-step derivative (machine-accurate, no cancellation)."""
    if var == "t":
        return kernel(complex(t, hcs), s, p).imag / hcs
    return kernel(t, complex(s, hcs), p).imag / hcs


def _fd_central(kernel, var, t, s, p, h):
    if var == "t":
        return (kernel(t + h, s, p) - kernel(t - h, s, p)) / (2.0 * h)
    return (kernel(t, s + h, p) - kernel(t, s - h, p)) / (2.0 * h)


@pytest.mark.parametrize("eos_id", _FD_ONLY)
@pytest.mark.parametrize("var", ["t", "s"])
def test_fd_step_halving_is_second_order(eos_id, var):
    """Halving h shrinks the centred-FD error ~4x (O(h^2)); no blow-up."""
    kernel = get_backend(eos_id).density  # native unit is dbar for all four
    t, s, p = 12.0, 35.0, 2000.0
    exact = _cstep(kernel, var, t, s, p)
    assert np.isfinite(exact)
    h = 0.02
    err_h = abs(_fd_central(kernel, var, t, s, p, h) - exact)
    err_half = abs(_fd_central(kernel, var, t, s, p, h / 2.0) - exact)
    assert np.isfinite(err_h) and np.isfinite(err_half)
    assert err_half < err_h                 # convergence
    assert 3.0 < err_h / err_half < 5.0      # ~4x  (second order)
    assert err_half < 1e-4                   # no cancellation blow-up


@pytest.mark.parametrize("t,s,p", [(10.0, 35.0, 0.0), (5.0, 34.0, 1000.0)])
def test_jmd95_alpha_beta_gsw_sanity(t, s, p):
    """gsw (TEOS-10) is an *independent* anchor for jmd95 alpha/beta: a loose
    sanity bound (the EOS differ physically), not an equality."""
    gsw = pytest.importorskip("gsw")
    eos = xeos.equation_of_state("jmd95")
    np.testing.assert_allclose(float(eos.alpha(t, s, p)), gsw.alpha(s, t, p), rtol=0.02)
    np.testing.assert_allclose(float(eos.beta(t, s, p)), gsw.beta(s, t, p), rtol=0.02)
