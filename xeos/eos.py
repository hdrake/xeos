"""The :class:`EquationOfState` facade — the object users actually call.

It wraps a registered :class:`~xeos.registry.EOSBackend`, converts user pressure
(default decibar) into the backend's native unit, dispatches kernels through
:func:`~xeos.xarray_utils.apply_eos` (so xarray labels / dask laziness survive),
and derives ``alpha``/``beta`` from the backend's analytic density derivatives,
falling back to centred finite differences when a backend supplies none.
"""

import importlib.util
import warnings

import numpy as np
import xarray as xr

from .conventions import PressureUnit, SalinityKind, to_native_pressure
from .registry import get_backend, list_eos, EOSBackend
from .xarray_utils import apply_eos

__all__ = ["EquationOfState"]


def _is_lazy_input(a):
    """True for inputs whose range can't be checked without a compute.

    (:class:`xarray.DataArray` -- may wrap dask -- or a bare ``dask.array.Array``).
    """
    if isinstance(a, xr.DataArray):
        return True
    return type(a).__module__.startswith("dask")

# CF-style attributes attached to xarray outputs.  Every ``units`` string here is
# parseable by UDUNITS-2, as CF requires.  ``alpha``/``drho_dt`` are spelled with
# K-1 rather than degC-1: both are reciprocals of a temperature *difference*, for
# which the kelvin and the degree Celsius are identical, and K-1 sidesteps the
# question of whether an offset unit may be exponentiated at all.  Only ``rho``
# and ``specific_volume`` get a ``standard_name``: the CF table does define
# ``sea_water_thermal_expansion_coefficient`` but has no counterpart for the
# haline contraction coefficient, and inventing one would be worse than omitting
# it, so none of the four coefficient fields is given a standard name.
_ATTRS = {
    "rho": {"standard_name": "sea_water_density", "units": "kg m-3",
            "long_name": "in-situ density"},
    "specific_volume": {"standard_name": "sea_water_specific_volume",
                        "units": "m3 kg-1", "long_name": "specific volume"},
    "alpha": {"units": "K-1", "long_name": "thermal expansion coefficient"},
    "drho_dt": {"units": "kg m-3 K-1",
                "long_name": "density derivative wrt temperature"},
}

# ``beta`` and ``drho_ds`` are absent from _ATTRS because their units depend on
# which salinity the backend expects -- practical salinity (PSS-78) is treated as
# dimensionless, absolute salinity is g kg-1 -- so they are resolved per call from
# ``self.salinity`` by :func:`_salinity_attrs`.
_SALINITY_ATTRS = {
    "beta": {
        "long_name": "haline contraction coefficient",
        "units": {SalinityKind.PRACTICAL: "1", SalinityKind.ABSOLUTE: "kg g-1"},
    },
    "drho_ds": {
        "long_name": "density derivative wrt salinity",
        "units": {SalinityKind.PRACTICAL: "kg m-3",
                  SalinityKind.ABSOLUTE: "kg2 m-3 g-1"},
    },
}


def _salinity_attrs(name, kind):
    """CF attrs for ``beta``/``drho_ds`` given a backend's :class:`SalinityKind`."""
    spec = _SALINITY_ATTRS[name]
    return {"units": spec["units"][SalinityKind(kind)],
            "long_name": spec["long_name"]}


_DT = 1.0e-3  # finite-difference step in temperature [degC]
_DS = 1.0e-3  # finite-difference step in salinity


class EquationOfState:
    """A single equation of state with a uniform ``(t, s, p)`` call signature.

    Parameters
    ----------
    eos : str or EOSBackend
        Canonical EOS id (see :func:`xeos.list_eos`) or a backend instance.
    pressure_input_unit : {"dbar", "Pa"}, default "dbar"
        Unit of the ``p`` argument passed to the methods below.
    accelerate : bool, default False
        Use the optional numba-accelerated kernels (density, and the analytic
        ``drho_dt``/``drho_ds`` where a backend provides them; FD-only backends get
        their alpha/beta from the accelerated density).  Requires ``numba``
        (``pip install xeos[complete]``).  A backend that has no fast path raises
        :class:`NotImplementedError`; numba being absent raises :class:`ImportError`.
        When ``False`` (the default) numba is never imported.
    """

    def __init__(self, eos, pressure_input_unit="dbar", accelerate=False):
        self.backend: EOSBackend = eos if isinstance(eos, EOSBackend) else get_backend(eos)
        self.pressure_input_unit = PressureUnit(pressure_input_unit)
        self.accelerate = bool(accelerate)
        if self.accelerate:
            if self.backend.density_fast is None:
                accel = sorted(i for i in list_eos()
                               if get_backend(i).density_fast is not None)
                raise NotImplementedError(
                    f"EOS {self.backend.id!r} has no numba-accelerated kernel; "
                    f"build it with accelerate=False. Accelerated backends: "
                    f"{', '.join(accel)}."
                )
            if importlib.util.find_spec("numba") is None:
                raise ImportError(
                    "accelerate=True requires the optional 'numba' dependency. "
                    "Install it with `pip install xeos[complete]`."
                )
            self._density = self.backend.density_fast
            # Prefer an accelerated analytic derivative; else numpy analytic; else
            # None -> the facade differences the (accelerated) density kernel.
            self._drho_dt_kernel = self.backend.drho_dt_fast or self.backend.drho_dt
            self._drho_ds_kernel = self.backend.drho_ds_fast or self.backend.drho_ds
        else:
            self._density = self.backend.density
            self._drho_dt_kernel = self.backend.drho_dt
            self._drho_ds_kernel = self.backend.drho_ds

    # -- introspection -----------------------------------------------------
    @property
    def id(self):
        return self.backend.id

    @property
    def temperature(self):
        return self.backend.temperature

    @property
    def salinity(self):
        return self.backend.salinity

    @property
    def reference(self):
        return self.backend.reference

    @property
    def description(self):
        return self.backend.description

    def __repr__(self):
        return (f"<EquationOfState {self.backend.id!r}: "
                f"{self.temperature.value} / {self.salinity.value}; "
                f"ref: {self.reference}>")

    # -- internals ---------------------------------------------------------
    def _warn_if_out_of_range(self, t, s, p):
        """Warn (never clamp) when plain numpy/scalar inputs leave the backend's
        documented ``valid_range``.  Lazy inputs (DataArray / dask) are skipped so
        no compute is forced.  One combined warning per call."""
        vr = self.backend.valid_range
        if not vr:
            return
        if any(_is_lazy_input(a) for a in (t, s, p)):
            return
        # Compare in user units: t [degC], s, p [dbar].
        if self.pressure_input_unit is PressureUnit.PASCAL:
            p_dbar = np.asarray(p, dtype=float) / 1.0e4
        else:
            p_dbar = np.asarray(p, dtype=float)
        checks = (("temperature", np.asarray(t, dtype=float), "t"),
                  ("salinity", np.asarray(s, dtype=float), "s"),
                  ("pressure", p_dbar, "p_dbar"))
        problems = []
        for label, arr, key in checks:
            if arr.size == 0 or key not in vr:
                continue
            lo, hi = vr[key]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
                amin = float(np.nanmin(arr))
                amax = float(np.nanmax(arr))
            if amin < lo or amax > hi:
                problems.append(
                    f"{label} [{amin:.4g}, {amax:.4g}] outside [{lo:g}, {hi:g}]")
        if problems:
            warnings.warn(
                f"{self.backend.id}: inputs outside the documented valid range "
                f"({'; '.join(problems)}); EOS accuracy is not guaranteed there.",
                stacklevel=2,
            )

    def _native_p(self, p):
        # Convert user pressure -> dbar -> backend native unit.
        if self.pressure_input_unit is PressureUnit.PASCAL:
            p_dbar = p / 1.0e4
        else:
            p_dbar = p
        return to_native_pressure(p_dbar, self.backend.pressure_unit)

    def _drho_dt(self, t, s, pn):
        if self._drho_dt_kernel is not None:
            return self._drho_dt_kernel(t, s, pn)
        return (self._density(t + _DT, s, pn)
                - self._density(t - _DT, s, pn)) / (2.0 * _DT)

    def _drho_ds(self, t, s, pn):
        if self._drho_ds_kernel is not None:
            return self._drho_ds_kernel(t, s, pn)
        # Centred difference, but switch to a forward difference where salinity is
        # below the step (several EOS contain sqrt(s), so s - _DS < 0 -> NaN).
        # This keeps beta finite for near-fresh water (river plumes, ice melt).
        s_lo = np.where(s >= _DS, s - _DS, s)
        denom = np.where(s >= _DS, 2.0 * _DS, _DS)
        return (self._density(t, s + _DS, pn)
                - self._density(t, s_lo, pn)) / denom

    # -- quantities --------------------------------------------------------
    def rho(self, t, s, p):
        """In-situ density [kg m-3]."""
        self._warn_if_out_of_range(t, s, p)
        pn = self._native_p(p)
        return apply_eos(self._density, t, s, pn, attrs=_ATTRS["rho"])

    def specific_volume(self, t, s, p):
        """Specific volume [m3 kg-1]."""
        self._warn_if_out_of_range(t, s, p)
        pn = self._native_p(p)
        if self.backend.specific_volume is not None:
            func = self.backend.specific_volume
        else:
            def func(t_, s_, p_):
                return 1.0 / self._density(t_, s_, p_)
        return apply_eos(func, t, s, pn, attrs=_ATTRS["specific_volume"])

    def drho_dt(self, t, s, p):
        """Partial derivative of density wrt temperature [kg m-3 K-1]."""
        self._warn_if_out_of_range(t, s, p)
        pn = self._native_p(p)
        return apply_eos(lambda t_, s_, p_: self._drho_dt(t_, s_, p_),
                         t, s, pn, attrs=_ATTRS["drho_dt"])

    def drho_ds(self, t, s, p):
        """Partial derivative of density wrt salinity.

        Units follow the backend's salinity kind: [kg m-3] for practical salinity
        (PSS-78, dimensionless) and [kg2 m-3 g-1] for absolute salinity [g kg-1].
        The returned attributes record which.
        """
        self._warn_if_out_of_range(t, s, p)
        pn = self._native_p(p)
        return apply_eos(lambda t_, s_, p_: self._drho_ds(t_, s_, p_),
                         t, s, pn,
                         attrs=_salinity_attrs("drho_ds", self.salinity))

    def alpha(self, t, s, p):
        """Thermal expansion coefficient ``-(1/rho) drho/dT`` [K-1]."""
        self._warn_if_out_of_range(t, s, p)
        pn = self._native_p(p)

        def func(t_, s_, p_):
            return -self._drho_dt(t_, s_, p_) / self._density(t_, s_, p_)

        return apply_eos(func, t, s, pn, attrs=_ATTRS["alpha"])

    def beta(self, t, s, p):
        """Haline contraction coefficient ``(1/rho) drho/dS``.

        Units follow the backend's salinity kind: dimensionless [1] for practical
        salinity (PSS-78) and [kg g-1] for absolute salinity [g kg-1].  The
        returned attributes record which.

        Caveat for near-fresh water: backends whose surface polynomial carries
        an ``s**1.5`` term (``jmd95``, ``unesco``, ``mdjwf``) have a *true*
        ``drho/dS`` that diverges as ``S -> 0``. Where those backends fall back
        to finite differences, ``_drho_ds`` switches to a one-sided step below
        the difference increment so ``beta`` stays finite for river-plume / ice-
        melt inputs -- but the value returned there is a finite proxy for an
        unbounded quantity and should not be trusted quantitatively at very low
        salinity. (Density itself remains well behaved.)
        """
        self._warn_if_out_of_range(t, s, p)
        pn = self._native_p(p)

        def func(t_, s_, p_):
            return self._drho_ds(t_, s_, p_) / self._density(t_, s_, p_)

        return apply_eos(func, t, s, pn,
                         attrs=_salinity_attrs("beta", self.salinity))
