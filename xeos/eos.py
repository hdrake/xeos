"""The :class:`EquationOfState` facade — the object users actually call.

It wraps a registered :class:`~xeos.registry.EOSBackend`, converts user pressure
(default decibar) into the backend's native unit, dispatches kernels through
:func:`~xeos.xarray_utils.apply_eos` (so xarray labels / dask laziness survive),
and derives ``alpha``/``beta`` from the backend's analytic density derivatives,
falling back to centred finite differences when a backend supplies none.
"""

import numpy as np

from .conventions import PressureUnit, to_native_pressure
from .registry import get_backend, EOSBackend
from .xarray_utils import apply_eos

__all__ = ["EquationOfState"]

# CF-style attributes attached to xarray outputs.
_ATTRS = {
    "rho": {"standard_name": "sea_water_density", "units": "kg m-3",
            "long_name": "in-situ density"},
    "specific_volume": {"standard_name": "sea_water_specific_volume",
                        "units": "m3 kg-1", "long_name": "specific volume"},
    "alpha": {"units": "degC-1", "long_name": "thermal expansion coefficient"},
    "beta": {"units": "(salinity unit)-1",
             "long_name": "haline contraction coefficient"},
    "drho_dt": {"units": "kg m-3 degC-1",
                "long_name": "density derivative wrt temperature"},
    "drho_ds": {"units": "kg m-3 (salinity unit)-1",
                "long_name": "density derivative wrt salinity"},
}

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
    """

    def __init__(self, eos, pressure_input_unit="dbar"):
        self.backend: EOSBackend = eos if isinstance(eos, EOSBackend) else get_backend(eos)
        self.pressure_input_unit = PressureUnit(pressure_input_unit)

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
                f"{self.temperature.value} / {self.salinity.value}>")

    # -- internals ---------------------------------------------------------
    def _native_p(self, p):
        # Convert user pressure -> dbar -> backend native unit.
        if self.pressure_input_unit is PressureUnit.PASCAL:
            p_dbar = p / 1.0e4
        else:
            p_dbar = p
        return to_native_pressure(p_dbar, self.backend.pressure_unit)

    def _drho_dt(self, t, s, pn):
        if self.backend.drho_dt is not None:
            return self.backend.drho_dt(t, s, pn)
        return (self.backend.density(t + _DT, s, pn)
                - self.backend.density(t - _DT, s, pn)) / (2.0 * _DT)

    def _drho_ds(self, t, s, pn):
        if self.backend.drho_ds is not None:
            return self.backend.drho_ds(t, s, pn)
        # Centred difference, but switch to a forward difference where salinity is
        # below the step (several EOS contain sqrt(s), so s - _DS < 0 -> NaN).
        # This keeps beta finite for near-fresh water (river plumes, ice melt).
        s_lo = np.where(s >= _DS, s - _DS, s)
        denom = np.where(s >= _DS, 2.0 * _DS, _DS)
        return (self.backend.density(t, s + _DS, pn)
                - self.backend.density(t, s_lo, pn)) / denom

    # -- quantities --------------------------------------------------------
    def rho(self, t, s, p):
        """In-situ density [kg m-3]."""
        pn = self._native_p(p)
        return apply_eos(self.backend.density, t, s, pn, attrs=_ATTRS["rho"])

    def specific_volume(self, t, s, p):
        """Specific volume [m3 kg-1]."""
        pn = self._native_p(p)
        if self.backend.specific_volume is not None:
            func = self.backend.specific_volume
        else:
            def func(t_, s_, p_):
                return 1.0 / self.backend.density(t_, s_, p_)
        return apply_eos(func, t, s, pn, attrs=_ATTRS["specific_volume"])

    def drho_dt(self, t, s, p):
        """Partial derivative of density wrt temperature [kg m-3 degC-1]."""
        pn = self._native_p(p)
        return apply_eos(lambda t_, s_, p_: self._drho_dt(t_, s_, p_),
                         t, s, pn, attrs=_ATTRS["drho_dt"])

    def drho_ds(self, t, s, p):
        """Partial derivative of density wrt salinity [kg m-3 (salt unit)-1]."""
        pn = self._native_p(p)
        return apply_eos(lambda t_, s_, p_: self._drho_ds(t_, s_, p_),
                         t, s, pn, attrs=_ATTRS["drho_ds"])

    def alpha(self, t, s, p):
        """Thermal expansion coefficient ``-(1/rho) drho/dT`` [degC-1]."""
        pn = self._native_p(p)

        def func(t_, s_, p_):
            return -self._drho_dt(t_, s_, p_) / self.backend.density(t_, s_, p_)

        return apply_eos(func, t, s, pn, attrs=_ATTRS["alpha"])

    def beta(self, t, s, p):
        """Haline contraction coefficient ``(1/rho) drho/dS`` [(salt unit)-1]."""
        pn = self._native_p(p)

        def func(t_, s_, p_):
            return self._drho_ds(t_, s_, p_) / self.backend.density(t_, s_, p_)

        return apply_eos(func, t, s, pn, attrs=_ATTRS["beta"])
