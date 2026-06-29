"""Functional shims: ``xeos.rho(t, s, p, eos=...)`` and friends.

Thin wrappers over :class:`~xeos.eos.EquationOfState` for one-off calls.  For
repeated use, build an ``EquationOfState`` once (via :func:`xeos.equation_of_state`
or :func:`xeos.from_model`) and call its methods.
"""

from .models import equation_of_state

__all__ = ["rho", "alpha", "beta", "specific_volume"]


def _eos(eos, pressure_input_unit, params):
    return equation_of_state(eos, pressure_input_unit=pressure_input_unit, **params)


def rho(t, s, p, eos, pressure_input_unit="dbar", **params):
    """In-situ density [kg m-3] from the named ``eos``."""
    return _eos(eos, pressure_input_unit, params).rho(t, s, p)


def alpha(t, s, p, eos, pressure_input_unit="dbar", **params):
    """Thermal expansion coefficient [degC-1] from the named ``eos``."""
    return _eos(eos, pressure_input_unit, params).alpha(t, s, p)


def beta(t, s, p, eos, pressure_input_unit="dbar", **params):
    """Haline contraction coefficient [(salt unit)-1] from the named ``eos``."""
    return _eos(eos, pressure_input_unit, params).beta(t, s, p)


def specific_volume(t, s, p, eos, pressure_input_unit="dbar", **params):
    """Specific volume [m3 kg-1] from the named ``eos``."""
    return _eos(eos, pressure_input_unit, params).specific_volume(t, s, p)
