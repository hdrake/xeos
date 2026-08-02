"""Functional shims: ``xeos.rho(t, s, p, eos=...)`` and friends.

Thin wrappers over :class:`~xeos.eos.EquationOfState` for one-off calls.  For
repeated use, build an ``EquationOfState`` once (via :func:`xeos.equation_of_state`
or :func:`xeos.from_model`) and call its methods.
"""

from .models import equation_of_state

__all__ = ["rho", "alpha", "beta", "specific_volume"]


def _eos(eos, pressure_input_unit, accelerate, params):
    return equation_of_state(
        eos, pressure_input_unit=pressure_input_unit, accelerate=accelerate, **params
    )


def rho(t, s, p, eos, pressure_input_unit="dbar", accelerate=False, **params):
    """In-situ density [kg m-3] from the named ``eos``."""
    return _eos(eos, pressure_input_unit, accelerate, params).rho(t, s, p)


def alpha(t, s, p, eos, pressure_input_unit="dbar", accelerate=False, **params):
    """Thermal expansion coefficient [K-1] from the named ``eos``."""
    return _eos(eos, pressure_input_unit, accelerate, params).alpha(t, s, p)


def beta(t, s, p, eos, pressure_input_unit="dbar", accelerate=False, **params):
    """Haline contraction coefficient from the named ``eos``.

    The units are *spelled* to suit the EOS's salinity kind -- [1000] for
    practical salinity (PSS-78 is dimensionless but scaled by 1e-3), [kg g-1]
    for absolute salinity -- but the two spellings are the same unit.
    """
    return _eos(eos, pressure_input_unit, accelerate, params).beta(t, s, p)


def specific_volume(t, s, p, eos, pressure_input_unit="dbar", accelerate=False, **params):
    """Specific volume [m3 kg-1] from the named ``eos``."""
    return _eos(eos, pressure_input_unit, accelerate, params).specific_volume(t, s, p)
