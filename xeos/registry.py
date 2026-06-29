"""Registry mapping canonical EOS ids to backend implementations + metadata.

Each backend is a small bundle of callables (at minimum ``density``) plus the
input conventions it expects.  Backends register themselves at import time via
:func:`register`; :mod:`xeos.models` then maps each ocean model's own selector
strings (e.g. MOM6 ``EQN_OF_STATE = "WRIGHT_FULL"``) onto these canonical ids.
"""

from dataclasses import dataclass, field
from typing import Callable, Optional

from .conventions import TemperatureKind, SalinityKind, PressureUnit

__all__ = ["EOSBackend", "register", "get_backend", "list_eos"]


@dataclass(frozen=True)
class EOSBackend:
    """A single equation-of-state implementation and its input conventions.

    Kernels operate on plain array-likes (scalars, numpy, dask) in the backend's
    *native* pressure unit; the :class:`~xeos.eos.EquationOfState` facade handles
    xarray wrapping and unit conversion.  ``density`` is required; analytic
    derivatives are optional (the facade falls back to finite differences).
    """

    id: str
    density: Callable
    temperature: TemperatureKind
    salinity: SalinityKind
    pressure_unit: PressureUnit
    reference: str
    description: str = ""
    # Optional analytic primitives (native units):
    drho_dt: Optional[Callable] = None  # d(rho)/d(temperature) [kg m-3 degC-1]
    drho_ds: Optional[Callable] = None  # d(rho)/d(salinity)    [kg m-3 (salt unit)-1]
    specific_volume: Optional[Callable] = None  # [m3 kg-1]
    valid_range: dict = field(default_factory=dict)


_REGISTRY: dict[str, EOSBackend] = {}


def register(backend: EOSBackend) -> EOSBackend:
    """Add ``backend`` to the global registry (keyed by ``backend.id``)."""
    if backend.id in _REGISTRY:
        raise ValueError(f"EOS id {backend.id!r} is already registered.")
    _REGISTRY[backend.id] = backend
    return backend


def get_backend(eos_id: str) -> EOSBackend:
    """Look up a registered backend by canonical id, with a helpful error."""
    try:
        return _REGISTRY[eos_id]
    except KeyError:
        raise KeyError(
            f"Unknown EOS {eos_id!r}. Available: {', '.join(sorted(_REGISTRY))}."
        ) from None


def list_eos() -> list[str]:
    """Return the sorted list of registered canonical EOS ids."""
    return sorted(_REGISTRY)
