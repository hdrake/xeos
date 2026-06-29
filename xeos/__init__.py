"""xeos: lightweight, xarray-enabled wrappers for seawater equations of state.

Pick the EOS that matches your ocean-model run — by the model's own selector
string — and apply it to xarray/dask data with a uniform API::

    import xeos
    eos = xeos.from_model("MOM6", "WRIGHT_FULL")
    density = eos.rho(theta, salt, pressure)   # theta, salt, pressure as DataArrays

See :func:`xeos.list_eos` for the available equations of state.
"""

from . import backends  # noqa: F401  (registers all built-in EOS at import time)

from .eos import EquationOfState
from .models import from_model, equation_of_state, MODEL_SELECTORS
from .api import rho, alpha, beta, specific_volume
from .registry import list_eos, get_backend
from .conventions import TemperatureKind, SalinityKind, PressureUnit
from .version import __version__

__all__ = [
    "EquationOfState",
    "from_model",
    "equation_of_state",
    "MODEL_SELECTORS",
    "rho",
    "alpha",
    "beta",
    "specific_volume",
    "list_eos",
    "get_backend",
    "TemperatureKind",
    "SalinityKind",
    "PressureUnit",
    "__version__",
]
