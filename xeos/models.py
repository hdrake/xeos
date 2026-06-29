"""Resolve an ocean model's own EOS selector string to an :class:`EquationOfState`.

The headline entry point is :func:`from_model`: pass the model name and the exact
selector you set in the model (MOM6 ``EQN_OF_STATE``, MITgcm ``eosType``, or an
Oceananigans EOS type) and get back the matching ``xeos`` equation of state, so
post-processing uses the same EOS the simulation did.
"""

from .eos import EquationOfState
from .registry import get_backend, list_eos
from .backends._linear import make_linear

__all__ = ["from_model", "equation_of_state", "MODEL_SELECTORS"]

# (model -> {normalised selector -> canonical EOS id}).  Selectors are matched
# case-insensitively with surrounding whitespace stripped.  Only Phase-1 schemes
# are listed; further selectors are added as their backends land.
MODEL_SELECTORS = {
    "MOM6": {
        "LINEAR": "linear",
        "WRIGHT_FULL": "wright97-full",
        "WRIGHT": "wright97-reduced",
        "WRIGHT_RED": "wright97-reduced",
        "WRIGHT_REDUCED": "wright97-reduced",
        "ROQUET_RHO": "teos10-poly55",
        "NEMO": "teos10-poly55",
        "TEOS10": "teos10",
    },
    "MITGCM": {
        "LINEAR": "linear",
        "JMD95Z": "jmd95",
        "JMD95P": "jmd95",
        "TEOS10": "teos10",
    },
    "OCEANANIGANS": {
        "LINEAR": "linear",
        "LINEAREQUATIONOFSTATE": "linear",
        "TEOS10": "teos10-poly55",
        "TEOS10EQUATIONOFSTATE": "teos10-poly55",
        "TEOS10SEAWATERPOLYNOMIAL": "teos10-poly55",
    },
}

# Friendly model-name aliases.
_MODEL_ALIASES = {
    "MOM": "MOM6",
    "MOM6": "MOM6",
    "MITGCM": "MITGCM",
    "MITGCMUTILS": "MITGCM",
    "OCEANANIGANS": "OCEANANIGANS",
    "OCEANANIGANS.JL": "OCEANANIGANS",
}


def equation_of_state(eos, pressure_input_unit="dbar", **params):
    """Build an :class:`EquationOfState` from a canonical id (or backend).

    ``params`` customise parameterised schemes; currently the ``linear`` EOS
    accepts ``rho0``, ``talpha`` and ``sbeta``.
    """
    if isinstance(eos, str) and eos == "linear" and params:
        backend = make_linear(**params)
        return EquationOfState(backend, pressure_input_unit=pressure_input_unit)
    if params:
        raise TypeError(f"EOS {eos!r} does not accept parameters {sorted(params)}.")
    return EquationOfState(eos, pressure_input_unit=pressure_input_unit)


def from_model(model, selector, pressure_input_unit="dbar", **params):
    """Resolve ``model``'s native ``selector`` string to an :class:`EquationOfState`.

    Examples
    --------
    >>> from_model("MOM6", "WRIGHT_FULL")        # doctest: +SKIP
    >>> from_model("MITgcm", "JMD95Z")           # doctest: +SKIP
    >>> from_model("Oceananigans", "TEOS10EquationOfState")  # doctest: +SKIP
    """
    model_key = _MODEL_ALIASES.get(str(model).strip().upper())
    if model_key is None:
        raise KeyError(
            f"Unknown model {model!r}. Known models: "
            f"{', '.join(sorted(set(_MODEL_ALIASES.values())))}."
        )
    table = MODEL_SELECTORS[model_key]
    key = str(selector).strip().upper()
    if key not in table:
        raise KeyError(
            f"Unknown {model_key} EOS selector {selector!r}. "
            f"Supported (this xeos version): {', '.join(sorted(table))}."
        )
    eos_id = table[key]
    # Validate the backend exists (guards against typos in the selector table).
    get_backend(eos_id)
    return equation_of_state(eos_id, pressure_input_unit=pressure_input_unit, **params)
