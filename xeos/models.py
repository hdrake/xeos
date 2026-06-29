"""Resolve an ocean model's own EOS selector string to an :class:`EquationOfState`.

The headline entry point is :func:`from_model`: pass the model name and the exact
selector you set in the model (MOM6 ``EQN_OF_STATE``, MITgcm ``eosType``,
MPAS-Ocean ``config_eos_type``, or an Oceananigans EOS type) and get back the
matching ``xeos`` equation of state, so post-processing uses the same EOS the
simulation did.
"""

from .eos import EquationOfState
from .registry import get_backend, list_eos
from .backends._linear import make_linear

__all__ = ["from_model", "equation_of_state", "MODEL_SELECTORS"]

# (model -> {normalised selector -> canonical EOS id}).  Selectors are matched
# case-insensitively with surrounding whitespace stripped.  Further selectors are
# added as their backends land.
MODEL_SELECTORS = {
    "MOM6": {
        "LINEAR": "linear",
        "WRIGHT_FULL": "wright97-full",
        "WRIGHT": "wright97-reduced",
        "WRIGHT_RED": "wright97-reduced",
        "WRIGHT_REDUCED": "wright97-reduced",
        # MOM6's "UNESCO" is, despite the name, the Jackett & McDougall (1995)
        # potential-temperature fit -- byte-for-byte the JMD95 kernel, NOT the
        # original Fofonoff & Millard EOS-80 (that is xeos's `unesco`, = MITgcm's
        # UNESCO). Verified to machine precision against MOM_EOS_UNESCO.F90; the
        # two differ by up to ~0.4 kg/m3. See reference/_build_unesco_fortran.py.
        "UNESCO": "jmd95",
        "JACKETT_MCD": "jmd95",
        "ROQUET_RHO": "teos10-poly55",
        "NEMO": "teos10-poly55",
        "ROQUET_SPV": "roquet-spv",
        "TEOS10": "teos10",
    },
    "MITGCM": {
        "LINEAR": "linear",
        "UNESCO": "unesco",
        "JMD95Z": "jmd95",
        "JMD95P": "jmd95",
        "MDJWF": "mdjwf",
        "TEOS10": "teos10",
    },
    "MPAS": {
        "LINEAR": "mpas-linear",
        "JM": "mpas-jm",
        "WRIGHT": "mpas-wright",
    },
    "OCEANANIGANS": {
        "LINEAR": "linear",
        "LINEAREQUATIONOFSTATE": "linear",
        "TEOS10": "teos10-poly55",
        "TEOS10EQUATIONOFSTATE": "teos10-poly55",
        "TEOS10SEAWATERPOLYNOMIAL": "teos10-poly55",
        # Idealized second-order Roquet forms, RoquetSeawaterPolynomial(:X).
        "ROQUETLINEAR": "roquet-linear",
        "LINEARROQUET": "roquet-linear",
        "CABBELING": "roquet-cabbeling",
        "CABBELINGTHERMOBARICITY": "roquet-cabbeling-thermobaricity",
        "FREEZING": "roquet-freezing",
        "SECONDORDER": "roquet-second-order",
        "SIMPLESTREALISTIC": "roquet-simplest-realistic",
    },
}

# Friendly model-name aliases.
_MODEL_ALIASES = {
    "MOM": "MOM6",
    "MOM6": "MOM6",
    "MITGCM": "MITGCM",
    "MITGCMUTILS": "MITGCM",
    "MPAS": "MPAS",
    "MPAS-O": "MPAS",
    "MPASO": "MPAS",
    "MPAS-OCEAN": "MPAS",
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
    >>> from_model("MPAS-Ocean", "jm")           # doctest: +SKIP
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
