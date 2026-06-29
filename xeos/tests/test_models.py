"""Model selector-string resolution (the headline 'match my run' feature)."""

import pytest

import xeos
from xeos.models import MODEL_SELECTORS
from xeos.registry import list_eos


@pytest.mark.parametrize("model,selector,expected", [
    ("MOM6", "WRIGHT_FULL", "wright97-full"),
    ("MOM6", "WRIGHT", "wright97-reduced"),
    ("MOM6", "WRIGHT_REDUCED", "wright97-reduced"),
    ("MOM6", "LINEAR", "linear"),
    ("MOM6", "ROQUET_RHO", "teos10-poly55"),
    ("MITgcm", "JMD95Z", "jmd95"),
    ("MITgcm", "JMD95P", "jmd95"),
    ("Oceananigans", "TEOS10EquationOfState", "teos10-poly55"),
    ("Oceananigans", "LinearEquationOfState", "linear"),
])
def test_selector_resolves(model, selector, expected):
    assert xeos.from_model(model, selector).id == expected


def test_selector_is_case_and_whitespace_insensitive():
    assert xeos.from_model("mom6", "  wright_full ").id == "wright97-full"


def test_model_alias():
    assert xeos.from_model("MOM", "LINEAR").id == "linear"


def test_unknown_model_raises():
    with pytest.raises(KeyError, match="Unknown model"):
        xeos.from_model("ROMS", "LINEAR")


def test_unknown_selector_lists_options():
    with pytest.raises(KeyError, match="Supported"):
        xeos.from_model("MOM6", "NOT_AN_EOS")


def test_every_selector_points_to_registered_backend():
    registered = set(list_eos())
    for model, table in MODEL_SELECTORS.items():
        for selector, eos_id in table.items():
            assert eos_id in registered, f"{model}:{selector} -> {eos_id} not registered"


def test_from_model_passes_linear_params():
    eos = xeos.from_model("MITgcm", "LINEAR", rho0=1025.0, talpha=1e-4, sbeta=8e-4)
    assert abs(float(eos.rho(0.0, 0.0, 0.0)) - 1025.0) < 1e-9
