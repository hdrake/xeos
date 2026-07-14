"""API ergonomics: bare-dask laziness, model/selector introspection, accelerate kwarg."""

import numpy as np
import pytest

import xeos
from xeos.xarray_utils import apply_eos


def test_bare_dask_array_stays_lazy_in_apply_eos():
    da = pytest.importorskip("dask.array")

    def kernel(t, s, p):
        return t + s + np.sqrt(p)

    t = da.from_array(np.array([1.0, 2.0, 3.0]), chunks=1)
    s = da.from_array(np.array([10.0, 20.0, 30.0]), chunks=1)
    p = da.from_array(np.array([4.0, 9.0, 16.0]), chunks=1)
    out = apply_eos(kernel, t, s, p)
    assert isinstance(out, da.Array)  # still lazy, not eagerly materialised
    eager = kernel(np.asarray(t), np.asarray(s), np.asarray(p))
    np.testing.assert_allclose(out.compute(), eager)


def test_bare_dask_array_stays_lazy_in_eos_rho():
    da = pytest.importorskip("dask.array")
    eos = xeos.equation_of_state("jmd95")
    t = da.from_array(np.array([5.0, 10.0, 15.0]), chunks=1)
    s = da.from_array(np.array([35.0, 35.5, 36.0]), chunks=1)
    p = da.from_array(np.array([0.0, 500.0, 1000.0]), chunks=1)
    lazy = eos.rho(t, s, p)
    assert isinstance(lazy, da.Array)  # bare dask in -> bare dask out, still lazy
    eager = eos.rho(np.array([5.0, 10.0, 15.0]),
                    np.array([35.0, 35.5, 36.0]),
                    np.array([0.0, 500.0, 1000.0]))
    np.testing.assert_allclose(lazy.compute(), np.asarray(eager))


def test_scalar_and_numpy_unaffected_by_dask_branch():
    eos = xeos.equation_of_state("jmd95")
    out = eos.rho(3.0, 35.5, 3000.0)
    assert np.isclose(float(out), 1041.83267, atol=1e-4)
    arr = eos.rho(np.array([3.0, 3.0]), np.array([35.5, 35.5]), np.array([3000.0, 3000.0]))
    np.testing.assert_allclose(np.asarray(arr), 1041.83267, atol=1e-4)


def test_list_models():
    assert xeos.list_models() == ["MITGCM", "MOM6", "MPAS", "OCEANANIGANS"]


def test_selectors_for_returns_mapping():
    assert xeos.selectors_for("MOM6")["WRIGHT_FULL"] == "wright97-full"
    assert xeos.selectors_for("MITGCM")["JMD95Z"] == "jmd95"


def test_selectors_for_alias_normalisation():
    # Same friendly aliases from_model accepts.
    assert xeos.selectors_for("mom6") == xeos.selectors_for("MOM6")
    assert xeos.selectors_for("MITgcm") == xeos.selectors_for("MITGCM")
    assert xeos.selectors_for("MPAS-Ocean") == xeos.selectors_for("MPAS")


def test_selectors_for_returns_a_copy():
    got = xeos.selectors_for("MOM6")
    got["WRIGHT_FULL"] = "tampered"
    assert xeos.selectors_for("MOM6")["WRIGHT_FULL"] == "wright97-full"


def test_selectors_for_unknown_model_raises_keyerror():
    with pytest.raises(KeyError) as exc:
        xeos.selectors_for("NotAModel")
    assert "Known models" in str(exc.value)


def test_selector_table_shape_and_backends_registered():
    table = xeos.selector_table()
    assert table == sorted(table)  # sorted (model, selector, eos_id) rows
    registered = set(xeos.list_eos())
    for row in table:
        assert len(row) == 3
        model, selector, eos_id = row
        assert isinstance(model, str) and isinstance(selector, str)
        assert eos_id in registered, (model, selector, eos_id)
    # Every selector across every model is represented.
    total = sum(len(xeos.selectors_for(m)) for m in xeos.list_models())
    assert len(table) == total


def test_accelerate_default_forwards_end_to_end():
    eos = xeos.from_model("MITgcm", "JMD95Z", accelerate=False)
    assert np.isclose(float(eos.rho(10.0, 35.0, 0.0)),
                      float(xeos.from_model("MITgcm", "JMD95Z").rho(10.0, 35.0, 0.0)))


def test_accelerate_kwarg_accepted_by_all_shims():
    # accelerate=False must be accepted/forwarded without error by every entry point.
    assert np.isfinite(float(xeos.equation_of_state("jmd95", accelerate=False).rho(10.0, 35.0, 0.0)))
    assert np.isfinite(float(xeos.rho(10.0, 35.0, 0.0, eos="jmd95", accelerate=False)))
    assert np.isfinite(float(xeos.alpha(10.0, 35.0, 0.0, eos="jmd95", accelerate=False)))
    assert np.isfinite(float(xeos.beta(10.0, 35.0, 0.0, eos="jmd95", accelerate=False)))
    assert np.isfinite(float(xeos.specific_volume(10.0, 35.0, 0.0, eos="jmd95", accelerate=False)))


def test_accelerate_forwards_for_linear_parameterised():
    eos = xeos.equation_of_state("linear", accelerate=False,
                                 rho0=1027.0, talpha=2.0e-4, sbeta=7.4e-4)
    assert np.isclose(float(eos.rho(0.0, 0.0, 0.0)), 1027.0)
