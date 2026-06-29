# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`xeos` provides lightweight, xarray/dask-aware wrappers over many seawater
Equations of State (EOS), so analysts can apply the *same* EOS their ocean-model
run used (MOM6, Oceananigans, MITgcm) — selected by the model's own selector
string. The polynomial/rational EOS are vendored as numpy kernels; core runtime
deps are **numpy + xarray only**. Full TEOS-10 is delegated to the optional `gsw`
extra (`pip install xeos[teos10]`).

## Commands

```bash
pip install -e .[test]                              # editable install + gsw + pytest
pytest                                              # full suite
pytest xeos/tests/test_backends.py                  # cross-validation vs frozen truth
pytest xeos/tests/test_models.py::test_selector_resolves   # a single test
pylint xeos && black xeos                           # lint / format (declared in ci/environment.yml)
```

CI (`.github/workflows/ci.yml`) builds a conda env from `ci/environment.yml`,
does an editable install, and runs `pytest` across Python 3.11–3.14.

## Architecture

The package is a small registry-and-facade design. Data flows:
`from_model(model, selector)` → canonical EOS id → registered `EOSBackend`
→ wrapped in an `EquationOfState` facade → kernels dispatched through xarray.

- **`registry.py`** — `EOSBackend` dataclass (a `density` kernel + optional
  analytic `drho_dt`/`drho_ds` + convention metadata) and a global registry.
  Backends self-register at import time.
- **`backends/`** — one module per EOS; importing `backends/__init__.py`
  registers them all. Vendored kernels: `_linear`, `_wright` (full + reduced
  coefficient sets, native pressure **Pa**), `_jmd95` and `_unesco` (native
  **dbar**; `_unesco` reuses `_jmd95._rho_surface`), `_mdjwf` (rational fit,
  **dbar**), `_roquet` (55-term TEOS-10 density polynomial, **dbar**≈depth),
  `_roquet_spv` (55-term specific-volume form, MOM6 `ROQUET_SPV`), and
  `_roquet_idealized` (6 second-order Roquet forms via one factory, conservative
  temp / absolute salinity, Z = −p). `_teos10` is a thin lazy-`gsw` wrapper.
- **`eos.py`** — `EquationOfState` facade. Converts user pressure (default dbar)
  to each backend's native unit, dispatches via `xarray_utils.apply_eos`, and
  computes `alpha = -drho_dt/rho`, `beta = drho_ds/rho` from analytic derivatives,
  falling back to centred finite differences when a backend supplies none.
- **`models.py`** — `MODEL_SELECTORS` alias table mapping each model's selector
  strings (MOM6 `EQN_OF_STATE`, MITgcm `eosType`, Oceananigans EOS types) to
  canonical ids; `from_model()` and `equation_of_state()` (the latter handles
  parameterised schemes like `linear`).
- **`conventions.py`** — `TemperatureKind`/`SalinityKind`/`PressureUnit` enums
  and optional gsw-backed conversion helpers. **xeos never silently converts
  inputs**; backends declare the kinds they expect. TEOS-10 + Roquet take
  conservative temperature / absolute salinity; everything else potential
  temperature / practical salinity.
- **`xarray_utils.py`** — `apply_eos` wraps kernels with
  `xr.apply_ufunc(dask="parallelized")` when any input is a DataArray (preserving
  labels/dask, attaching CF attrs); otherwise calls the kernel on numpy arrays.

### Adding an EOS
Drop a `backends/_name.py` that builds an `EOSBackend` and calls `register(...)`,
import it in `backends/__init__.py`, and add its model selector strings to
`MODEL_SELECTORS`. Tests iterating the registry / truth fixtures pick it up.

### Critical gotchas
- **Pressure units differ per backend** (Wright = Pa, JMD95/Roquet/gsw = dbar);
  the facade converts, so a kernel always receives its declared native unit.
- **Wright variants share one formula, differ only in coefficients.** `momlevel`
  implements the *reduced*-range coefficients (= MOM6 `WRIGHT`/`WRIGHT_RED`), not
  `WRIGHT_FULL`; that is why the reference fixture validates `wright97-reduced`
  and `wright97-full` is checked structurally instead.
- **`ROQUET_SPV` uses `deltaS=24`, NOT 32.** The widely-used `polyTEOS10.py`
  reference has a typo in its `polyTEOS10_55t` routine (`deltaS=32`, copied from the
  density form), making its specific-volume output disagree with its own published
  check values. `_roquet_spv.py` uses the correct `24` and is validated against
  MOM6's authoritative Fortran (`MOM_EOS_Roquet_SpV.F90`), not the buggy Python —
  see `xeos/tests/reference/_build_roquet_spv_fortran.py`. (Bug reported upstream.)
- **Not yet implemented:** MOM6 `JACKETT_06`, MOM6 `WRIGHT` legacy-buggy, MITgcm
  `POLY3` (per-level runtime coefficients), MITgcm `IDEALGAS`. Add as new
  `backends/_*.py` + selector entries when a trustworthy reference is available.

## Testing & reference truth

`test_backends.py` validates each vendored kernel against frozen values in
`xeos/tests/reference/truth.json`. Those values are generated from authoritative
reference packages (`gsw`, `fastjmd95`, `momlevel`, `polyTEOS10`) by
`xeos/tests/reference/generate_truth.py` in a **separate pinned environment**
(`xeos/tests/reference/environment.yml`) — kept apart precisely because some
references (momlevel) are heavyweight and intentionally excluded from xeos's
runtime deps. The committed `truth.json` is stamped with the exact reference
versions used; the regular test suite reads it and needs none of those packages.
Regenerate via `xeos/tests/reference/README.md` and commit the updated JSON.

## Versioning

Single-sourced in `xeos/version.py`, read by hatchling (`[tool.hatch.version]`).
