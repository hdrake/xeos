# xeos

**Lightweight, xarray-enabled wrappers for seawater equations of state.**

Ocean models (MOM6, Oceananigans, MITgcm) differ in the equation of state (EOS)
they use, and many let you change it at run time. Python post-processing then
often applies a *different* EOS than the simulation did, silently corrupting
derived quantities like density, thermal expansion, and water-mass transformation
diagnostics. `xeos` lets you pick the EOS that matches your run — **by the model's
own selector string** — and apply it to xarray/dask data through one uniform API.

It stays lightweight on purpose: the polynomial/rational equations of state are
vendored as small numpy kernels, so the core install needs only **numpy + xarray**.
Full TEOS-10 (via `gsw`) is an optional extra.

## Install

```bash
pip install xeos              # core (numpy + xarray): all vendored EOS
pip install xeos[teos10]      # adds full TEOS-10 via gsw
pip install xeos[complete]    # gsw + numba acceleration
```

## Usage

Match your model run by its selector string:

```python
import xeos

# MOM6 run with EQN_OF_STATE = "WRIGHT_FULL"
eos = xeos.from_model("MOM6", "WRIGHT_FULL")
rho = eos.rho(theta, salt, pressure)     # xarray DataArrays in, labeled DataArray out
a   = eos.alpha(theta, salt, pressure)   # thermal expansion
b   = eos.beta(theta, salt, pressure)    # haline contraction

# MITgcm eosType = 'JMD95Z'
xeos.from_model("MITgcm", "JMD95Z").rho(theta, salt, p)

# Oceananigans TEOS10EquationOfState
xeos.from_model("Oceananigans", "TEOS10EquationOfState").rho(CT, SA, p)
```

Or address an EOS directly:

```python
xeos.equation_of_state("jmd95").rho(t, s, p)
xeos.rho(t, s, p, eos="wright97-full")          # one-off functional form
xeos.list_eos()                                  # what's available
```

Inputs may be scalars, numpy arrays, or xarray `DataArray`s (dask-backed arrays
stay lazy). Pressure is sea pressure in **dbar** by default.

## Conventions

`xeos` does **not** silently convert inputs. Each EOS declares the temperature and
salinity it expects: TEOS-10 and the Roquet polynomials use **conservative
temperature** + **absolute salinity**; the others use **potential temperature** +
**practical salinity** (see `eos.temperature` / `eos.salinity`). Explicit
conversion helpers live in `xeos.conventions` (these need the `gsw` extra).

## Supported equations of state

Phase 1 (implemented): linear, Wright 1997 (full & reduced range), Jackett &
McDougall 1995, the Roquet 55-term TEOS-10 polynomial (Oceananigans /
SeawaterPolynomials.jl), and full TEOS-10 via `gsw`. Further MOM6/MITgcm/Oceananigans
schemes (UNESCO, Jackett 2006, MDJWF, the Roquet specific-volume and idealized
forms, POLY3) are planned and slot into the same registry.

## Development

```bash
pip install -e .[test]
pytest                       # validates vendored kernels against frozen fixtures
```

Test "truth" values are generated from authoritative reference packages in a
pinned, separate environment and frozen into `xeos/tests/reference/truth.json`;
the test suite reads that file and stays lightweight. See
[`xeos/tests/reference/README.md`](xeos/tests/reference/README.md) to regenerate.
