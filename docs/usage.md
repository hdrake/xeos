# Usage

`xeos` gives every equation of state the same `(t, s, p)` call signature.
Inputs may be scalars, numpy arrays, or xarray `DataArray`s (dask-backed arrays
stay lazy). Pressure is sea pressure in **dbar** by default.

## Match your model run by its selector string

The headline entry point is {func}`xeos.from_model`: pass the model name and the
exact selector you set in the model, and post-processing uses the same EOS the
simulation did.

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

## Address an EOS directly

```python
xeos.equation_of_state("jmd95").rho(t, s, p)
xeos.rho(t, s, p, eos="wright97-full")          # one-off functional form
xeos.list_eos()                                 # what's available
```

The parameterised `linear` EOS accepts `rho0`, `talpha`, and `sbeta`:

```python
eos = xeos.equation_of_state("linear", rho0=1025.0, talpha=2.0e-4, sbeta=7.6e-4)
```

## Conventions

`xeos` does **not** silently convert inputs. Each EOS declares the temperature
and salinity it expects: TEOS-10 and the Roquet polynomials use **conservative
temperature** + **absolute salinity**; the others use **potential temperature** +
**practical salinity** (see `eos.temperature` / `eos.salinity`). Explicit
conversion helpers live in {mod}`xeos.conventions` (these need the `gsw` extra).

```python
eos = xeos.equation_of_state("teos10-poly55")
eos.temperature   # TemperatureKind.CONSERVATIVE
eos.salinity      # SalinityKind.ABSOLUTE
```

## Supported equations of state

`xeos.list_eos()` returns the current set:

- **linear** — configurable (MOM6/MITgcm/Oceananigans `LINEAR`)
- **wright97-full**, **wright97-reduced** — Wright 1997 (MOM6 `WRIGHT_FULL`, `WRIGHT`/`WRIGHT_RED`)
- **jmd95** — Jackett & McDougall 1995 (MITgcm `JMD95Z`/`JMD95P`)
- **unesco** — UNESCO/EOS-80 (MOM6 `UNESCO`/`JACKETT_MCD`, MITgcm `UNESCO`)
- **mdjwf** — McDougall et al. 2003 (MITgcm `MDJWF`)
- **teos10-poly55** — Roquet 55-term polynomial / TEOS-10 density form
  (Oceananigans `TEOS10EquationOfState`, MOM6 `ROQUET_RHO`/`NEMO`)
- **roquet-{linear,cabbeling,cabbeling-thermobaricity,freezing,second-order,simplest-realistic}**
  — idealized second-order Roquet forms (Oceananigans `RoquetSeawaterPolynomial(:…)`)
- **teos10** — full TEOS-10 via `gsw` (MOM6/MITgcm `TEOS10`)
