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
- **roquet-spv** — Roquet 55-term specific-volume form (MOM6 `ROQUET_SPV`)
- **roquet-{linear,cabbeling,cabbeling-thermobaricity,freezing,second-order,simplest-realistic}**
  — idealized second-order Roquet forms (Oceananigans `RoquetSeawaterPolynomial(:…)`)
- **teos10** — TEOS-10 via `gsw` (MOM6/MITgcm `TEOS10`). See the note below: this
  is itself the Roquet 75-term polynomial, not the exact Gibbs function.

## The TEOS-10 / Roquet family

"Roquet's EOS" refers to two very different things, and `xeos` ships both. Neither
is a different *thermodynamics* from TEOS-10 — they are polynomial **fits** to it.

| EOS | What it is | Accuracy vs TEOS-10 |
|-----|------------|---------------------|
| canonical TEOS-10 (Gibbs function) | density as a derivative of one thermodynamic potential `g(SA, T, p)`; globally valid but expensive. `gsw`'s `*_t_exact` routines. | exact (the standard) |
| `teos10` (`gsw.rho`) | the **75-term** Roquet specific-volume polynomial — what GSW actually uses by default | ~10⁻³ kg m⁻³ inside the "oceanographic funnel" |
| `teos10-poly55` | the **55-term** Roquet density polynomial (`bsq` form) | ~10⁻²–10⁻³ kg m⁻³ |
| `roquet-spv` | the **55-term** Roquet *specific-volume* polynomial (non-Boussinesq MOM6) | ~10⁻²–10⁻³ kg m⁻³ |
| `roquet-{linear,cabbeling,…}` | deliberately simplified ≤2nd-order forms that isolate one nonlinear effect (cabbeling, thermobaricity) for process studies | crude by design — not meant to be accurate |

```{important}
What `xeos` (and most analysis code) calls "full TEOS-10" via `gsw.rho` is **already
a Roquet polynomial** — the 75-term specific-volume fit of Roquet et al. (2015),
adopted by GSW as its standard fast implementation. The genuinely exact
Gibbs-function density is `gsw.rho_t_exact`. So the practical difference between
`teos10` and `teos10-poly55` is just **75-term vs 55-term** fits of the same target,
agreeing to ~10⁻² kg m⁻³ — not "exact vs approximate".
```

The accurate polynomials (`teos10*`, `roquet-spv`) use conservative temperature and
absolute salinity, are fitted only inside the realistic T–S–p "funnel", and are
~10× cheaper than the Gibbs function (no transcendentals) — which is why ocean
models use them. The idealized forms trade nearly all accuracy for interpretability
and should only be used to reproduce a run that itself used them.

### References

- Roquet, Madec, McDougall & Barker (2015), *Ocean Modelling* **90**, 29–43 — the
  accurate 55-/75-term polynomials (`teos10`, `teos10-poly55`, `roquet-spv`).
- Roquet, Madec, McDougall & Barker (2015), *J. Phys. Oceanogr.* **45**, 2564–2579 —
  the idealized simplified forms (`roquet-{linear,cabbeling,…}`).
- IOC, SCOR & IAPSO (2010); McDougall & Barker (2011) — the TEOS-10 standard / GSW.
