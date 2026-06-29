# xeos

**Lightweight, xarray-enabled wrappers for seawater equations of state.**

Ocean models (MOM6, MITgcm, MPAS-Ocean, Oceananigans) differ in the equation of
state (EOS) they use, and many let you change it at run time. Python post-processing then
often applies a *different* EOS than the simulation did, silently corrupting
derived quantities like density, thermal expansion, and water-mass transformation
diagnostics. `xeos` lets you pick the EOS that matches your run — **by the model's
own selector string** — and apply it to xarray/dask data through one uniform API.

It stays lightweight on purpose: the polynomial/rational equations of state are
vendored as small numpy kernels, so the core install needs only **numpy + xarray**.
TEOS-10 (via `gsw`) is an optional extra.

```python
import xeos

# MOM6 run with EQN_OF_STATE = "WRIGHT_FULL"
eos = xeos.from_model("MOM6", "WRIGHT_FULL")
rho = eos.rho(theta, salt, pressure)     # xarray DataArrays in, labeled DataArray out
```

```{toctree}
:maxdepth: 2
:caption: Contents

installation
usage
api
```
