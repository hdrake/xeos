---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: Python 3
  name: python3
---

# Reproducing a model's own EOS: σ₂ from an MITgcm run

Ocean models compute density with a specific equation of state (EOS). If your
post-processing uses a *different* EOS than the run did, every derived
quantity — density, potential density, thermal-expansion and haline-contraction
coefficients, water-mass-transformation diagnostics — inherits a silent bias.
The bias is small in an absolute sense (often ≲ 0.1 kg m⁻³) but it is
*systematic*, and it can be comparable to the density signals analysts care
about.

This example takes a snapshot from an MITgcm run, computes potential density
referenced to 2000 dbar (σ₂) with **the run's own EOS**, and compares it against
what you would get from **TEOS-10** — a different EOS with different
temperature/salinity conventions. The point is not that one is "right": it is
that the two disagree by a reproducible amount, and `xeos` lets you match
whichever one your simulation actually used.

## The dataset

The data is a single snapshot from the MITgcm
[`tutorial_global_oce_latlon`](https://mitgcm.readthedocs.io/) experiment
(the coarse `global_ocean.90x40x15` global configuration): potential
temperature `THETA` (°C), practical salinity `SALT` (psu), on a 90×40×15
longitude/latitude/depth grid. We fetch it from Zenodo with `pooch` (the hash
is pinned, so the download is verified and cached).

```{code-cell} ipython3
import pooch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import xeos
from xeos.conventions import to_absolute_salinity, to_conservative_temperature

path = pooch.retrieve(
    url="doi:10.5281/zenodo.21251181/mitgcm_example_dataset_v2.nc",
    known_hash="md5:3176deb0b4d556d2981dff03ba2ed624",
)
ds = xr.open_dataset(path)
ds
```

The grid carries a land/ocean mask (`maskC`). We apply it up front so that
land cells (where `SALT` is stored as 0) never enter the EOS — feeding a
salinity of 0 to a seawater polynomial is both physically meaningless and a
source of spurious values.

```{code-cell} ipython3
ocean = ds.maskC                      # True in wet cells, broadcasts over time
theta = ds.THETA.where(ocean)         # potential temperature [degC]
salt = ds.SALT.where(ocean)           # practical salinity [psu]
```

## EOS #1 — the run's actual EOS (JMD95Z)

That MITgcm configuration sets `eosType = 'JMD95Z'`, the Jackett & McDougall
(1995) potential-temperature fit. We select it *by the model's own selector
string*, so post-processing uses exactly the density the model integrated.

```{note}
The Zenodo record and MITgcm docs attribute this snapshot to
`tutorial_global_oce_latlon`, whose `input/data` sets `eosType = 'JMD95Z'`
(pressure ≈ depth in dbar). A separate `global_ocean.90x40x15` verification
directory uses `JMD95P` instead. Both are the same JMD95 fit and both resolve
to xeos's `jmd95` backend, so the σ₂ here is insensitive to that attribution
detail — but it is worth flagging.
```

σ₂ is potential density referenced to 2000 dbar, i.e. density evaluated at a
reference pressure of 2000 dbar, minus 1000. `xeos` takes pressure in dbar by
default, so we pass the scalar `2000.0`.

```{code-cell} ipython3
eos1 = xeos.from_model("MITgcm", "JMD95Z")
eos1
```

```{code-cell} ipython3
sigma2_jmd95 = eos1.rho(theta, salt, 2000.0) - 1000.0
sigma2_jmd95.attrs["long_name"] = r"$\sigma_2$ (JMD95)"
sigma2_jmd95.attrs["units"] = "kg m-3"
sigma2_jmd95
```

## EOS #2 — TEOS-10, with the right conventions

TEOS-10 does **not** speak potential temperature and practical salinity: it
takes **conservative temperature** and **absolute salinity**. `xeos` never
converts inputs silently, so we convert explicitly with `xeos.conventions`
(these helpers use `gsw`). Absolute salinity depends on location, so we pass the
grid longitude/latitude (`XC`/`YC`); xarray broadcasts the 1-D coordinates
against the 3-D `THETA`/`SALT` fields automatically. We use the same 2000 dbar
reference pressure as σ₂.

```{code-cell} ipython3
SA = to_absolute_salinity(salt, 2000.0, lon=ds.XC, lat=ds.YC)
CT = to_conservative_temperature(theta, salt, 2000.0, lon=ds.XC, lat=ds.YC)
SA.name, CT.name = "SA", "CT"
print("SA dims:", SA.dims, "  CT dims:", CT.dims)
```

```{code-cell} ipython3
eos2 = xeos.equation_of_state("teos10")
eos2
```

```{code-cell} ipython3
sigma2_teos10 = eos2.rho(CT, SA, 2000.0) - 1000.0
sigma2_teos10.attrs["long_name"] = r"$\sigma_2$ (TEOS-10)"
sigma2_teos10.attrs["units"] = "kg m-3"
sigma2_teos10
```

## How much does the EOS choice matter?

```{code-cell} ipython3
diff = sigma2_teos10 - sigma2_jmd95
print(f"sigma2 range (JMD95):   {float(sigma2_jmd95.min()):.3f} .. "
      f"{float(sigma2_jmd95.max()):.3f} kg/m^3")
print(f"sigma2 range (TEOS-10): {float(sigma2_teos10.min()):.3f} .. "
      f"{float(sigma2_teos10.max()):.3f} kg/m^3")
print(f"TEOS-10 minus JMD95:    {float(diff.min()):.4f} .. "
      f"{float(diff.max()):.4f} kg/m^3")
```

The two σ₂ fields agree to a few hundredths of a kg m⁻³. That is small — but not
negligible: it is the same order as the density contrast across a weak front,
and being *systematic* it does not average away. Let us look at where it lives.

```{code-cell} ipython3
# Surface layer (Z index 0), single time snapshot.
s1 = sigma2_jmd95.isel(time=0, Z=0)
s2 = sigma2_teos10.isel(time=0, Z=0)
d = (s2 - s1)

vmin = float(min(s1.min(), s2.min()))
vmax = float(max(s1.max(), s2.max()))
dmax = float(abs(d).max())

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
for ax, field, title in ((axes[0], s1, "JMD95Z (the run's EOS)"),
                         (axes[1], s2, "TEOS-10")):
    pc = ax.pcolormesh(ds.XC, ds.YC, field, shading="auto",
                       vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("longitude")
axes[0].set_ylabel("latitude")
fig.colorbar(pc, ax=axes, label=r"surface $\sigma_2$  [kg m$^{-3}$]",
             shrink=0.9)
plt.show()
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
pc = ax.pcolormesh(ds.XC, ds.YC, d, shading="auto",
                   vmin=-dmax, vmax=dmax, cmap="RdBu_r")
ax.set_title(r"TEOS-10 $-$ JMD95Z  (surface $\sigma_2$)")
ax.set_xlabel("longitude")
ax.set_ylabel("latitude")
fig.colorbar(pc, ax=ax, label=r"$\Delta\sigma_2$  [kg m$^{-3}$]")
plt.show()
```

## Takeaway

The EOS-choice difference is spatially structured — it tracks the temperature
and salinity distribution, so it projects onto exactly the gradients that drive
circulation and water-mass diagnostics rather than acting as a constant offset.
Whenever you compute density (or α, β, neutral surfaces, transformation rates)
from model output, use the EOS the model used. With `xeos` that is a one-liner:
name your model and its selector string, and the numbers match the simulation
by construction — no silent EOS substitution.
