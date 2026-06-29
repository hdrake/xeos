# Reference truth fixtures

`truth.json` holds frozen density / `alpha` / `beta` values from the authoritative
reference packages, used by `test_backends.py` to confirm that xeos's vendored
kernels reproduce their sources to machine precision.

**The test suite reads only `truth.json`** — it does *not* import the reference
packages. That keeps CI lightweight (xeos's runtime deps are just numpy + xarray)
while the validation stays anchored to authoritative implementations.

## Provenance

Every regeneration stamps `truth.json` with the exact reference-package versions
used (`provenance.reference_versions`). The grid of input `(t, s, p)` points and
the inputs themselves are stored alongside the expected outputs.

| Reference package      | xeos backend validated | Notes |
|------------------------|------------------------|-------|
| `fastjmd95`            | `jmd95`                | density |
| `momlevel.eos.wright`  | `wright97-reduced`     | density + alpha/beta; same functional form as `wright97-full`, so this validates the shared code path |
| `polyTEOS10.py`        | `teos10-poly55`        | density + alpha/beta; downloaded at generation time (not committed) |
| `gsw`                  | `teos10`               | density + alpha/beta |
| `MITgcmutils.density`  | `unesco`, `mdjwf`      | density + `alpha`/`beta` (centred-difference of the reference density, validating xeos's own FD path) |
| MOM6 `MOM_EOS_Roquet_SpV.F90` (compiled with gfortran) | `roquet-spv` | density + `alpha`/`beta`; built/run on demand by `_build_roquet_spv_fortran.py` (MOM6 source not committed) |

The following have no standalone Python reference and are checked structurally in
`test_api.py`: `wright97-full` (shared Wright formula, plus a gsw sanity bound of
0.5 kg/m3 over the oceanographic range to catch gross coefficient typos), `linear`,
and the idealized second-order Roquet forms (`roquet-linear`, `roquet-cabbeling`,
`roquet-cabbeling-thermobaricity`, `roquet-freezing`, `roquet-second-order`,
`roquet-simplest-realistic`; one hand-computed check value each plus exact analytic
`alpha`/`beta` vs finite differences).

> **Note on `roquet-spv`:** the widely-used `polyTEOS10.py` reference is unusable
> here — its `polyTEOS10_55t` routine has a `deltaS=32` typo (should be `24`) that
> makes its specific-volume output disagree with its own published check values
> (reported upstream: fabien-roquet/polyTEOS#2). `roquet-spv` is therefore validated against MOM6's
> authoritative Fortran (`MOM_EOS_Roquet_SpV.F90`), compiled with gfortran. The
> driver self-checks the published specvol value (9.732820466e-04 at SA=30, CT=10,
> p=1000 dbar) before emitting truth, so upstream reformatting can't silently
> corrupt the fixtures. Requires `gfortran` (in this folder's `environment.yml`).

## Regenerating

```bash
conda env create -f environment.yml
conda activate xeos-reference
python generate_truth.py          # rewrites truth.json (downloads polyTEOS10.py if absent)
```

Commit the updated `truth.json` (and bump versions in `environment.yml` if they
changed). `polyTEOS10.py` is git-ignored — it is third-party reference code pulled
on demand, and only the resulting numbers are committed.
