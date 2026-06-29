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

`wright97-full` and `linear` have no standalone Python reference; they are checked
structurally (shared-formula correctness, and `alpha`/`beta` vs finite differences)
in `test_api.py`.

## Regenerating

```bash
conda env create -f environment.yml
conda activate xeos-reference
python generate_truth.py          # rewrites truth.json (downloads polyTEOS10.py if absent)
```

Commit the updated `truth.json` (and bump versions in `environment.yml` if they
changed). `polyTEOS10.py` is git-ignored — it is third-party reference code pulled
on demand, and only the resulting numbers are committed.
