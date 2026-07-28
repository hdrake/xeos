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

## Install

```bash
pip install xeos              # core (numpy + xarray): all vendored EOS
pip install xeos[teos10]      # adds TEOS-10 via gsw
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

# MPAS-Ocean config_eos_type = 'jm'
xeos.from_model("MPAS-Ocean", "jm").rho(theta, salt, p)

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

`xeos.list_eos()` returns the current set. As of now:

- **linear** — configurable (MOM6/MITgcm/Oceananigans `LINEAR`)
- **wright97-full**, **wright97-reduced** — Wright 1997 (MOM6 `WRIGHT_FULL`,
  `WRIGHT_RED`/`WRIGHT_REDUCED`). The bare MOM6 `WRIGHT` selector also resolves to
  `wright97-reduced` **with a warning**: it is the *corrected* reduced fit, whereas
  MOM6's legacy `WRIGHT` runs the uncorrected kernel (see "not yet implemented").
- **jmd95** — Jackett & McDougall 1995 (MITgcm `JMD95Z`/`JMD95P`; **also** MOM6
  `UNESCO`/`JACKETT_MCD`, which are this fit — *not* EOS-80)
- **unesco** — UNESCO/EOS-80, Fofonoff & Millard 1983 (MITgcm `UNESCO`)
- **mdjwf** — McDougall et al. 2003 (MITgcm `MDJWF`)
- **teos10-poly55** — Roquet 55-term polynomial / TEOS-10 density form
  (Oceananigans `TEOS10EquationOfState`, MOM6 `ROQUET_RHO`/`NEMO`)
- **roquet-spv** — Roquet 55-term specific-volume form (MOM6 `ROQUET_SPV`)
- **roquet-{linear,cabbeling,cabbeling-thermobaricity,freezing,second-order,simplest-realistic}**
  — idealized second-order Roquet forms (Oceananigans `RoquetSeawaterPolynomial(:…)`)
- **mpas-linear**, **mpas-jm**, **mpas-wright** — MPAS-Ocean / E3SM
  `config_eos_type` = `linear`/`jm`/`wright`; `mpas-jm` and `mpas-wright` reuse the
  `jmd95` and `wright97-reduced` kernels (MPAS-O's `jm`/`wright` are the same EOS)
- **teos10** — TEOS-10 via `gsw` (its 75-term Roquet polynomial, not the exact
  Gibbs function; MOM6/MITgcm `TEOS10`)

Not yet implemented (planned, slot into the same registry): MOM6 `JACKETT_06` and
the legacy-buggy `WRIGHT` kernel (the bare `WRIGHT` selector currently resolves to
the corrected `wright97-reduced` with a warning), and MITgcm `POLY3` (per-level
runtime coefficients).

Full literature references with DOIs are in the
[usage docs](docs/usage.md#references).

## Development

```bash
pip install -e .[test]
pytest                       # validates vendored kernels against frozen fixtures
```

Test "truth" values are generated from authoritative reference packages in a
pinned, separate environment and frozen into `xeos/tests/reference/truth.json`;
the test suite reads that file and stays lightweight. See
[`xeos/tests/reference/README.md`](xeos/tests/reference/README.md) to regenerate.

## Releasing

**The git tag is the version.** `xeos` has no version string checked into the
source tree: `hatch-vcs` derives it from the tag at build time and writes
`xeos/_version.py` (gitignored, but shipped inside the sdist and wheel). To cut
a release you tag; there is no file to bump and nothing to keep in sync.

1. Make sure `main` is green and has everything you want in the release.
2. **Publish a GitHub Release** whose tag is `vX.Y.Z`, targeting the commit you
   want to ship:
   ```bash
   gh release create vX.Y.Z --target "$(git rev-parse origin/main)" \
     --title vX.Y.Z --generate-notes
   ```
   Publishing it (not merely pushing a tag) is what fires the workflow.
3. The **Publish to PyPI** workflow builds from that tag and uploads to PyPI via
   Trusted Publishing (OIDC — no token secret). It checks out with
   `fetch-depth: 0` so the tag is visible to `hatch-vcs`, and asserts that the
   built version matches the tag before publishing.
4. Verify: <https://pypi.org/project/xeos/>.
5. conda-forge builds from the PyPI sdist and lags by design — see
   [`conda/README.md`](conda/README.md). After the first release the autotick
   bot opens the version-bump PR for you.

To rehearse without publishing, run the workflow manually
(**Actions → Publish to PyPI → Run workflow**); on `workflow_dispatch` it builds
and can optionally push to TestPyPI, but never to PyPI.

### If a release fails

PyPI **permanently** reserves a filename: a given `xeos-X.Y.Z-*.whl` can be
uploaded exactly once, and deleting the release does not free it. So a failed
publish is almost always fixed by *retagging*, not by re-running the job.

- **`400 File already exists`** — the build produced a version that is already
  on PyPI. Under tag-derived versioning this means the tag itself is wrong or
  duplicated. Check what was built:
  `gh run view <run-id> --log | grep 'Uploading xeos-'`.
- **Version looks like `0.0.0.dev...` or `X.Y.Z.devN+g<sha>`** — the tag was not
  reachable from the checkout. Either the release was created before the tag
  existed, or a `fetch-depth: 0` was dropped from the workflow.
- **To move a tag that was published at the wrong commit**, delete the release
  and its tag, then recreate both at the right commit — re-running the failed
  job would just rebuild the same wrong version:
  ```bash
  gh release delete vX.Y.Z --cleanup-tag --yes
  gh release create vX.Y.Z --target <full-sha> --title vX.Y.Z --notes-file notes.md
  ```
  (`--target` needs a **full** 40-character SHA or a branch name; an abbreviated
  SHA is rejected with `Release.target_commitish is invalid`.) Save the old
  release notes first — deleting the release discards them.
- **If a bad version did reach PyPI**, do not try to reuse the number. Yank it
  on PyPI and release `X.Y.Z+1`.
