# Conda packaging for xeos

`meta.yaml` in this directory is a [conda-forge](https://conda-forge.org/)
recipe for `xeos`. It is **noarch: python** (pure Python, no compiled
extensions, no entry points) and builds from the sdist published on PyPI. It
uses the `python_min` pin conda-forge recommends for `noarch: python` recipes,
so bumping the minimum Python version means editing a single `{% set %}` line.

This copy is a reference/working copy. The recipe actually under review lives on
the [`hdrake/staged-recipes@add-xeos`](https://github.com/hdrake/staged-recipes/tree/add-xeos)
fork branch; keep the two in sync when you change one.

## Runtime dependencies

Only the lightweight core is required:

- `python >=3.11`
- `numpy`
- `xarray`

`gsw` (full TEOS-10) and `numba` (acceleration) are **optional** — they are the
PyPI `xeos[complete]` extra and are deliberately *not* listed as `run`
dependencies, so the conda package stays lightweight. Users who need them
install `gsw` / `numba` themselves.

## Current status

`xeos` is on **PyPI** (`pip install xeos`) but not yet on **conda-forge**. The
first-time recipe has been submitted and is awaiting review/merge:

- staged-recipes PR: <https://github.com/conda-forge/staged-recipes/pull/34003>
- Recipe is at `version: 0.1.0` with the real sdist sha256 filled in and the
  conda-forge linter passing. It is waiting on a conda-forge reviewer to merge.
  If it stalls, a polite ping (or `@conda-forge/help-python`) on the PR is the
  usual nudge.

Once that PR merges, conda-forge bootstraps a **`conda-forge/xeos-feedstock`**
repo and publishes to the channel, after which `conda install -c conda-forge
xeos` works.

## Releasing to conda-forge (staged-recipes flow)

conda-forge packages are not built from this repo directly; you submit the
recipe once to
[`conda-forge/staged-recipes`](https://github.com/conda-forge/staged-recipes),
after which a dedicated `xeos-feedstock` repo is created and all future updates
happen there (mostly automatically). This is the flow that produced PR #34003
above; it is kept here for reference and for any future re-submission.

First-time submission:

1. **Publish to PyPI first.** conda-forge builds from the PyPI sdist, so the
   `xeos-<version>.tar.gz` must already be on PyPI (the GitHub
   "Publish to PyPI" workflow does this on a GitHub Release).
2. **Get the sdist sha256** and put it in the `sha256:` field of `meta.yaml`
   (replacing the value for the previous release). Either copy it from the
   sdist's "view hashes" link on <https://pypi.org/project/xeos/#files>, or run
   (substituting the version being released):
   ```bash
   curl -sL https://pypi.org/pypi/xeos/0.1.0/json \
     | python -c "import json,sys; print([u['digests']['sha256'] for u in json.load(sys.stdin)['urls'] if u['packagetype']=='sdist'][0])"
   ```
3. **Fork `conda-forge/staged-recipes`**, create a branch, and copy this
   recipe to `recipes/xeos/meta.yaml` in that fork (conda-forge expects the
   recipe under `recipes/<name>/`). Make sure `{% set version %}` matches the
   PyPI release and the sha256 is up to date.
4. **Open a PR** against `conda-forge/staged-recipes`. CI lints the recipe and
   does a test build on Linux/macOS/Windows; fix anything it flags. A
   conda-forge maintainer reviews and merges.
5. After merge, conda-forge bootstraps a **`conda-forge/xeos-feedstock`** repo
   and publishes `xeos` to the `conda-forge` channel:
   ```bash
   conda install -c conda-forge xeos
   ```
   Add yourself (and any co-maintainers) under `extra.recipe-maintainers`.

## Updating after the feedstock exists

Once `xeos-feedstock` exists you normally do **not** touch staged-recipes again.
For each new release the conda-forge autotick bot opens a PR bumping `version`
and `sha256`; just review and merge it. For dependency/recipe changes, edit
`recipe/meta.yaml` in the feedstock (bump `build.number`) and open a PR there.

## Local lint / test (optional)

```bash
conda install -n base conda-smithy conda-build
conda smithy recipe-lint conda/        # lint this recipe
conda build conda/                     # full local build (needs the PyPI sdist + real sha256)
```
