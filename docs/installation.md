# Installation

`xeos` requires Python 3.11+. The core install pulls in only **numpy** and
**xarray**; heavier backends and acceleration are opt-in extras.

```bash
pip install xeos              # core (numpy + xarray): all vendored EOS
pip install xeos[teos10]      # adds full TEOS-10 via gsw
pip install xeos[complete]    # gsw + numba acceleration
```

## From source

```bash
git clone https://github.com/hdrake/xeos
cd xeos
pip install -e .[test]        # editable install + gsw + pytest
pytest                        # validate vendored kernels against frozen fixtures
```
