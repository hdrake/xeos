"""Apply numpy EOS kernels to scalars / numpy / xarray / dask uniformly.

When any input is an :class:`xarray.DataArray`, kernels are wrapped with
:func:`xarray.apply_ufunc` (``dask="parallelized"``) so labels, coordinates and
dask-laziness are preserved and CF-style attributes can be attached to the
result.  Otherwise the kernel is called directly on the (broadcast) arrays.
"""

import numpy as np
import xarray as xr  # hard runtime dependency

__all__ = ["apply_eos"]


def _is_dataarray(obj):
    return isinstance(obj, xr.DataArray)


def apply_eos(func, *args, attrs=None):
    """Apply ``func(*args)`` preserving xarray labels/dask when present.

    Parameters
    ----------
    func : callable
        Elementwise kernel taking the broadcast ``args``.
    *args
        Array-likes (scalars, numpy arrays, dask arrays, or DataArrays).
    attrs : dict, optional
        CF-style attributes attached to the resulting DataArray.
    """
    if any(_is_dataarray(a) for a in args):
        result = xr.apply_ufunc(
            func, *args, dask="parallelized", output_dtypes=[np.float64],
        )
        if attrs:
            result = result.assign_attrs(attrs)
        return result
    return func(*[a if np.isscalar(a) else np.asarray(a, dtype=float) for a in args])
