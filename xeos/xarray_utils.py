"""Apply numpy EOS kernels to scalars / numpy / xarray / dask uniformly.

When any input is an :class:`xarray.DataArray`, kernels are wrapped with
:func:`xarray.apply_ufunc` (``dask="parallelized"``) so labels, coordinates and
dask-laziness are preserved and CF-style attributes can be attached to the
result.  Otherwise the kernel is called directly on the (broadcast) arrays.
"""

import numpy as np
import xarray as xr  # hard runtime dependency

try:  # soft dependency: keep bare dask arrays lazy when present
    import dask.array as da
except ImportError:  # pragma: no cover - dask is optional
    da = None

__all__ = ["apply_eos"]


def _is_dataarray(obj):
    return isinstance(obj, xr.DataArray)


def _coerce(arg):
    """Coerce a non-DataArray arg for direct kernel evaluation.

    Scalars pass through as scalars; bare :class:`dask.array.Array` inputs pass
    through unchanged (the kernels use only numpy arithmetic + ``np.sqrt``, which
    dask arrays support, so the result stays lazy); everything else is coerced to
    a float numpy array.
    """
    if np.isscalar(arg):
        return arg
    if da is not None and isinstance(arg, da.Array):
        return arg
    return np.asarray(arg, dtype=float)


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
    return func(*[_coerce(a) for a in args])
