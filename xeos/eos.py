import seawater as sw
import fastjmd95
import momlevel
import gsw
import xarray as xr
import warnings

eos_list = ["TEOS-10", "EOS-80", "JMD-95", "Wright-97"]
expected_p_units = {
    "TEOS-10": ("dbar", "db", "decibar", "decibars"),
    "EOS-80": ("dbar", "db", "decibar", "decibars"),
    "JMD-95": ("dbar", "db", "decibar", "decibars"),
    "Wright-97": ("pa", "pascal", "pascals"),
}

def _check_p_units(p, eos):
    """Simple check for pressure units in an xarray DataArray."""

    # Explicitly check against the dictionary keys
    if eos not in expected_p_units.keys():
        return

    target_units = expected_p_units[eos]
    
    if hasattr(p, "attrs"):
        units = p.attrs.get("units")
        valid_units_str = ", ".join(target_units)

        if units is None:
            warnings.warn(f"No 'units' attribute found in `p`. {eos} expects one of: {valid_units_str}.")
            
        elif units.lower() not in target_units:
            warnings.warn(f"Unit mismatch: `p` has '{units}', but {eos} expects one of: {valid_units_str}.")

def _raise_eos_error():
    raise ValueError(f"This eos is not currently supported; choose from f{eos_list}.")

def rho(t, s, p, eos):
    _check_p_units(p, eos)

    if eos=="TEOS-10":
        return xr.apply_ufunc(
                gsw.rho, s, t, p, dask="parallelized"
            )
    elif eos=="EOS-80":
        return xr.apply_ufunc(
                sw.dens, s, t, p, dask="parallelized"
        )
    elif eos=="JMD-95":
        return fastjmd95.rho(s, t, p)
    elif eos=="Wright-97":
        return momlevel.derived.calc_rho(t, s, p, eos="Wright")
    else:
        _raise_eos_error()

def alpha(t, s, p, eos):
    _check_p_units(p, eos)

    if eos=="TEOS-10":
        return xr.apply_ufunc(
                gsw.alpha, s, t, p, dask="parallelized"
            )
    elif eos=="EOS-80":
        return xr.apply_ufunc(
                sw.alpha, s, t, p, dask="parallelized"
        )
    elif eos=="JMD-95":
        return -fastjmd95.drhodt(s, t, p)/rho(t, s, p, eos="JMD-95")
    elif eos=="Wright-97":
        return momlevel.derived.calc_alpha(t, s, p, eos="Wright")
    else:
        _raise_eos_error()

def beta(t, s, p, eos):
    _check_p_units(p, eos)

    if eos=="TEOS-10":
        return xr.apply_ufunc(
                gsw.beta, s, t, p, dask="parallelized"
            )
    elif eos=="EOS-80":
        return xr.apply_ufunc(
                sw.beta, s, t, p, dask="parallelized"
        )
    elif eos=="JMD-95":
        return fastjmd95.drhods(s, t, p)/rho(t, s, p, eos="JMD-95")
    elif eos=="Wright-97":
        return momlevel.derived.calc_beta(t, s, p, eos="Wright")
    else:
        _raise_eos_error()
