import xeos

import warnings
import numpy as np
import pytest
import xarray as xr
ds = xr.Dataset({
    "T": xr.DataArray(np.array([10.])),
    "S": xr.DataArray(np.array([35.5])),
    "p": xr.DataArray(np.array([0.]))
})

def test_densities():
    for eos in xeos.eos_list:
        assert np.all(1027 < xeos.rho(ds.T, ds.S, ds.p, eos=eos) < 1028)

def test_alpha():
    for eos in xeos.eos.eos_list:
        assert np.all( 1.e-5 < xeos.alpha(ds.T, ds.S, ds.p, eos=eos) < 1.e-3)

def test_beta():
    for eos in xeos.eos.eos_list:
        assert np.all( 1.e-5 < xeos.beta(ds.T, ds.S, ds.p, eos=eos) < 1.e-3)

def test_check_p_units():
    p_no_units = xr.DataArray(np.array([0.0]))

    for eos in xeos.eos.eos_list:
        with pytest.warns(UserWarning):
            xeos.eos._check_p_units(p_no_units, eos)

        p_with_units = xr.DataArray(
            np.array([0.0]), attrs={"units": xeos.eos.expected_p_units[eos][0]}
        )
        with warnings.catch_warnings(record=True) as no_warning:
            warnings.simplefilter("always")
            xeos.eos._check_p_units(p_with_units, eos)
        assert len(no_warning) == 0
