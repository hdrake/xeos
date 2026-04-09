import xeos

import numpy as np
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