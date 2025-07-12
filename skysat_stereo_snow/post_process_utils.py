# ! /usr/bin/python

import xdem
import numpy as np
import matplotlib.pyplot as plt
import json

def calculate_variogram_range(raster):
    print('Calculating the empirical variogram...')
    var = xdem.spatialstats.sample_empirical_variogram(raster)
    print('Modeling the variogram...')
    func_sum_vgm, params_vgm = xdem.spatialstats.fit_sum_model_variogram(
        list_models=["Spherical"], empirical_variogram=var
    )
    xdem.spatialstats.plot_variogram(var, list_fit_fun=[func_sum_vgm],
                                     xscale_range_split=[10, 100, 1000, 10000])
    plt.show()
    range_ = np.round(float(params_vgm['range'].values[0]))
    print('Modeled range = ', range_)
    return range_

def calculate_terrain_parameters(refelev):
    slope = refelev.slope()
    aspect = refelev.aspect()

    # Calculate variogram ranges
    res0 = refelev.res
    refelev_res = calculate_variogram_range(refelev)
    slope_res = calculate_variogram_range(slope)
    aspect_res = calculate_variogram_range(aspect)

    # Save results in dictionary
    res_dict = {
        'original_sill_m': json.dumps(res0[0]),
        'elevation_sill_m': json.dumps(refelev_res),
        'slope_sill_m': json.dumps(slope_res),
        'aspect_sill_m': json.dumps(aspect_res)
    }
    return res_dict

def resample(raster, new_res):
    """Resample raster to the new resolution."""
    return raster.reproject(res=[new_res, new_res], nodata=np.nan)
