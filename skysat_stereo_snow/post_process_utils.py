# ! /usr/bin/python

import xdem
from scipy.ndimage import maximum_filter
import numpy as np
from topo_descriptors import topo, helpers
import rioxarray as rxr
import xarray as xr

# Function to calculate terain parameters
def calculate_terrain_params(dem, res_dict, raster_target_grid=None):
    """
    Calculate elevation, slope, aspect, topographic position index (TPI), and the 
    wind redistribution potential variable from Winstral (Sx) at the specified spatial resolutions. 
    
    Parameters
    ----------
    dem: xdem.DEM
        input digital elevation model (DEM)
    res_dict: dict
        dictionary containing the spatial resolutions at which to calculate each terrain parameter
    raster_target_grid: xdem.DEM
        reference raster that will be used for final gridding
        
    Returns
    ----------
    elev: xdem.DEM
        elevation raster with units the same as the input dem
    slope: xdem.DEM
        slope raster [degrees]
    aspect: xdem.DEM
        aspect raster [degrees]
    sx: xdem.DEM
        wind redistribution potential
    """
    print('Calculating terrain parameters')

    # Calculate slope and aspect
    elev = dem
    slope = xdem.terrain.slope(dem)
    aspect = xdem.terrain.aspect(dem)
    
    # determine raster for final grid
    if raster_target_grid is None:
        raster_target = dem
    else:
        raster_target = raster_target_grid

    # resample at appropriate resolution
    def resample(raster, new_res, raster_target_grid):
        raster = raster.reproject(res=[new_res, new_res], nodata=np.nan)
        raster = raster.reproject(raster_target_grid, nodata=np.nan)
        return raster
    elev = resample(elev, res_dict['elevation_sill_m'], raster_target)
    slope = resample(slope, res_dict['slope_sill_m'], raster_target)
    aspect = resample(aspect, res_dict['aspect_sill_m'], raster_target)
    
    # Calculate Topographic Position Index (TPI)
    tpi = xdem.terrain.topographic_position_index(elev)
    
    # Calculate wind redistribution potential (Sx)
    # dem_xr = xr.Dataset(data_vars={'elevation': dem.data},
    #                     coords=dem.coords)
    def calculate_sx(slope, search_radius=200):
        # Convert search radius to number of pixels
        res = slope.coords()[0][0][1] -slope.coords()[0][0][0]
        search_radius_px = int(search_radius / res)
        # Compute maximum slope within the search radius for each cell
        sx = maximum_filter(slope.data, size=search_radius_px)
        sx_xdem = xdem.DEM.from_array(sx, transform=slope.transform, crs=slope.crs, nodata=np.nan)
        return sx_xdem
    sx = calculate_sx(slope)
    return elev, slope, aspect, tpi, sx
    