#! /usr/bin/python

import math
import geopandas as gpd
import geedim as gd
import os
import subprocess
import rioxarray as rxr
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import ee
import json
from shapely.geometry import Polygon
import xdem
from rasterio.enums import Resampling

def convert_wgs_to_utm(lon: float = None, 
                       lat: float = None):
    """
    Return best UTM epsg-code based on WGS84 lat and lon coordinate pair

    Parameters
    ----------
    lon: float
        longitude coordinate
    lat: float
        latitude coordinate

    Returns
    ----------
    epsg_code: str
        optimal UTM zone, e.g. "EPSG:32606"
    """
    utm_band = str((math.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0' + utm_band
    if lat >= 0:
        epsg_code = 'EPSG:326' + utm_band
        return epsg_code
    epsg_code = 'EPSG:327' + utm_band
    return epsg_code


def create_bbox_from_meta(meta_fns: list[str] = None, 
                          buffer: float = 0, 
                          plot: bool = False) -> tuple[gpd.GeoDataFrame, str]:
    """
    Create bounding geometry from a list of metadata files. The bounding geometry is buffered by a specified distance.

    Parameters
    ----------
    meta_fns: list
        list of metadata files
    buffer: float
        buffer distance in meters
    plot: bool
        whether to plot the bounding geometry

    Returns
    ----------
    bounds_buffer_gdf: geopandas.GeoDataFrame
        bounding geometry buffered by the specified distance
    epsg_utm: str
        optimal UTM zone for the bounding geometry

    """
    # Iterate over metadata files
    xmin, xmax, ymin, ymax = 1e10, -1e10, 1e10, -1e10
    for meta_fn in meta_fns:
        meta = json.load(open(meta_fn))
        bounds = np.array(meta['geometry']['coordinates'])[0]
        xbounds, ybounds = bounds[:,0], bounds[:,1]
        xmin_im, xmax_im, ymin_im, ymax_im = np.min(xbounds), np.max(xbounds), np.min(ybounds), np.max(ybounds)
        if xmin_im < xmin:
            xmin = xmin_im
        if xmax_im > xmax:
            xmax = xmax_im
        if ymin_im < ymin:
            ymin = ymin_im
        if ymax_im > ymax:
            ymax = ymax_im

    # Create bounding geometry and buffer
    bounds_poly = Polygon([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]])
    bounds_gdf = gpd.GeoDataFrame(geometry=[bounds_poly], crs='EPSG:4326')
    epsg_utm = convert_wgs_to_utm(bounds_poly.centroid.coords.xy[0][0], bounds_poly.centroid.coords.xy[1][0])
    print(f'Optimal UTM zone = {epsg_utm}')
    bounds_utm_gdf = bounds_gdf.to_crs(epsg_utm)
    bounds_utm_buffer_gdf = bounds_utm_gdf.buffer(buffer)
    bounds_buffer_gdf = bounds_utm_buffer_gdf.to_crs('EPSG:4326')

    # Plot
    if plot:
        fig, ax = plt.subplots(1,1,figsize=(6,6))
        ax.plot(*bounds_gdf.geometry[0].exterior.coords.xy, '-k', label='Image bounds')
        ax.plot(*bounds_buffer_gdf.geometry[0].exterior.coords.xy, '-m', label='Clipping geometry')
        ax.legend(loc='upper right')
        plt.show()

    return bounds_buffer_gdf, epsg_utm


def fill_nodata(dem_fn: str = None, 
                out_fn=None) -> str:
    """
    Fill nodata values in a DEM using gdal_fillnodata.py
    
    Parameters
    ----------
    dem_fn: str
        input DEM file path
    out_fn: str
        output DEM file path (if None, will be dem_fn with '_filled' suffix)

    Returns
    ----------
    out_fn: str
        output DEM file path
    """
    if out_fn is None:
        out_fn = os.path.splitext(dem_fn)[0] + '_filled.tif'
    if not os.path.exists(out_fn):
        print('Filling gaps in DEM')
        # construct command
        cmd = ['gdal_fillnodata.py', 
                '-md', '10',
                '-of', 'GTiff',
                dem_fn, out_fn]
        # run command
        out = subprocess.run(cmd, shell=False, capture_output=True)
        print(out)
    else:
        print('Filled DEM already exists, skipping.')
    return out_fn


def query_gee_for_refdem(dem_fn: str = None, 
                         refdem_name: str = "COPDEM", 
                         out_dir: str = None, 
                         crs: str = "EPSG:4326", 
                         scale: int = 30) -> str:
    """
    Query GEE for reference DEM, clip to the DEM bounds + 1 km, and download to file.

    Parameters
    ----------
    dem_fn: str
        input DEM file path to get bounds from
    refdem_name: str
        reference DEM name, options are 'ArcticDEM', 'REMA', or 'COPDEM'
    out_dir: str
        output directory to save the clipped DEM
    crs: str
        output DEM coordinate reference system
    scale: int
        output DEM resolution in meters
    
    Returns
    ----------
    out_filled_fn: str
        output filled DEM file path
    """
    os.makedirs(out_dir, exist_ok=True)

    # Initialiize GEE
    try:
        ee.Initialize()
    except:
        ee.Authenticate()
        ee.Initialize()

    # Get DEM bounds
    dem = rxr.open_rasterio(dem_fn)
    # make sure its in WGS84
    if dem.rio.crs != "EPSG:4326":
        dem = dem.rio.reproject("EPSG:4326")
    left, bottom, right, top = dem.rio.bounds()

    # Reformat AOI for querying and clipping DEM
    region = ee.Geometry.Polygon([[left, bottom], [right, bottom], [right, top],
                                  [left, top], [left, bottom]]).buffer(1e3)

    # Query GEE for DEM
    if refdem_name=='ArcticDEM':
        dem_im = gd.MaskedImage.from_id("UMN/PGC/ArcticDEM/V4/2m_mosaic")
        band = 'elevation'
        out_fn = os.path.join(out_dir, 'ArcticDEM_Mosaic_clip.tif')
    elif refdem_name=='REMA':
        dem_im = gd.MaskedImage.from_id("UMN/PGC/REMA/V1_1/8m")
        band = 'elevation'
        out_fn = os.path.join(out_dir, 'REMA_Mosaic_clip.tif')
    elif refdem_name=='COPDEM':
        dem_col = gd.MaskedCollection.from_name("COPERNICUS/DEM/GLO30").search(start_date='1900-01-01',
                                                                                end_date='2025-01-01',
                                                                                region=region)
        # mosaic all images over the region
        dem_im = dem_col.composite(method='mosaic')
        band = 'DEM'
        out_fn = os.path.join(out_dir, 'COPDEM_Mosaic_clip.tif')

    # Download DEM 
    if not os.path.exists(out_fn):
        dem_im.download(out_fn, region=region, scale=scale, bands=[band], crs=crs)
    else:
        print(f'Clipped {refdem_name} already exists in file, skipping download.')

    # Fill holes
    out_filled_fn = fill_nodata(out_fn)
    
    return out_filled_fn


# def coregister_dems(refdem_fn, dem_fn, dem_out_fn=None):
#     if dem_out_fn is None:
#         dem_out_fn = dem_fn.replace('.tif', '_coregistered.tif')
#     if not os.path.exists(dem_out_fn):
#         # load DEMs
#         refdem = xdem.DEM(refdem_fn)
#         dem = xdem.DEM(dem_fn).reproject(refdem)

#         # run coregistration pipeline
#         coreg = xdem.coreg.CoregPipeline([xdem.coreg.ICP(), 
#                                           xdem.coreg.NuthKaab()]).fit(refdem, dem)
#         dem_nk = coreg.apply(dem)
#         dem_nk.save(dem_out_fn)
#         print('Coregistered DEM saved to file:', dem_out_fn)
#     else:
#         print('Coregistered DEM already exists, skipping.')
    
#     return dem_out_fn


# def merge_dems(coregdem_fn, second_dem_fn, out_fn=None, dstnodata=-9999, overwrite=False):
#     if out_fn is None:
#         out_fn = os.path.splitext(coregdem_fn)[0] + '__' + os.path.splitext(os.path.basename(second_dem_fn))[0] + '_merged.tif'
    
#     if (not os.path.exists(out_fn)) | overwrite:
#         # Load input files
#         coregdem = rxr.open_rasterio(coregdem_fn).squeeze()
#         crs = coregdem.rio.crs
#         second_dem = rxr.open_rasterio(second_dem_fn).squeeze()
#         second_dem = second_dem.rio.write_nodata(coregdem.rio.nodata)

#         # Match projections, using coregdem resolution to project onto second_dem
#         second_dem = second_dem.rio.reproject(resolution=coregdem.rio.resolution(), 
#                                               resampling=Resampling.bilinear,
#                                               dst_crs=crs)
#         coregdem_reproj = coregdem.rio.reproject_match(second_dem)

#         # Merge DEMs, replacing coregdem nodata values with second_dem
#         merged_dem = xr.where((coregdem_reproj==coregdem_reproj.rio.nodata) | (np.isnan(coregdem_reproj)),
#                                second_dem, coregdem_reproj)
#         merged_dem = merged_dem.rio.write_crs(crs)
#         merged_dem = merged_dem.rio.write_nodata(dstnodata)

#         # Save to file
#         merged_dem.rio.to_raster(out_fn)
#         print('Merged DEM saved to file:', out_fn)
    
#     else:
#         print('Merged DEM already exists in file, skipping.')
    
#     return out_fn
