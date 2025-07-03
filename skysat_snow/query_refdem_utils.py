#! /usr/bin/python

import math
import geopandas as gpd
import geedim as gd
import os
import pyproj
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

def convert_wgs_to_utm(lon: float, lat: float):
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


def create_bbox_from_meta(meta_fns, buffer, plot=True):
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


def fill_nodata(dem_fn, out_fn=None):
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


def query_gee_for_copdem(aoi, out_fn=None, crs='EPSG:4326', scale=30):
    """
    Query GEE for the COPDEM, clip to the AOI, and return as xarray.Dataset.

    Parameters
    ----------
    aoi: geopandas.geodataframe.GeoDataFrame
        area of interest used for clipping the DEM 
    out_fn: str
        file name for output DEM
    crs: str
        Coordinate Reference System of output DEM

    Returns
    ----------
    dem_ds: xarray.Dataset
        dataset of elevations over the AOI
    """

    # Reproject AOI to EPSG:4326 if necessary
    aoi_wgs = aoi.to_crs('EPSG:4326')

    # Reformat AOI for querying and clipping DEM
    region = ee.Geometry.Polygon(list(zip(aoi_wgs.geometry[0].exterior.coords.xy[0],
                                          aoi_wgs.geometry[0].exterior.coords.xy[1])))
    {'type': 'Polygon',
              'coordinates': [[
                  [aoi_wgs.geometry.bounds.minx[0], aoi_wgs.geometry.bounds.miny[0]],
                  [aoi_wgs.geometry.bounds.maxx[0], aoi_wgs.geometry.bounds.miny[0]],
                  [aoi_wgs.geometry.bounds.maxx[0], aoi_wgs.geometry.bounds.maxy[0]],
                  [aoi_wgs.geometry.bounds.minx[0], aoi_wgs.geometry.bounds.maxy[0]],
                  [aoi_wgs.geometry.bounds.minx[0], aoi_wgs.geometry.bounds.miny[0]]
              ]]
              }

    # Query GEE for DEM
    dem_col = gd.MaskedCollection.from_name("COPERNICUS/DEM/GLO30").search(start_date='1900-01-01',
                                                                           end_date='2025-01-01',
                                                                           region=region)
    # Mosaic all images over the region
    dem_im = dem_col.composite(method='mosaic')

    # Download DEM 
    if not os.path.exists(out_fn):
        dem_im.download(out_fn, region=region, scale=scale, bands=['DEM'], crs=crs)

    # Reproject from the EGM96 geoid to the WGS84 ellipsoid
    s_crs = pyproj.CRS.from_epsg(int(crs.split(':')[1]))
    s_proj_string = s_crs.to_proj4() + " +vunits=m +nodefs"
    t_proj_string = s_proj_string 
    s_proj_string += f' +geoidgrids=egm96_15.gtx'
    out_ellip_fn = out_fn.replace('.tif', '_WGS84_ellipsoid.tif')
    if not os.path.exists(out_ellip_fn):
        cmd = f'''gdalwarp -s_srs "{s_proj_string}" -t_srs "{t_proj_string}" {out_fn} {out_ellip_fn}'''
        output = subprocess.run(cmd, capture_output=True, shell=True)
        print(output)
        print('DEM reprojected to the WGS84 ellipsoid and saved to file:', out_ellip_fn)
    else:
        print('DEM reprojected to the WGS84 ellipsoid already exists in file, skipping.')
        
    # Simplify CRS to UTM Zone without ellipsoidal height
    # get UTM zone from EPSG code
    if '326' in crs:
        utm_zone = crs.split('EPSG:326')[1]
    out_ellip_utm_fn = out_ellip_fn.replace('.tif', f'_UTM{utm_zone}.tif')
    if not os.path.exists(out_ellip_utm_fn):
        cmd = f'''gdalwarp -s_srs "{t_proj_string}" -t_srs "+proj=utm +zone={utm_zone} +datum=WGS84" {out_ellip_fn} {out_ellip_utm_fn}'''
        output = subprocess.run(cmd, capture_output=True, shell=True)
        print(output)
        print(f'DEM reprojected to UTM Zone {utm_zone} and saved to file:', out_ellip_utm_fn)
    else:
        print(f'DEM reprojected to UTM Zone {utm_zone} already exists in file, skipping.')

    # Fill holes
    out_ellip_utm_filled_fn = fill_nodata(out_ellip_utm_fn)

    # Open DEM as xarray.DataArray and plot
    dem = rxr.open_rasterio(out_ellip_utm_filled_fn).squeeze()
    fig, ax = plt.subplots()
    dem_im = ax.imshow(dem.data, cmap='terrain',
              extent=(np.min(dem.x.data), np.max(dem.x.data), 
                      np.min(dem.y.data), np.max(dem.y.data)))
    fig.colorbar(dem_im, ax=ax, label='Elevation [m]')
    ax.set_title(os.path.basename(out_ellip_utm_filled_fn))
    plt.show()
    
    return out_ellip_utm_filled_fn

def query_gee_for_arcticdem(aoi, out_fn=None, crs='EPSG:4326', scale=30):
    """
    Query GEE for the ArcticDEM, clip to the AOI, and return as xarray.Dataset.

    Parameters
    ----------
    aoi: geopandas.geodataframe.GeoDataFrame
        area of interest used for clipping the DEM 
    out_fn: str
        file name for output DEM
    crs: str
        Coordinate Reference System of output DEM

    Returns
    ----------
    dem_ds: xarray.Dataset
        dataset of elevations over the AOI
    """

    # Reproject AOI to EPSG:4326 if necessary
    aoi_wgs = aoi.to_crs('EPSG:4326')

    # Reformat AOI for querying and clipping DEM
    region = ee.Geometry.Polygon(list(zip(aoi.geometry[0].exterior.coords.xy[0], aoi.geometry[0].exterior.coords.xy[1])))

    # Query GEE for DEM
    dem_im = gd.MaskedImage.from_id("UMN/PGC/ArcticDEM/V3/2m_mosaic")

    # Download DEM 
    if not os.path.exists(out_fn):
        dem_im.download(out_fn, region=region, scale=scale, bands=['elevation'], crs=crs)

    # Fill holes
    out_filled_fn = out_fn.replace('.tif', '_filled.tif')
    _ = fill_nodata(out_fn, out_fn=out_filled_fn)

    # Open DEM as xarray.DataArray and plot
    dem = rxr.open_rasterio(out_filled_fn).squeeze()
    fig, ax = plt.subplots()
    dem_im = ax.imshow(dem.data, cmap='terrain',
              extent=(np.min(dem.x.data), np.max(dem.x.data), 
                      np.min(dem.y.data), np.max(dem.y.data)))
    fig.colorbar(dem_im, ax=ax, label='Elevation [m]')
    ax.set_title(os.path.basename(out_filled_fn))
    plt.show()
    
    return dem

def coregister_dems(refdem_fn, dem_fn, dem_out_fn=None):
    if dem_out_fn is None:
        dem_out_fn = dem_fn.replace('.tif', '_coreg.tif')
    if not os.path.exists(dem_out_fn):
        print('Coregistering using the Nuth and Kaab approach')
        refdem = xdem.DEM(refdem_fn) # high-res
        dem = xdem.DEM(dem_fn) # lower-res
        refdem_reproj = refdem.reproject(dem)
        nk = xdem.coreg.NuthKaab().fit(refdem_reproj, dem)
        print(nk.meta)
        dem_nk = nk.apply(dem)
        dem_nk.save(dem_out_fn)
        print('Coregistered DEM saved to file:', dem_out_fn)
    else:
        print('Coregistered DEM already exists, skipping.')
    
    return dem_out_fn


def merge_dems(coregdem_fn, second_dem_fn, out_fn=None, dstnodata=-9999, overwrite=False):
    if out_fn is None:
        out_fn = os.path.splitext(coregdem_fn)[0] + '__' + os.path.splitext(os.path.basename(second_dem_fn))[0] + '_merged.tif'
    
    if (not os.path.exists(out_fn)) | overwrite:
        # Load input files
        coregdem = rxr.open_rasterio(coregdem_fn).squeeze()
        crs = coregdem.rio.crs
        second_dem = rxr.open_rasterio(second_dem_fn).squeeze()
        second_dem = second_dem.rio.write_nodata(coregdem.rio.nodata)

        # Match projections, using coregdem resolution to project onto second_dem
        second_dem = second_dem.rio.reproject(resolution=coregdem.rio.resolution(), 
                                              resampling=Resampling.bilinear,
                                              dst_crs=crs)
        coregdem_reproj = coregdem.rio.reproject_match(second_dem)

        # Merge DEMs, replacing coregdem nodata values with second_dem
        merged_dem = xr.where((coregdem_reproj==coregdem_reproj.rio.nodata) | (np.isnan(coregdem_reproj)),
                               second_dem, coregdem_reproj)
        merged_dem = merged_dem.rio.write_crs(crs)
        merged_dem = merged_dem.rio.write_nodata(dstnodata)

        # Save to file
        merged_dem.rio.to_raster(out_fn)
        print('Merged DEM saved to file:', out_fn)
    
    else:
        print('Merged DEM already exists in file, skipping.')
    
    return out_fn
