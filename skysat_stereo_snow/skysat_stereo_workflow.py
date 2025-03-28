#! /usr/bin/env python

import os,sys,glob,re,shutil
import numpy as np
import geopandas as gpd
import pandas as pd
from pygeotools.lib import iolib,malib
from tqdm import tqdm
from p_tqdm import p_umap, p_map
from skysat_stereo import skysat
from skysat_stereo import asp_utils as asp
from rpcm import geo
from skysat_stereo import misc_geospatial as misc
import subprocess
from shapely.geometry import Polygon
import itertools
from pyproj import Transformer
import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt
import matplotlib
import geoutils as gu
import xdem
import multiprocessing

def run_cmd(bin, args, **kw):
    """
    Wrapper for subprocess function to execute bash commands.

    Parameters
    ----------
    bin: str
        command to be excuted (e.g., stereo or gdalwarp)
    args: list
        arguments to the command as a list
    
    Returns
    ----------
    out: str
        log (stdout) as str if the command executed, error message if the command failed
    """
    #Note, need to add full executable
    # binpath = '/Users/raineyaberle/Research/PhD/SnowDEMs/StereoPipeline-3.5.0-alpha-2024-10-05-x86_64-OSX/bin/' + bin
    binpath = shutil.which(bin)
    # print(binpath)
    if binpath is None:
        msg = ("Unable to find executable %s\n"
        "Install ASP and ensure it is in your PATH env variable\n"
       "https://ti.arc.nasa.gov/tech/asr/intelligent-robotics/ngt/stereo/" % bin)
        sys.exit(msg)
    call = [binpath,]
    if args is not None: 
        call.extend(args)
    try:
        out = subprocess.run(call,check=True,capture_output=True,encoding='UTF-8').stdout
    except:
        out = "the command {} failed to run, see corresponding asp log".format(call)
    return out


def convert_wgs_to_utm(lon: float, lat: float):
    """
    Return optimal UTM EPSG code based on WGS84 lat and lon coordinate pair.

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
    utm_band = str((np.floor((lon + 180) / 6) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0' + utm_band
    if lat >= 0:
        epsg_code = 'EPSG:326' + utm_band
        return epsg_code
    epsg_code = 'EPSG:327' + utm_band
    return epsg_code


def calculate_image_bounds(img_folder, out_folder):
    """
    Save a bounding box geopackage for all images in a folder.

    Parameters
    ----------
    img_folder: str or Path
        folder containing geoTIFF files
    out_folder: str or Path
        folder where bounding box will be saved
    
    Returns
    ----------
    bound_fn: str
        file name of the output bounding box
    bound_buffer_fn: str
        file name of the output bounding box + 1 km buffer
    utm_epsg: str
        EPSG code for the optimal UTM zone, e.g. "EPSG:32601"
    """
    # Get list of images
    try:
        img_list = sorted(glob.glob(os.path.join(img_folder,'*.tif')) + glob.glob(os.path.join(img_folder,'*.tiff')))
        print("Number of images {}".format(len(img_list)))
    except:
        raise Exception("No images found in the directory. Check path and make sure they end with a '.tif' or '.tiff' extension")

    # Define output files
    bound_fn = os.path.join(out_folder, 'image_bounds.gpkg')
    bound_buffer_fn = os.path.join(out_folder, 'image_bounds_buffer1km.gpkg')

    # Get list of image bounds
    n_proc = multiprocessing.cpu_count()
    shp_list = p_map(skysat.skysat_footprint,img_list,num_cpus=n_proc)    
    
    # Merge image bounds
    merged_shape = misc.shp_merger(shp_list)
    bbox = merged_shape.total_bounds
    print(f'Bounding box lon_lat is:{bbox}')

    # Solve for optimal UTM zone
    utm_epsg = convert_wgs_to_utm(bbox[0], bbox[1])
    
    # Convert to GeoDataFrame
    bound_poly = Polygon([[bbox[0],bbox[3]], [bbox[2],bbox[3]], [bbox[2],bbox[1]], [bbox[0],bbox[1]]])
    bound_shp = gpd.GeoDataFrame(index=[0], geometry=[bound_poly], crs=utm_epsg)
    # buffer by 1km
    bound_buffer_shp = bound_shp.buffer(1e3)
    
    # Save to file
    bound_shp.to_file(bound_fn,driver='GPKG')
    print('Image bounds saved to file:', bound_fn)
    bound_buffer_shp.to_file(bound_buffer_fn, driver='GPKG')
    print('Image bounds + 1km buffer saved to file:', bound_buffer_fn)

    return bound_fn, bound_buffer_fn, utm_epsg


def identify_overlapping_image_pairs(img_folder, tcrs, out_folder, overlap_perc=0.1, cross_track=False):
    """
    Identify all overlapping pairs from a list of images. 

    Parameters
    ----------
    img_folder: str or Path
        folder containing geoTIFF files
    tcrs: str
        target CRS for calculating the overlap percentages 
    out_folder: str or Path
        folder where output data tables will be saved
    overlap_perc: float (default = 0.1)
        Minimum overlap percentage 
    cross_track: bool (default=False)
        whether to include cross-track pairs. Otherwise only true stereo pairs (different cameras) included
    
    Returns
    ---------
    out_stereo_pkl: str
        file name of the output stereo pair data pickle with overlap percentages 
    out_stereo_txt: str
        file name of the output stereo pair list. May be used as inputs in ASP. 
    """
    # Make output directory if it does not exist
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    # Define outputs
    out_txt = os.path.join(out_folder, 'overlap.txt')
    out_overlap_pkl = os.path.splitext(out_txt)[0]+'_with_overlap_perc.pkl'
    out_stereo_pkl = os.path.splitext(out_overlap_pkl)[0]+'_stereo_only.pkl'
    out_stereo_txt = os.path.splitext(out_overlap_pkl)[0]+'_stereo_only.txt'

    # Get list of images
    try:
        img_list = sorted(glob.glob(os.path.join(img_folder,'*.tif')) + glob.glob(os.path.join(img_folder,'*.tiff')))
        print("Number of images {}".format(len(img_list)))
    except:
        raise Exception("No images found in the directory. Check path and make sure they end with a '.tif' or '.tiff' extension")


    # Identify possible combinations
    img_combinations = list(itertools.combinations(img_list,2))
    n_comb = len(img_combinations)
    perc_overlap = np.ones(n_comb,dtype=float)*overlap_perc
    ncpu = multiprocessing.cpu_count()
    tv = p_map(skysat.frame_intsec, img_combinations, [tcrs]*n_comb, perc_overlap, num_cpus=ncpu)
    # result to this contains truth value (0 or 1, overlap percentage)
    truth_value = [tvs[0] for tvs in tv]
    overlap = [tvs[1] for tvs in tv]
    valid_list = list(itertools.compress(img_combinations,truth_value))
    overlap_perc_list = list(itertools.compress(overlap,truth_value))
    print(f"Number of valid combinations are {len(valid_list)}, out of total {len(img_list)} input images making total combinations {n_comb}")
    
    # Save outputs
    # overlap txt
    with open(out_txt, 'w') as f:
        img1_list = [x[0] for x in valid_list]
        img2_list = [x[1] for x in valid_list]
        for idx,i in enumerate(valid_list):
            f.write(f"{os.path.abspath(img1_list[idx])} {os.path.abspath(img2_list[idx])}\n")
    # overlap pkl
    img1_list = [x[0] for x in valid_list]
    img2_list = [x[1] for x in valid_list]
    out_df = pd.DataFrame({'img1':img1_list,'img2':img2_list,'overlap_perc':overlap_perc_list})
    out_df.to_pickle(out_overlap_pkl)
    # stereo-only pickle
    stereo_only_df = skysat.prep_trip_df(out_overlap_pkl, cross_track=cross_track)
    stereo_only_df.to_pickle(out_stereo_pkl)
    # stereo-only txt (for bundle_adjust, etc.) 
    stereo_only_df[['img1','img2']].to_csv(out_stereo_txt, sep=' ', header=False, index=False)
    
    return out_stereo_pkl, out_stereo_txt


def skysat_preprocess(img_folder, frame_index=None, product_level='l1a', overlap_pkl=None,
                      dem_fn=None, out_folder=None):
    """
    Resection and construct pinhole cameras for all images in the input folder. 

    Parameters
    ----------
    img_folder: str or Path
        folder containing geoTIFF files
    product_level: str
        product level of the input images: "l1a" or "l1b"
    overlap_pkl: str or Path
        file name of the overlapping stereo pairs data table (see "identify_overlapping_image_pairs")
    dem_fn: str or Path
        file name of the reference DEM
    out_folder: str or Path
        folder where output cameras will be saved
    
    Returns
    ----------
    cam_gen_log: log
        log of the cam_gen command
    """
    # make output directory if it doesn't exist
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    
    # Read overlap data table
    df = pd.read_pickle(overlap_pkl)
    img_list = list(np.unique(np.array(list(df.img1.values)+list(df.img2.values))))
    img_list = [os.path.splitext(os.path.basename(img))[0] for img in img_list]
    cam_list = [glob.glob(os.path.join(img_folder,'{}*.tif'.format(img)))[0] for img in img_list]
    n = len(img_list)

    # Set up camera specs
    if product_level == 'l1b':
        pitch = [0.8]*n
    else:
        pitch = [1.0]*n
    out_fn = [os.path.join(out_folder,'{}_rpc.tsai'.format(frame)) for frame in img_list]
    out_gcp = [os.path.join(out_folder,'{}_rpc.gcp'.format(frame)) for frame in img_list]
    camera = cam_list
    frame_index = [None]*n
    img_list = cam_list
    gcp_factor = 8
    fl = [553846.153846]*n
    cx = [1280]*n
    cy = [540]*n
    gcp_std = [1]*n
    datum = ['WGS84']*n
    refdem = [dem_fn]*n
    ht_datum = [np.nanmedian(iolib.fn_getma(dem_fn).data)]*n # use this value for height where DEM has no-data

    # Determine number of jobs and threads per job
    ncpu = multiprocessing.cpu_count()
    njobs = 4
    threads_per_job = int(np.floor(ncpu/njobs))
    threads_list = [str(threads_per_job)]*n
    print(f"Running across {njobs} CPU with {threads_per_job} threads per CPU")
    
    # Run cam_gen
    print("Starting camera resection procedure")
    cam_gen_log = p_map(asp.cam_gen, img_list, fl, cx, cy, pitch, ht_datum, gcp_std, out_fn, out_gcp,
                        datum, refdem, camera, frame_index, threads_list, num_cpus=njobs)
    print(cam_gen_log[0])
    print("writing gcp with basename removed")
    # count expexted gcp 
    print(f"Total expected GCP {gcp_factor*n}")    
    asp.clean_gcp(out_gcp, out_folder)
    
    return cam_gen_log   


def construct_land_cover_masks(multispec_path, out_folder, ndvi_threshold=0.5, ndsi_threshold=0.0, plot_results=True):
    """
    Construct masks for trees, snow, and stable surfaces from a 4-band image. 

    Parameters
    ----------
    multispec_path: str or Path
        path to the multispectral mosaic or folder containing image to mosaic
    out_folder: str or Path
        folder where output land cover masks will be saved
    ndvi_threshold: float (default=0.5)
        Normalized Difference Vegetation Index (NDVI) threshold used to identify trees
    ndsi_threshold: float (default=0.0)
        Modified Normalized Difference Snow Index (NDSI) threshold used to identify snow. 
        A threshold of 0.3 is also applied to the red band to identify snow. 
    plot_results: bool (default=True)
        whether to plot the land cover masks and save a figure to out_folder
    
    Returns 
    ----------
    trees_mask_fn: str
        file name of the output trees mask
    snow_mask_fn: str
        file name of the output snow mask
    ss_mask_fn: str
        file name of the output stable surfaces mask
    """
    # Make output directory if it doesn't exist
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    
    # Define output files
    trees_mask_fn = os.path.join(out_folder, 'trees_mask.tif')
    snow_mask_fn = os.path.join(out_folder, 'snow_mask.tif')
    ss_mask_fn = os.path.join(out_folder, 'stable_surfaces_mask.tif')
    fig_fn = os.path.join(out_folder, 'land_cover_masks.png')
    
    # Check if multispec_dir is a directory of images or a single file
    if os.path.isdir(multispec_path):
        multispec_mosaic_fn = os.path.join(out_folder, '4band_mosaic.tif')
        # Check if mosaic exists
        if not os.path.exists(multispec_mosaic_fn):
            print('Mosacking 4-band SR images...')
            # Grab all 4-band SR image file names
            multispec_fns = sorted(glob.glob(os.path.join(multispec_path, '*_SR.tif')))
            # Construct gdal_merge arguments
            merge_args = multispec_fns + ['--ot', 'Int16', '--no-bigtiff', '-o', multispec_mosaic_fn]
            # Run command
            run_cmd('image_mosaic', merge_args)
        else:
            print('4-band mosaic already exists, skipping gdal_merge.')
    elif os.path.isfile(multispec_path):
        multispec_mosaic_fn = multispec_path

    # Function to load 4-band mosaic if needed
    mosaic = None
    def load_mosaic(multispec_mosaic_fn):
        mosaic_xr = rxr.open_rasterio(multispec_mosaic_fn).astype(float)
        crs = f'EPSG:{mosaic_xr.rio.crs.to_epsg()}'
        mosaic = xr.Dataset(coords={'y': mosaic_xr.y, 'x':mosaic_xr.x})
        bands = ['blue', 'green', 'red', 'NIR']
        for i, b in enumerate(bands):
            mosaic[b] = mosaic_xr.isel(band=i)
        # account for image scalar and no data values
        mosaic = xr.where(mosaic==0, np.nan, mosaic / 1e4)
        mosaic.rio.write_crs(crs, inplace=True)
        return mosaic, crs

    # Construct trees mask
    if not os.path.exists(trees_mask_fn):
        print('Constructing trees mask...')
        if not mosaic:
            mosaic, crs = load_mosaic(multispec_mosaic_fn)
        # Calculate NDVI
        ndvi = (mosaic.NIR - mosaic.red) / (mosaic.NIR + mosaic.red)
        # Apply threshold
        trees_mask = xr.where(ndvi >= ndvi_threshold, 1, 0).astype(int)
        # Save to file
        trees_mask = xr.where(np.isnan(mosaic.blue), -9999, trees_mask) # set no data values to -9999
        trees_mask = trees_mask.assign_attrs({'Description': 'Trees mask constructing by thresholding the NDVI of the 4-band mosaic image.',
                                              '_FillValue': -9999,
                                              'NDVI bands': 'NIR, red',
                                              'NDVI threshold': ndvi_threshold})
        trees_mask.rio.write_crs(crs, inplace=True)
        trees_mask.rio.to_raster(trees_mask_fn, dtype='int16')
        print('Trees mask saved to file:', trees_mask_fn)
    else:
        print('Trees mask exists in directory, skipping.')

    # Construct snow mask
    if not os.path.exists(snow_mask_fn):
        print('Constructing snow mask...')
        if not mosaic:
            mosaic, crs = load_mosaic(multispec_mosaic_fn)
        # Calculate NDSI
        ndsi = (mosaic.red - mosaic.NIR) / (mosaic.red + mosaic.NIR)
        # Apply thresholds
        snow_mask = xr.where((ndsi >= ndsi_threshold) & (mosaic.red > 0.3), 1, 0).astype(int)
        # Save to file
        snow_mask = xr.where(np.isnan(mosaic.blue), -9999, snow_mask) # set no data values to -9999
        snow_mask = snow_mask.assign_attrs({'Description': 'Snow mask constructed by thresholding the NDSI of the orthomosaic image',
                                            '_FillValue': -9999,
                                            'NDSI bands': 'red, NIR',
                                            'NDSI threshold': ndsi_threshold})
        snow_mask.rio.write_crs(crs, inplace=True)
        snow_mask.rio.to_raster(snow_mask_fn, dtype='int16')
        print('Snow mask saved to file:', snow_mask_fn)
    else:
        print('Snow mask exists in directory, skipping.')

    # Construct stable surfaces (snow-free and tree-free) mask
    if not os.path.exists(ss_mask_fn):
        print('Constructing stable surfaces mask...')
        if not mosaic:
            mosaic, crs = load_mosaic(multispec_mosaic_fn)
        # Load trees and snow masks
        trees_mask = rxr.open_rasterio(trees_mask_fn).squeeze()
        snow_mask = rxr.open_rasterio(snow_mask_fn).squeeze()
        # Stable surfaces = snow-free and tree-free
        ss_mask = xr.where((trees_mask==0) & (snow_mask==0), 1, 0)
        # Save to file
        ss_mask = xr.where(np.isnan(mosaic.blue), -9999, ss_mask)
        ss_mask = ss_mask.assign_attrs({'Description': 'Stable surfaces (snow-free and tree-free) mask.',
                                        '_FillValue': -9999})
        ss_mask.rio.write_crs(crs, inplace=True)
        ss_mask.rio.to_raster(ss_mask_fn, dtype='int16')
        print('Stable surfaces mask saved to file:', ss_mask_fn)
    else: 
        print('Stable surfaces mask exists in directory, skipping.')

    # Plot land cover masks
    if plot_results & (not os.path.exists(fig_fn)):
        print('Plotting land cover masks...')
        if not mosaic:
            mosaic, crs = load_mosaic(multispec_mosaic_fn)
        # Load masks
        trees_mask = rxr.open_rasterio(trees_mask_fn).squeeze()
        snow_mask = rxr.open_rasterio(snow_mask_fn).squeeze()
        ss_mask = rxr.open_rasterio(ss_mask_fn).squeeze()
        # Define land cover colors
        colors_dict = {'trees': '#167700', 
                       'snow': '#55F5FF', 
                       'stable_surfaces': '#C3C3C3'}
        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(10,5))
        # RGB image
        ax[0].imshow(np.dstack([mosaic.red.data, mosaic.green.data, mosaic.blue.data]),
                     extent=(np.min(mosaic.x)/1e3, np.max(mosaic.x)/1e3, np.min(mosaic.y)/1e3, np.max(mosaic.y)/1e3))
        ax[0].set_title('RGB mosaic')
        ax[0].set_xlabel('Easting [km]')
        ax[0].set_ylabel('Northing [km]')
        # Land cover masks
        
        for mask, label in zip([trees_mask, snow_mask, ss_mask], list(colors_dict.keys())):
            cmap = matplotlib.colors.ListedColormap([(0,0,0,0), matplotlib.colors.to_rgb(colors_dict[label])])
            ax[1].imshow(mask, cmap=cmap, clim=(0,1),
                        extent=(np.min(mask.x)/1e3, np.max(mask.x)/1e3, np.min(mask.y)/1e3, np.max(mask.y)/1e3))
            # dummy point for legend
            ax[1].plot(0, 0, 's', color=colors_dict[label], markersize=10, label=label) 
        ax[1].set_xlim(ax[0].get_xlim())
        ax[1].set_ylim(ax[0].get_ylim())
        ax[1].legend(loc='best')
        ax[1].set_xlabel('Easting [km]')

        # Save figure
        fig.savefig(fig_fn, dpi=300, bbox_inches='tight')
        print('Figure saved to file:', fig_fn)
        plt.close()
    elif plot_results:
        print('Masks figure exists in directory, skipping plotting.')

    return trees_mask_fn, snow_mask_fn, ss_mask_fn


def prepare_reference_elevations(coreg_dem_fn=None, ortho_dem_fn=None, bound_fn=None, 
                                 ss_mask_fn=None, coreg_stable_only=False, crs="EPSG:4326", out_dir=None):
    """
    Trim DEM(s) to the specified footprint and optionally, mask the DEM(s) to a stable surfaces mask. 

    Parameters
    ----------
    coreg_dem_fn: str or Path
        file name of the DEM used for coregistration
    ortho_dem_fn: str or Path
        file name of the DEM used for orthorectification (will not be masked if coreg_stable_only=True)
    bound_fn: str or Path
        file name of the geospatial file used for trimming the DEM(s)
    ss_mask_fn: str or Path
        file name of the stable surfaces mask
    coreg_stable_only: bool (default=False)
        whether to mask the coreg_dem using the stable surfaces mask
    crs: str (default="EPSG:4326)
        Coordinate Reference System of the outputs
    out_dir: str or Path
        path to the folder where outputs will be saved
    
    Returns
    ----------
    coreg_dem_fn: str
        file name of the trimmed (and masked if specified) coreg_dem
    ortho_dem_fn: str
        file name of the trimmed ortho_dem
    """
    # Make output directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Trim DEMs to SkySat footprint + 1km buffer to speed up computations
    def trim_dem(dem_fn, crop_shp_fn, out_dir, end_crs):
        # copy DEM to output directory
        shutil.copy2(dem_fn, out_dir)
        dem_fn = os.path.join(out_dir, os.path.basename(dem_fn))
        # Trim DEM to SkySat footprint + 1 km buffer
        dem_crop_fn = os.path.join(out_dir, os.path.splitext(os.path.basename(dem_fn))[0] + '_shpclip.tif')
        if not os.path.exists(dem_crop_fn):
            print("Cropping DEM")
            dem = gu.Raster(dem_fn)
            crop_shp = gu.Vector(crop_shp_fn).reproject(dem)
            dem_crop = dem.crop(crop_shp)
            if dem_crop.crs != end_crs:
                dem_crop = dem_crop.reproject(end_crs)
            dem_crop.save(dem_crop_fn)
        return dem_crop_fn
    
    coreg_dem_fn = trim_dem(coreg_dem_fn, bound_fn, out_dir, crs)
    if coreg_dem_fn != ortho_dem_fn:
        ortho_dem_fn = trim_dem(ortho_dem_fn, bound_fn, out_dir, crs)
    
    # Mask coreg DEM to stable surfaces
    if coreg_stable_only:
        def mask_coreg_dem(dem_fn, ss_mask_fn, out_dir):
            dem_stable_fn = os.path.join(out_dir, os.path.splitext(os.path.basename(dem_fn))[0] + '_stable.tif')
            if not os.path.exists(dem_stable_fn):
                print("Masking coreg DEM to stable surfaces")
                dem = rxr.open_rasterio(dem_fn).squeeze()
                ss_mask = rxr.open_rasterio(ss_mask_fn).squeeze().rio.reproject_match(dem)
                dem_masked = xr.where(ss_mask==1, dem, np.nan)
                dem_masked.rio.write_crs(dem.rio.crs, inplace=True)
                dem_masked.rio.to_raster(dem_stable_fn, dtype='float32', nodata=np.nan)
                print('Stable surfaces DEM saved to file:', dem_stable_fn)
            return dem_stable_fn
        coreg_dem_fn = mask_coreg_dem(coreg_dem_fn, ss_mask_fn, out_dir)
    
    return coreg_dem_fn, ortho_dem_fn


def execute_skysat_orthorectification(img_list, out_folder, dem='WGS84', tr=None, tsrs=None, cam_folder=None, ba_prefix=None,
                                      session=None, overlap_txt=None, copy_rpc=1, orthomosaic=0):
    """
    Mapproject images onto a reference DEM and optionally, create median mosaic of mapprojected images. 

    Parameters
    ----------
    img_list: list of str
        list of image file names
    out_folder: str or Path
        path to the folder where mapprojected images and cameras will be saved
    dem: str or Path (default="WGS84")
        reference DEM used for mapprojection. If None, will use the WGS84 datum.
    tr: float or str
        target spatial resolution of the mapprojected images (meters)
    tsrs: str
        target coordinate reference system of the mapprojected images (e.g., "EPSG:4326")
    cam_folder: str or Path

    ba_prefix: str or Path

    session: str

    overlap_txt: str or Path

    frame_index_fn: str or Path

    copy_rpc: bool

    orthomosaic: bool

    Returns
    ----------
    None
    """

    # If overlap list provided, need to remove images and cameras which were not optimized during bundle adjustment
    if overlap_txt is not None:
        initial_count = len(img_list)
        with open(overlap_txt) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        l_img = [x.split(' ')[0] for x in content]
        r_img = [x.split(' ')[1] for x in content]
        total_img = l_img + r_img
        uniq_idx = np.unique(total_img, return_index=True)[1]
        img_list = [total_img[idx] for idx in sorted(uniq_idx)]
        print(f"Out of the initial {initial_count} images, {len(img_list)} will be orthorectified using adjusted cameras")

    # Set up mapproject arguments
    img_prefix = [os.path.splitext(os.path.basename(img))[0] for img in img_list]
    out_list = [os.path.join(out_folder, img+'_map.tif') for img in img_prefix]
    session_list = [session] * len(img_list)
    dem_list = [dem] * len(img_list)
    tr_list = [tr] * len(img_list)
    srs_list = [tsrs]*len(img_list)
    # Get list of cameras for each image
    if session == 'pinhole':
        if ba_prefix:
            cam_list = [glob.glob(os.path.abspath(ba_prefix)+'-'+os.path.splitext(os.path.basename(x))[0]+'*.tsai')[0] for x in img_list]
            print("No of cameras is {}".format(len(cam_list)))
        else:
            print(os.path.join(os.path.abspath(cam_folder),os.path.splitext(os.path.basename(img_list[0]))[0]+'*.tsai'))
            cam_list = [glob.glob(os.path.join(os.path.abspath(cam_folder),os.path.splitext(os.path.basename(x))[0]+'*.tsai'))[0] for x in img_list]
    else:
        cam_list = [None] * len(img_list)
    ba_prefix_list = [ba_prefix] * len(img_list)
    
    # Determine how to split up jobs between threads
    ncpu = multiprocessing.cpu_count()
    njobs = 4
    threads = int(np.floor(ncpu/njobs))
    threads_list = [threads] * len(img_list)

    # Run mapproject in parallel
    print("Mapping given images")
    ortho_logs = p_map(asp.mapproject, img_list, out_list, session_list, dem_list, tr_list, srs_list, cam_list,
                       ba_prefix_list, [None]*len(img_list), threads_list, num_cpus=4)
    ortho_log = os.path.join(out_folder, 'ortho.log')
    print("Saving Orthorectification log at {}".format(ortho_log))
    with open(ortho_log,'w') as f:
        for log in ortho_logs:
            f.write(log)
    
    if copy_rpc == 1:
        print("Copying RPC from native image to orthoimage in parallel")
        copy_rpc_out = p_map(skysat.copy_rpc, img_list, out_list, num_cpus=iolib.cpu_count())
    
    if orthomosaic == 1:
        print("Will also produce median, weighted average and highest resolution orthomosaic")
        # sort images based on timestamps and resolutions
        img_list, time_list = skysat.sort_img_list(out_list)
        res_sorted_list = skysat.res_sort(out_list)

        # define mosaic prefix containing timestamps of inputs
        mos_prefix = '_'.join(np.unique([t.split('_')[0] for t in time_list]))+'__'+'_'.join(np.unique([t.split('_')[1] for t in time_list]))

        # define output filenames
        res_sorted_mosaic = os.path.join(out_folder, '{}_finest_orthomosaic.tif'.format(mos_prefix))
        median_mosaic = os.path.join(out_folder, '{}_median_orthomosaic.tif'.format(mos_prefix))
        wt_avg_mosaic = os.path.join(out_folder, '{}_wt_avg_orthomosaic.tif'.format(mos_prefix))
        indi_mos_list = [os.path.join(out_folder, f'{time}_first_orthomosaic.tif') for time in time_list]


        print("producing finest resolution on top mosaic, per-pixel median and wt_avg mosaic")
        all_3_view_mos_logs = p_map(asp.dem_mosaic, [res_sorted_list]*3, [res_sorted_mosaic,median_mosaic,wt_avg_mosaic], 
                                    ['None']*3, [None]*3, ['first','median',None],[None]*3,num_cpus=4)

        print("producing idependent mosaic for different views in parallel")
        indi_mos_count = len(time_list)
        if indi_mos_count>3:
            tile_size = 400
        else:
            tile_size = None

        indi_mos_log = p_map(asp.dem_mosaic,img_list, indi_mos_list, ['None']*indi_mos_count, [None]*indi_mos_count, 
            ['first']*indi_mos_count,[tile_size]*indi_mos_count)

        # write out log files
        out_log = os.path.join(out_folder, 'science_mode_ortho_mos.log')
        total_mos_log = all_3_view_mos_logs+indi_mos_log
        print("Saving orthomosaic log at {}".format(out_log))
        with open(out_log,'w') as f:
            for log in itertools.chain.from_iterable(total_mos_log):
                f.write(log)


def prep_stereo_jobs_df(overlap_pkl, true_stereo=True, cross_track=False):
    """
    Prepare dataframe from input plckle file containing overlapping images with percentages

    Parameters
    ----------
    overlap_list: str
        Path to pickle file containing overlapping images produced from skysat_overlap_parallel.py
    true_stereo: bool
        True means output dataframe has only pairs fromed by scenes from different views
    cross_track: bool
        whether to include cross-track stereo pairs

    Returns
    ----------
    df: Pandas Dataframe
        dataframe containing list of plausible overlapping stereo pairs
    """
    # check date, if date not equal drop
    # then check time, if time equal drop
    # if satellite unequal, drop
    # then check overlap percent
    # then make different folders for different time period
    # to add timestamp/convergence angle filter, as list grows
    df = pd.read_pickle(overlap_pkl)
    df['sat1'] = [os.path.basename(x).split('_', 15)[2].split('_')[0] for x in df.img1.values]
    df['sat2'] = [os.path.basename(x).split('_', 15)[2].split('_')[0] for x in df.img2.values]
    df['date1'] = [os.path.basename(x).split('_', 15)[0] for x in df.img1.values]
    df['date2'] = [os.path.basename(x).split('_', 15)[0] for x in df.img2.values]
    df['time1'] = [os.path.basename(x).split('_', 15)[1] for x in df.img1.values]
    df['time2'] = [os.path.basename(x).split('_', 15)[1] for x in df.img2.values]
    if true_stereo:
        # returned df has only those pairs which form true stereo
        df = df[df['time1'] != df['time2']]
        if not cross_track:
            df = df[df['date1'] == df['date2']]
            df = df[df['sat1'] == df['sat2']]
    # filter to overlap percentage of around 5%
    df['overlap_perc'] = df['overlap_perc']
    df = df[(df['overlap_perc'] > 10)]
    df['identifier_text'] = df['date1'] + '_' + df['time1'] + '_' + df['date2'] + '_' + df['time2']
    df.reset_index(drop=True, inplace=True)
    print("Number of pairs over which stereo will be attempted are {}".format(len(df)))
    return df


def execute_skysat_stereo(img_folder, out_folder, session='rpc', dem=None, texture='high', cam_folder=None, 
                          ba_prefix=None, block=1, crop_map=0, entry_point='pprc', threads=None, overlap_pkl=None, 
                          cross_track=False, correlator_mode=False):
    """
    """
    from p_tqdm import p_map
    img_folder = os.path.abspath(img_folder)
    try:
        img_list = sorted(glob.glob(os.path.join(img_folder, '*.tif')))
        temp = img_list[1]
    except BaseException:
        img_list = sorted(glob.glob(os.path.join(img_folder, '*.tiff')))
    if len(img_list) == 0:
        print("No images in the specified folder, exiting")
        sys.exit()

    if crop_map == 1:
        crop_map = True
    else: 
        crop_map = False
    
    # Define number of processes and threads per process
    if not threads:
        ncpu = multiprocessing.cpu_count()
        threads = 8
        nproc = int(np.floor(ncpu / threads))
        
    job_list = skysat.triplet_stereo_job_list(cross_track=cross_track, t=session,
        threads=threads, overlap_list=overlap_pkl, img_list=img_list, ba_prefix=ba_prefix, 
        cam_fol=cam_folder, dem=dem, crop_map=crop_map, texture=texture, outfol=out_folder, block=block,
        entry_point=entry_point)
    
    print(f'Will run {len(job_list)} jobs across {nproc} processes with {threads} threads each')

    if correlator_mode:
        job_list = [job + ['--correlator-mode'] for job in job_list]
    
    # decide on number of processes
    print(job_list[0])
    stereo_log = p_map(asp.run_cmd,['parallel_stereo']*len(job_list), job_list, num_cpus=nproc)
    stereo_log_fn = os.path.join(out_folder, 'stereo_log.log')
    print("Consolidated stereo log saved at {}".format(stereo_log_fn))


def dense_match_wrapper(stereo_dir, ba_dir, modify_overlap=False, img_fol=None, overlap_pkl=None, dense_match_pkl=None, out_overlap_pkl=None):
    """
    Copy dense matches from stereo to directory for bundle adjust use. Optionally, update the overlap list. 

    Parameters
    --------
    stereo_dir: str or Path
        path to folder containing stereo outputs
    ba_dir: str or Path
        path to folder where bundle_adjust will take place
    modify_overlap: bool (default=False)
        whether to modify the overlap list based on which image pairs successfully found dense matches
    img_fol: str or Path
        path to folder containing image files
    overlap_pkl: str or Path
        file name of the overlap pickle to be adjusted
    dense_match_pkl: str or Path
        file name of the pickle containing images with dense matches
    out_overlap_pkl: str or Path
        file name of the output adjusted overlap pickle
    
    Returns
    ---------
    None
    """
    # Identify matches from stereo
    triplet_stereo_matches = sorted(glob.glob(os.path.join(stereo_dir,'20*/*/run*-*disp*.match')))
    print('Found {} dense matches'.format(len(triplet_stereo_matches)))
    if  not os.path.isdir(ba_dir):
        os.makedirs(ba_dir)
    
    # Copy dense matches
    out_dense_match_list = [os.path.join(ba_dir,'run-'+os.path.basename(match).split('run-disp-',15)[1]) for match in triplet_stereo_matches]
    pbar = tqdm(total=len(triplet_stereo_matches))
    for idx,match in enumerate(triplet_stereo_matches):
        shutil.copy2(match, out_dense_match_list[idx])
        pbar.update(1)
    pbar.close()
    print("Copied all files successfully")
    
    # Update overlap list if specified
    if modify_overlap:
        orig_df = pd.read_pickle(overlap_pkl)
        dense_df = pd.read_pickle(dense_match_pkl)
        dense_img1 = list(dense_df.img1.values)
        dense_img2 = list(dense_df.img2.values)
        priority_list = list(zip(dense_img1,dense_img2))
        regular_img1 = [os.path.basename(x) for x in orig_df.img1.values]
        regular_img2 = [os.path.basename(x) for x in orig_df.img2.values]
        secondary_list = list(zip(regular_img1,regular_img2))
        # adapted from https://www.geeksforgeeks.org/python-extract-unique-tuples-from-list-order-irrespective/
        # note that I am using the more inefficient answer on purpose, because I want to use image pair order from the dense match overlap list
        total_list = priority_list + secondary_list
        final_overlap_set = set()
        temp = [final_overlap_set.add((a, b)) for (a, b) in total_list
                if (a, b) and (b, a) not in final_overlap_set]
        new_img1 = [os.path.join(img_fol,pair[0]) for pair in list(final_overlap_set)]
        new_img2 = [os.path.join(img_fol,pair[1]) for pair in list(final_overlap_set)]
        if not out_overlap_pkl:
            out_overlap = os.path.join(ba_dir, 'overlap_list_adapted_from_dense_matches.txt')
        else:
            out_overlap = os.path.join(ba_dir, out_overlap_pkl)
        
        print("Saving adjusted overlap list at {}".format(out_overlap))
        with open(out_overlap,'w') as f:
            for idx,img1 in enumerate(new_img1):
                out_str = '{} {}\n'.format(img1,new_img2[idx])
                f.write(out_str)


def bundle_adjustment(img_folder, ba_prefix, cam_folder, overlap_pkl=None, session='nadirpinhole',
                      ba_dem=None, ba_dem_uncertainty=10, cam_weight=0, num_iter=100, num_pass=2):
    """
    Run bundle adjustment on a set of images. Initial testing showed that select individual images had
    much higher errors in position/orientation compared to others, leading to insufficient adjustment upon 
    convergence when adjusting many images at once. Thus, this program adjusts one image at a time, starting with
    the most overlapping pair, fixing that pair, adjusting the next overlapping image, and iterating until all images are adjusted. 

    Parameters
    ---------
    img_folder: str or Path
        path to folder containing the images
    ba_prefix: str or Path
        prefix to use for bundle adjust outputs
    cam_folder: str or Path
        folder containing the cameras associated with each image
    overlap_pkl: str or Path
        file name of the overlapping stereo pairs pickle file (see "identify_overlapping_image_pairs")
    overlap_txt: str or Path
        file name of the overlapping stereo pairs text file (see "identify_overlapping_image_pairs")
    ba_dem: str or Path
        (optional) file name of the DEM to use in bundle_adjust. Note that the DEM should be well-aligned with the images.
    ba_dem_uncertainty: float (default=10)
        (optional) uncertainty parameter associated with the DEM (~meters)
    cam_weight: float (default=0)
        weight of the initial camera positions
    num_iter: int (default=100)
        number of bundle adjust iterations for each image adjustment
    num_pass: int (default=2)
        number of bundle adjust passes for each image adjustment
    
    Returns
    ----------
    None
    """
    # Create output directory
    out_dir = os.path.dirname(ba_prefix)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    # Grab list of images
    img_list = sorted(glob.glob(os.path.join(img_folder,'*.tif')) + glob.glob(os.path.join(img_folder, '*.tiff')))
    if len(img_list) < 2:
        if os.path.islink(img_list[0]):
            img_list = [os.readlink(x) for x in img_list]     
    
    # Load overlapping pairs
    overlap = pd.read_pickle(overlap_pkl)
    overlap = overlap.loc[(overlap['img1'].isin(img_list)) & (overlap['img2'].isin(img_list))]
    overlap = overlap.sort_values(by='overlap_perc', ascending=False)
    overlap.reset_index(drop=True, inplace=True)

    # Make sure all image and camera pairs exist
    # get camera extension
    if len(glob.glob(os.path.join(cam_folder, '*.tsai'))) > 0:
        cam_ext = '.tsai'
    else:
        cam_ext = '.TXT'
    # add camera columns to overlap dataframe
    def check_for_camera(cam_folder, img_fn, cam_ext):
        cam_fns = glob.glob(os.path.join(cam_folder, f"*{os.path.basename(img_fn).split('_basic')[0]}*{cam_ext}"))
        if len(cam_fns) > 0:
            cam_fn = cam_fns[0]
        else:
            cam_fn = np.nan
        return cam_fn
    overlap['cam1'] = [check_for_camera(cam_folder, x, cam_ext)
                       for x in overlap['img1'].values]
    overlap['cam2'] = [check_for_camera(cam_folder, x, cam_ext)
                       for x in overlap['img2'].values]
    
    # remove rows with no cameras
    overlap.dropna(inplace=True)
    # update image list
    img_list = sorted(pd.concat([overlap['img1'].drop_duplicates(), overlap['img2'].drop_duplicates()]).drop_duplicates().values)
    # Check if there are enough images
    if len(img_list) < 2:
        raise Exception('Less than two images with cameras found. Check file paths and cameras')
    print(f"Out of the initial {len(img_list)} images, {len(img_list)} will be orthorectified using adjusted cameras")

    # Set up bundle adjust base arguments
    ba_args = []
    ba_args.extend(['--threads', str(multiprocessing.cpu_count())])
    ba_args.extend(['-t', session])
    ba_args.extend(['-o', ba_prefix])
    # keypoint-finding args
    # relax triangulation error based filters to account for initial camera errors
    ba_args.extend(['--min-matches', '4'])
    ba_args.extend(['--disable-tri-ip-filter'])
    ba_args.extend(['--force-reuse-match-files'])
    ba_args.extend(['--ip-per-tile', '4000'])
    ba_args.extend(['--ip-inlier-factor', '0.2'])
    ba_args.extend(['--ip-num-ransac-iterations', '1000'])
    ba_args.extend(['--skip-rough-homography'])
    ba_args.extend(['--min-triangulation-angle', '0.0001'])
    # save control network created from match points
    ba_args.extend(['--save-cnet-as-csv'])
    # individually normalize images to properly stretch constraint 
    # helpful in keypoint detection
    ba_args.extend(['--individually-normalize'])
    # this generally assigns weight to penalize movement of camera extrinsics
    ba_args.extend(['--camera-position-weight', str(cam_weight)])
    # output updated cameras, not just the adjustments (only available for pinhole cameras)
    if session == 'nadirpinhole':
        ba_args.extend(['--inline-adjustments'])
    # specify number of passes and maximum iterations per pass
    ba_args.extend(['--num-iterations', str(num_iter)])
    ba_args.extend(['--num-passes', str(num_pass)])
    # add reference DEM if using
    if ba_dem:
        ba_args.extend(['--heights-from-dem', ba_dem])
        ba_args.extend(['--heights-from-dem-uncertainty', str(ba_dem_uncertainty)])


    # Run bundle adjustment in rounds...
    # First, adjust the first two images. Then, fix those two images and adjust the next most overlapping image. 
    # After each run, fix the adjusted images/cameras and continue until all images have been adjusted. 
    img_tba_list = img_list
    img_adjusted_list = []

    # Start with the most overlapping pair
    img1 = overlap.iloc[0]['img1']
    cam1 = overlap.iloc[0]['cam1']
    img2 = overlap.iloc[0]['img2']
    cam2 = overlap.iloc[0]['cam2']
    ba_img_list = [img1, img2]
    ba_cam_list = [cam1, cam2]
    fixed_indices = None

    i = 0 # iteration counter
    pbar = tqdm(total=len(img_list))
    while len(img_tba_list) > 0:
        # Set up round-specific arguments
        ba_args_round = ba_args.copy()
        # fix cameras (none on first round)
        if fixed_indices:
            ba_args_round.extend(['--fixed-camera-indices', fixed_indices])
        # add images and cameras
        ba_args_round += ba_img_list + ba_cam_list 

        # Run bundle adjust
        run_cmd('parallel_bundle_adjust', ba_args_round)

        # Set up next round
        # update adjusted and to-be-adjusted lists
        if i==0:
            img_adjusted_list += [img1, img2]
            pbar.update(2) # update progress bar
        else:
            img_adjusted_list += [img_tba]
            pbar.update(1) # update progress bar
        img_tba_list = [x for x in img_list if x not in img_adjusted_list]

        # check for remaining images
        if len(img_tba_list) > 0: 
            # filter the overlap dataframe
            # each row must have one adjusted and one un-adjusted image
            overlap_remaining = pd.concat([overlap.loc[~overlap['img1'].isin(img_adjusted_list)
                                                       & overlap['img2'].isin(img_adjusted_list)],
                                           overlap.loc[overlap['img1'].isin(img_adjusted_list)
                                                       & ~overlap['img2'].isin(img_adjusted_list)]
            ])
            # sort by decreasing overlap percentage
            overlap_remaining.sort_values(by='overlap_perc', ascending=False)
            overlap_remaining.reset_index(drop=True, inplace=True)
            # select the most overlapping image for adjustment
            if overlap_remaining['img1'].values[0] in img_adjusted_list:
                img_tba = overlap_remaining['img2'].values[0]
                cam_tba = overlap_remaining['cam2'].values[0]
            else:
                img_tba = overlap_remaining['img1'].values[0]
                cam_tba = overlap_remaining['cam1'].values[0]
            # get all images overlapping the tba image that have already been adjusted
            overlap_remaining_img = overlap_remaining.loc[(overlap_remaining['img1']==img_tba)
                                                            | (overlap_remaining['img2']==img_tba)]
            img_fixed_list = ([x for x in overlap_remaining_img['img1'].values if x!= img_tba] 
                              + [x for x in overlap_remaining_img['img2'].values if x!= img_tba])
            cam_fixed_list = ([x for x in overlap_remaining_img['cam1'].values if x!= cam_tba] 
                              + [x for x in overlap_remaining_img['cam2'].values if x!= cam_tba])
            # update bundle adjust image and cam lists
            ba_img_list = img_fixed_list + [img_tba]
            ba_cam_list = cam_fixed_list + [cam_tba]
            fixed_indices = ' '.join(np.arange(0,len(ba_img_list)-1).astype(str))

        i+=1
        
    pbar.close()
    print('Bundle adjust runs complete')

    # Check how many processes converged
    log_fns = sorted(glob.glob(ba_prefix + '-log*.txt'))
    nconv = 0
    for log_fn in log_fns:
        with open(log_fn, 'r') as f:
            log = f.read()
        if 'CONVERGENCE' in log:
            nconv += 1
    print(f'Number of bundle adjust runs that converged = {nconv} / {len(img_list)-1}')

def gridding_wrapper(pc_list,tr,tsrs=None):
    """
    Rasterize a list of point clouds. 

    Parameters
    ----------
    pc_list: list of str or Path
        list of point cloud file names to rasterize
    tr: float or int
        target resolution of the output raster
    tsrs: str
        target Coordinate Reference System of the outputs raster
    
    Returns
    ----------
    None
    """
    if tsrs is None:
        print("Projected Target CRS not provided, reading from the first point cloud")
        
        #fetch the PC-center.txt file instead
        # should probably make this default after more tests and confirmation with Oleg
        pc_center = os.path.splitext(pc_list[0])[0]+'-center.txt'
        with open(pc_center,'r') as f:
            content = f.readlines()
        X,Y,Z = [float(x) for x in content[0].split(' ')[:-1]]
        ecef_proj = 'EPSG:4978'
        geo_proj = 'EPSG:4326'
        ecef2wgs = Transformer.from_crs(ecef_proj,geo_proj)
        clat,clon,h = ecef2wgs.transform(X,Y,Z)
        epsg_code = f'EPSG:{geo.compute_epsg(clon,clat)}'
        print(f"Detected EPSG code from point cloud {epsg_code}") 
        tsrs = epsg_code
    n_cpu = multiprocessing.cpu_count()    
    point2dem_opts = asp.get_point2dem_opts(tr=tr, tsrs=tsrs,threads=1)
    job_list = [point2dem_opts + [pc] for pc in pc_list]
    p2dem_log = p_map(asp.run_cmd,['point2dem'] * len(job_list), job_list, num_cpus = n_cpu)
    print(p2dem_log)


def dem_mosaic_wrapper(dem_list, out_folder, tr=2, tsrs='EPSG:4326', tile_size=None):
    """
    Mosaic a list of DEMs using the median, count, and NMAD operators

    Parameters
    ----------
    dem_list: list of str or Path
        list of DEM file names to mosaic
    out_folder: str of Path
        path to the folder where mosaics will be saved
    tr: float or int (default=2)
        target resolution of the outputs mosaics
    tsrs: str (default="EPSG:4326")
        target Coordinate Reference System of the output mosaics
    tile_size: int
        (optional) size of the tiles for mosaicking in chunks
    
    Returns
    ----------
    None
    """
    # Make output folder if it does not exist
    if out_folder is None:
        out_folder = os.path.join(dir,'composite_dems')
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    # Mosaic DEMs
    print(f"Mosaicking {len(dem_list)} DEM strips using median, count, and NMAD operators")
    stats_list = ['median','count','nmad']
    for stat in stats_list:
        out_fn = os.path.join(out_folder, f'dem_{stat}_mos.tif')
        print(f"Creating {stat} mosaic")
        out_log = asp.dem_mosaic(dem_list, out_fn, tr=str(tr), tsrs=tsrs, stats=stat, tile_size=tile_size)
        print(out_log)
        # Save log
        out_log_fn = out_fn.replace('.tif', '.log')
        print("Saving triplet DEM mosaic log at {}".format(out_log_fn))
        with open(out_log_fn,'w') as f:
            f.write(out_log)


def align_dem(refdem_fn, dem_fn, out_dir, max_displacement=100, tr=2):
    """
    Align a DEM to a reference DEM using ASP's pc_align function in two steps. 
    First, use the Iterative Closest Point method for translational and rotational alignment. 
    Then, use the Nuth and Kaab method for subpixel translational alignment. 

    Parameters
    ----------
    refdem_fn: str or Path
        file name of the reference DEM
    dem_fn: str or Path
        file name of the DEM to be aligned
    out_dir: str or Path
        path to the folder where outputs will be saved
    max_displacement: float or int
        maximum displacement of the alignment, passed to pc_align
    
    Returns
    ----------
    dem_nk_out_fn: str
        file name of the aligned DEM
    """
    # Make output directory if it does not exist
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    # Round 1: ICP
    print('Round 1: Align DEMs using ICP point-to-plane method')
    # define outputs
    output_prefix = os.path.join(out_dir, 'run-icp')
    align_out_fn = output_prefix + '-trans_source.tif'
    grid_out_fn = output_prefix + '-trans_source-DEM.tif'
    # align
    if not os.path.exists(align_out_fn):
        align_cmd = ['--max-displacement', str(max_displacement), 
                        '--threads', str(multiprocessing.cpu_count()),
                        '--highest-accuracy', 
                        '--save-transformed-source-points', 
                        '--alignment-method', 'point-to-plane',
                        '-o', output_prefix,
                        refdem_fn, dem_fn]
        out = run_cmd('pc_align', align_cmd)
        print(out)
    # grid
    if os.path.exists(align_out_fn) and not os.path.exists(grid_out_fn):
        grid_cmd = ['--tr', str(tr), align_out_fn]
        out = run_cmd('point2dem', grid_cmd)
        print(out)
    
    # Round 2: Nuth and Kaab
    print('Round 2: Align DEMs using Nuth and Kaab method')
    # define outputs
    dem_nk_out_fn = os.path.join(out_dir, 'run-nk-DEM.tif')
    # load inputs
    if not os.path.exists(dem_nk_out_fn):
        dem = xdem.DEM(grid_out_fn, load_data=True)
        refdem = xdem.DEM(refdem_fn, load_data=True).reproject(dem)
        # align
        nk = xdem.coreg.NuthKaab().fit(refdem, dem)
        print(nk._meta)
        dem_nk = nk.apply(dem)
        # save output
        dem_nk.save(dem_nk_out_fn)
        print('Nuth and Kaab alignment complete!')
        print('Aligned DEM saved to:', dem_nk_out_fn)    

    return dem_nk_out_fn


def alignment_wrapper_single(ref_dem, source_dem, max_displacement, outprefix, align='point-to-plane',
                             trans_only=0, initial_align=None, threads=multiprocessing.cpu_count()):
    """
    Wrapper to align a DEM to a reference DEM using ASP's pc_align function. 

    Parameters
    ----------
    ref_dem: str or Path
        file name of the reference DEM
    source_dem_list: list of str or Path
        list of the DEMs to be aligned to the reference DEM
    max_displacement: float or int
        maximum displacement of the source DEMs to the reference, passed to pc_align
    outprefix: str or Path
        prefix of the output files
    align: str (default="point-to-plane")
        alignment method. See Section 16.54.16 in the ASP documentation for more info:
        https://stereopipeline.readthedocs.io/en/latest/tools/pc_align.html#pc-align-options
    trans_only: int (default=0)
        whether to only solve for translational alignment components (no rotation).
    intial_align: str or Path
        (optional) file name of the initial alignment transform
    threads: int (default = all available CPUs)
        how many threads to use for processing
    
    Returns
    ----------
    None
    """
    if trans_only == 0:
        trans_only = False
    else:
        trans_only = True
    asp.dem_align(ref_dem, source_dem, max_displacement, outprefix, align,
                  trans_only, threads=threads, initial_align=initial_align)


def alignment_wrapper_multi(ref_dem, source_dem_list, max_displacement, align='point-to-plane', 
                            initial_align=None, trans_only=0):
    """
    Wrapper to align multiple DEMs to a reference DEM.

    Parameters
    ----------
    ref_dem: str or Path
        file name of the reference DEM
    source_dem_list: list of str or Path
        list of the DEMs to be aligned to the reference DEM
    max_displacement: float or int
        maximum displacement of the source DEMs to the reference, passed to pc_align
    align: str (default="point-to-plane")
        alignment method. See Section 16.54.16 in the ASP documentation for more info:
        https://stereopipeline.readthedocs.io/en/latest/tools/pc_align.html#pc-align-options
    intial_align: str or Path
        (optional) file name of the initial alignment transform
    trans_only: int (default=0)
        whether to only solve for translational alignment components (no rotation).
    
    Returns
    ----------
    None
    """
    outprefix_list=['{}_aligned_to{}'.format(os.path.splitext(source_dem)[0],os.path.splitext(os.path.basename(ref_dem))[0]) for source_dem in source_dem_list]
    if trans_only == 0:
        trans_only = False
    else:
        trans_only = True
    n_source = len(source_dem_list)
    
    initial_align = [initial_align]*n_source
    ref_dem_list=[ref_dem] * n_source
    max_disp_list=[max_displacement] * n_source
    align_list=[align] * n_source
    trans_list=[trans_only] * n_source
    p_umap(asp.dem_align,ref_dem_list,source_dem_list,max_disp_list,outprefix_list,
           align_list,trans_list,[1]*n_source,initial_align,num_cpus = multiprocessing.cpu_count())
    
def align_cameras_wrapper(input_camera_list, transform_txt, outfolder, rpc=0, dem='None', img_list=None):
    """
    Wrapper for aligning a list of cameras using a transform matrix output from pc_align. 

    Parameters
    ----------
    input_camera_list: list of str or Path
        list of input cameras
    transform_txt: str or Path
        file name of the transform matrix
    outfolder: str or Path
        path to folder where aligned cameras will be saved
    rpc: int (default=0)
        whether input cameras are RPC models
    img_list: list of str or Path
        list of the images associated with each camera
    
    Returns
    ----------
    None
    """

    n_cam=len(input_camera_list)
    if (rpc == 1) & (dem != 'None'):
        print("Will also write RPC files")
        rpc = True
    else:
        dem = None
        img_list = [None] * n_cam
        rpc = False
    transform_list = [transform_txt]*n_cam
    outfolder = [outfolder] * n_cam
    write = [True] * n_cam
    rpc = [rpc] * n_cam
    dem = [dem] * n_cam
    
    p_umap(asp.align_cameras, input_camera_list, transform_list, outfolder, write, rpc, dem,
           img_list, num_cpus=multiprocessing.cpu_count())
                

def plot_composite_fig(ortho_fn, georegistered_median_dem_fn, count_fn, nmad_fn, outfn=None):
    """
    Plot figure of the final orthomosaic, DEM, count, and NMAD mosaics

    Parameters
    ----------
    ortho_fn: str or Path
        file name of the orthomosaic
    georegistered_median_dem_fn: str or Path
        file name of the DEM mosaic
    count_fn: str or Path
        file name of the count mosaic
    nmad_fn: str or Path
        file name of the NMAD mosaic
    outfn: str or Path
        file name of the output figure
    
    Returns
    ----------
    None
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import rasterio as rio
    from matplotlib import colors
    import numpy as np

    # Read files
    def read_data(raster):
        data = raster.read(1)
        data = np.ma.masked_where(data == raster.nodata, data)
        return data
    ortho = rio.open(ortho_fn)
    ortho_data = read_data(ortho)
    dem = rio.open(georegistered_median_dem_fn)
    dem_data = read_data(dem)
    count = rio.open(count_fn)
    count_data = read_data(count)
    nmad = rio.open(nmad_fn)
    nmad_data = read_data(nmad)

    def remove_ticks(axes):
        axes.set_xticks([])
        axes.set_yticks([])

    def add_colorbar(image, axes):
        x0, width = axes.get_position().x0, axes.get_position().width
        cax = fig.add_axes([x0, 0.1, width, 0.03])
        cbar = plt.colorbar(image, cax=cax, orientation='horizontal')
        return cbar

    # Plot
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    fig.subplots_adjust(bottom=0.15)
    # Orthoimage
    ax[0].imshow(ortho_data, cmap='Greys_r',
                extent=(ortho.bounds.left/1e3, ortho.bounds.right/1e3, ortho.bounds.bottom/1e3, ortho.bounds.top/1e3))
    ax[0].set_title('Orthoimage')
    remove_ticks(ax[0])
    # Add scale bar
    scalebar = AnchoredSizeBar(ax[0].transData,
                            1, '1 km', 'lower right', 
                            pad=0.3,
                            color='black',
                            frameon=True,
                            size_vertical=0.01)
    ax[0].add_artist(scalebar)
    # Shaded relief
    ls = colors.LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(dem_data, vert_exag=5)
    ax[1].imshow(hs, cmap='Greys_r',
                extent=(dem.bounds.left/1e3, dem.bounds.right/1e3, dem.bounds.bottom/1e3, dem.bounds.top/1e3))
    im = ax[1].imshow(dem_data, cmap='terrain', alpha=0.7, 
                    extent=(dem.bounds.left/1e3, dem.bounds.right/1e3, dem.bounds.bottom/1e3, dem.bounds.top/1e3))
    remove_ticks(ax[1])
    add_colorbar(im, ax[1])
    ax[1].set_title('DEM')
    # Count map
    im = ax[2].imshow(count_data, cmap='viridis', #clim=(0,np.nanpercentile(count_data, 95)),
                    extent=(count.bounds.left/1e3, count.bounds.right/1e3, count.bounds.bottom/1e3, count.bounds.top/1e3))
    remove_ticks(ax[2])
    add_colorbar(im, ax[2])
    ax[2].set_title('DEM counts')
    # NMAD
    # cmap = colors.LinearSegmentedColormap.from_list('custom', ['w', '#67000d'])
    im = ax[3].imshow(nmad_data, cmap='Reds', #clim=(0, np.nanpercentile(nmad_data, 95)),
                extent=(nmad.bounds.left/1e3, nmad.bounds.right/1e3, nmad.bounds.bottom/1e3, nmad.bounds.top/1e3))
    remove_ticks(ax[3])
    add_colorbar(im, ax[3])
    ax[3].set_title('NMAD')

    fig.savefig(outfn, dpi=300, bbox_inches='tight')
    print('Final figure saved to file:', outfn)
    plt.close()


def add_cam_intrinsics(input_file, output_file, distortion_model='TSAI',
                       intrinsics={'k1': -1e-12, 'k2': -1e-12, 'k3': -1e-12, 'p1': -1e-12, 'p2': -1e-12}):
    """
    Add distortion model parameters to cameras file for optimization in bundle_adjust, etc.

    Parameters
    ----------
    input_file: str or Path
        file name of the input camera
    output_file: str or Path
        file name of the output camera
    distortion_model: str (default="TSAI")
        which distortion model to use. See Section 20.1.2 in ASP's documentation for info: 
        https://stereopipeline.readthedocs.io/en/latest/pinholemodels.html
    intrinsics: dict
        distortion parameters and values
    
    Returns
    ----------
    None
    """
    # Read the .tsai file
    with open(input_file, 'r') as file:
        lines = file.readlines()
    updated_lines = lines
    # Remove NULL line
    updated_lines = [line for line in updated_lines if line!='NULL\n']
    # Add the distortion model
    if f"{distortion_model}\n" not in updated_lines:
        updated_lines.append(f"{distortion_model}\n")
    # Add distortion parameters
    for key in intrinsics.keys():
        addition = f"{key} = {intrinsics[key]:.8f}\n"
        if addition not in updated_lines:
            updated_lines.append(addition)
    # Save the updated file
    with open(output_file, 'w') as file:
        file.writelines(updated_lines)
    # print('Saved updated camera file:', output_file)


def align_individual_dems(dem_list, refdem_fn, out_dir, max_displacement=40, tr=0.5):
    """
    Coregister a list of DEMs to a reference DEM. For DEMs without at least 10% coverage of the reference DEM,
    the function will create a median mosaic of the aligned DEMs and iteratively coregister remaining DEMs 
    until no additional DEMs have sufficient overlap.
    
    Parameters
    ----------  
    dem_list: list
        List of DEM file paths to be coregistered.
    refdem_fn: str
        File path to the reference DEM.
    out_dir: str
        Output directory for aligned DEMs.
    max_displacement: int
    tr: float
        Resolution of the output DEMs.
    
    Returns
    ----------
    None
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    # Define function to get percent coverage of the reference DEM for a given DEM
    def get_coverage(refdem, dem_fn):
        dem = xdem.DEM(dem_fn, load_data=True)
        ref = refdem.reproject(dem)
        npx = np.where((dem.data.mask == False) & (ref.data.mask == False), 1, 0).sum()
        return {"filename": dem_fn, "refdem_percent_coverage": npx / dem.data.size * 100}

    # Define function to align a DEM to the reference DEM
    def align_dem(dem_fn, refdem_fn, out_dir, max_displacement=40, threads=4, tr=0.5, output_prefix=None):
        if output_prefix is None:
            output_prefix = os.path.join(out_dir, os.path.split(os.path.dirname(dem_fn))[1] + '-run')
        align_out_fn = output_prefix + '-trans_source.tif'
        grid_out_fn = output_prefix + '-trans_source-DEM.tif'
        if not os.path.exists(align_out_fn):
            align_cmd = ['--max-displacement', str(max_displacement), '--threads', str(threads),
                         '--highest-accuracy', '--save-transformed-source-points', '-o', output_prefix,
                         refdem_fn, dem_fn]
            run_cmd('pc_align', align_cmd)
        if os.path.exists(align_out_fn) and not os.path.exists(grid_out_fn):
            grid_cmd = ['--tr', str(tr), align_out_fn]
            run_cmd('point2dem', grid_cmd)
    
    # Define function to create median mosaic of a list of DEMs
    def create_median_mosaic(dem_list, out_fn):
        mosaic_cmd = dem_list + ['--median', '--tr', str(tr), '-o', out_fn]
        run_cmd('dem_mosaic', mosaic_cmd)

    # Calculate percent coverage of the reference DEM for each DEM
    print('Calculating percent coverage of the reference DEM for each DEM')
    refdem = xdem.DEM(refdem_fn, load_data=True)
    coverage_data = p_map(lambda x: get_coverage(refdem, x), dem_list, num_cpus=int(multiprocessing.cpu_count()/2))
    coverage_df = pd.DataFrame(coverage_data)
    
    # ROUND 1: Coregister DEMs with at least 10% overlap to the reference DEM
    dem_align_list = coverage_df[coverage_df['refdem_percent_coverage'] >= 10]['filename'].tolist()
    if not dem_align_list:
        print("No DEMs with at least 10% coverage of the reference DEM.")
        return
    print(f'Number of DEMs with reference DEM coverage >= 10%: {len(dem_align_list)}')
    for dem_fn in tqdm(dem_align_list):
        align_dem(dem_fn, refdem_fn, out_dir, max_displacement=max_displacement, threads=multiprocessing.cpu_count())
    # Create median mosaic of aligned DEMs
    mosaic_fn = os.path.join(out_dir, 'aligned_dem_mosaic.tif')
    print(f'Creating median mosaic of {len(dem_align_list)} aligned DEMs')
    create_median_mosaic(dem_align_list, mosaic_fn)
    
    # ROUND 2+: Coregister the remaining DEMs to the median mosaic of the aligned DEMs
    remaining_dems = [x for x in dem_list if x not in dem_align_list]
    while remaining_dems:        
        # Check which remaining DEMs have at least 10% coverage of the new median mosaic
        mosaic = xdem.DEM(mosaic_fn, load_data=True)
        coverage_data = p_map(lambda x: get_coverage(mosaic, x), remaining_dems, num_cpus=int(multiprocessing.cpu_count()/2))
        coverage_df = pd.DataFrame(coverage_data)
        new_align_list = coverage_df[coverage_df['refdem_percent_coverage'] >= 10]['filename'].tolist()
        if not new_align_list:
            print("No additional DEMs with at least 10% coverage of the median mosaic. Stopping.")
            break
        print(f'Number of additional DEMs with median mosaic coverage >= 10%: {len(new_align_list)}')
        # Coregister the new DEMs
        for dem_fn in tqdm(new_align_list):
            align_dem(dem_fn, mosaic_fn, out_dir, max_displacement=max_displacement, threads=multiprocessing.cpu_count())
        # Update the list of aligned and remaining DEMs
        dem_align_list.extend(new_align_list)
        remaining_dems = [x for x in remaining_dems if x not in dem_align_list]

        # Create new median mosaic of all aligned DEMs
        final_mosaic_fn = os.path.join(out_dir, 'aligned_dem_mosaic.tif')
        print(f'Creating new median mosaic of {len(dem_align_list)} aligned DEMs')
        create_median_mosaic(dem_align_list, final_mosaic_fn)
    
    print("Coregistration process complete!")


def filter_stereo_jobs(mapproj_stats_fn=None, gsd_thresh=4, overlap_pkl=None, overlap_filt_pkl=None):
    """
    Use bundle adjustment mapproj stats to filter which images get run through stereo. 
    See the Ames Stereo Pipeline documentation 16.5.11.10 for more info: https://stereopipeline.readthedocs.io/en/latest/tools/bundle_adjust.html#bundle-adjust
    
    Parameters
    ----------
    mapproj_stats_fn: str
        path to the "*-mapproj_match_offset_stats.txt" file output from bundle adjustment run with the "--mapproj-dem" flag.
    gsd_thresh: int,float
        threshold for the maximum median Ground Sample Distance (GSD) of a given image between mapprojected points w.r.t. other images
    overlap_pkl: str
        file name of the pickle file containing image pairs with stereo overlap
    overlap_filt_pkl:
        output file name for the filtered stereo overlap list (pickle format)
    
    Returns
    ----------
    img_list_filt: list of str
        list of image file names to run stereo on
    overlap_filt_pkl: str
        output file name for the filtered stereo overlap list (pickle format) 
    overlap_filt_txt
        output file name for the filtered stereo overlap list (TXT format)
    """
    # Read the mapproj stats file
    with open(mapproj_stats_fn, 'r') as file:
        lines = file.readlines()
    column_names = lines[1].lstrip('#').strip().split()
    mapproj_df = pd.read_csv(mapproj_stats_fn, skiprows=2, names=column_names, sep=' ')

    # Filter images using the median GSD difference w.r.t. other images
    mapproj_df_filt = mapproj_df.loc[mapproj_df['50%'] < gsd_thresh]
    img_list_filt = sorted(mapproj_df_filt['image_name'].values)
    print(f'{len(mapproj_df)-len(mapproj_df_filt)} images removed due to less reliable bundle adjustment results, {len(mapproj_df_filt)} images remain for stereo workflow.')

    # Update overlap file for stereo jobs
    if overlap_filt_pkl is None:
        overlap_filt_pkl = overlap_pkl.replace('.pkl', '_filtered.pkl')
    overlap = pd.read_pickle(overlap_pkl)
    overlap_filt = overlap.loc[(overlap['img1'].isin(img_list_filt)) & (overlap['img2'].isin(img_list_filt))]
    overlap_filt.to_pickle(overlap_filt_pkl)
    # Save as txt file as well
    overlap_filt_txt = os.path.splitext(overlap_filt_pkl)[0] + '.txt'
    overlap_filt[['img1', 'img2']].to_csv(overlap_filt_txt, sep=' ', index=False, header=False)
    print(f'Reduced number of stereo jobs from {len(overlap)} to {len(overlap_filt)}')
    
    return img_list_filt, overlap_filt_pkl, overlap_filt_txt


def coregister_individual_dems(in_dir, out_dir, griddem_fn):
    """
    Coregister individual DEMs to each other using the ICP point-to-plane method.
    
    Parameters
    ----------
    in_dir: str
        path to input folder containing subfolders with DEMs. Typically, the "final_pinhole_stereo" folder generated in the pipeline.
    out_dir: str
        path to output folder
    griddem_fn: str
        file name of the reference DEM used for gridding before coregistration
    
    Returns
    ----------
    dem_mos_fn: str
        file name of the median DEM mosaic created from all coregistered DEMs. 
    nmad_mos_fn: str
        file name of the NMAD mosaic created from all coregistered DEMs. 
    count_mos_fn: str
        file name of the NMAD mosaic created from all coregistered DEMs.
    """
    # Define some functions
    def get_dem_bbox(dem_fn):
        xmin, ymin, xmax, ymax = list(gu.Raster(dem_fn).bounds)
        bbox = Polygon([[xmin, ymin], [xmax, ymin],
                        [xmax, ymax], [xmin, ymax], [xmin, ymin]])
        return bbox

    def get_dem_area(dem_fn):
        dem_bbox = get_dem_bbox(dem_fn)
        return dem_bbox.area

    def get_overlapping_dems(reference_fn, dem_list):
        reference_bbox = get_dem_bbox(reference_fn)
        overlapping_dem_list = []
        for dem_fn in dem_list:
            dem_bbox = get_dem_bbox(dem_fn)
            if dem_bbox.intersects(reference_bbox):
                overlapping_dem_list.append(dem_fn)
        # Filter to those with overlapping data values
        overlapping_dem_dict = {}
        reference = xdem.DEM(reference_fn)
        for dem_fn in overlapping_dem_list:
            dem = xdem.DEM(dem_fn).reproject(reference)
            ddem = dem - reference
            n_nodata = len(dem.data.mask[ddem.data.mask==False])
            if n_nodata > 0:
                overlapping_dem_dict[dem_fn] = n_nodata * reference.res[0] * reference.res[1]
        return overlapping_dem_dict

    # def align_dems(reference_fn, source_fn, out_prefix):
        # reference = xdem.DEM(reference_fn).reproject(grid_dem)
        # source = xdem.DEM(source_fn).reproject(grid_dem)
        # nk = xdem.coreg.NuthKaab(subsample=1).fit(reference, source)
        # source_corr = nk.apply(source)
        # source_corr.save(source_out_fn)

    def mosaic_dems(dem_list, out_fn, stat, print_output=False):
        threads = multiprocessing.cpu_count()
        args = ['--tr', '2', 
                f'--{stat}',
                '--threads', str(threads), 
                '-o', out_fn] + dem_list 
        out = run_cmd('dem_mosaic', args)
        if print_output:
            print(out)
        return out

    # Load DEM for gridding
    griddem = xdem.DEM(griddem_fn)
    griddem = griddem.reproject(dst_crs=griddem.crs, dst_res=[5,5])
    
    # Create output folder
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Copy and rename individual DEMs into out_fol
    dem_list = glob.glob(os.path.join(in_dir, '20*', '20*', 'run-DEM.tif'))
    for dem in dem_list:
        new_dem = os.path.join(out_dir, dem.split('/')[-2] + '_run-DEM.tif')
        if not os.path.exists(new_dem):
            shutil.copy2(dem, new_dem)
    dem_list = glob.glob(os.path.join(out_dir, '20*_map_run-DEM.tif'))

    # Identify the largest DEM
    dem_areas = np.array([get_dem_area(dem_fn) for dem_fn in dem_list])
    imax = np.argwhere(dem_areas==np.max(dem_areas)).ravel()[0]
    reference_dem_fn = dem_list[imax]
    print(f'Starting with {reference_dem_fn} as the reference')
    coregistered_list = [reference_dem_fn]
    tba_dem_list = [dem_fn for dem_fn in dem_list if dem_fn!=reference_dem_fn]
    print(f'{len(tba_dem_list)} DEMs to be coregistered')

    # Start a progress bar
    pbar = tqdm(total=len(dem_list))
    pbar.update(1)

    # Begin coregistration
    while tba_dem_list:
        # Get overlapping DEMs
        overlapping_dems = get_overlapping_dems(reference_dem_fn, tba_dem_list)
        # sort by area of overlap
        overlapping_dems = dict(sorted(overlapping_dems.items(), key=lambda item: item[1], reverse=True))
        if len(overlapping_dems) > 0:
            source_dem = list(overlapping_dems.keys())[0]
            dem_out_fn = os.path.splitext(source_dem)[0] + '_coreg-trans_source-DEM.tif'
            if not os.path.exists(dem_out_fn):
                out_prefix = os.path.splitext(source_dem)[0] + '_coreg'
                alignment_wrapper_single(reference_dem_fn, source_dem, max_displacement=40, outprefix=out_prefix,
                                         align='point-to-plane', trans_only=1, initial_align=None)
            coregistered_list.append(dem_out_fn)

        else:
            # Restart with the largest remaining DEM
            print('No overlapping DEMs, restarting with the largest')
            tba_dem_areas = np.array([get_dem_area(dem_fn) for dem_fn in tba_dem_list])
            imax = np.argwhere(tba_dem_areas==np.max(tba_dem_areas)).ravel()[0]
            source_dem = tba_dem_list[imax]
            reference_dem_fn = source_dem
            coregistered_list.append(source_dem)

        # Create an intermediate DEM from all coregistered DEMs using median statistic
        dem_mos_fn = os.path.join(out_dir, 'mosaic_median.tif')
        mosaic_dems(coregistered_list, dem_mos_fn, stat='median')
        
        # Update reference DEM to intermediate DEM
        tba_dem_list.remove(source_dem)
        reference_dem_fn = dem_mos_fn
        
        pbar.update(1)

    pbar.close()

    # Create NMAD and count mosaics
    print('Creating NMAD and count mosaics')
    nmad_mos_fn = os.path.join(out_dir, 'mosaic_nmad.tif')
    out = mosaic_dems(coregistered_list, nmad_mos_fn, stat='nmad')
    count_mos_fn = os.path.join(out_dir, 'mosaic_count.tif')
    out = mosaic_dems(coregistered_list, count_mos_fn, stat='count')
    
    return dem_mos_fn, nmad_mos_fn, count_mos_fn