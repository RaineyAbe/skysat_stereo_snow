#! /usr/bin/env python

import os
from glob import glob
import numpy as np
import subprocess
import geoutils as gu
import xdem
import multiprocessing
from p_tqdm import p_map
from shapely.geometry import Polygon, Point
import pandas as pd
import geopandas as gpd
import itertools
import shutil
import rasterio as rio
from rasterio.transform import RPCTransformer
import re
import rioxarray as rxr
import xarray as xr
import matplotlib.pyplot as plt
from typing import List, Optional
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


def run_cmd(bin: str = None, 
            args: list = None, **kw) -> str:
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
    # binpath = "/Users/raineyaberle/Research/PhD/SnowDEMs/StereoPipeline-3.5.0-alpha-2024-10-05-x86_64-OSX/bin/" + bin
    binpath = "/bsuhome/raineyaberle/StereoPipeline-3.6.0-alpha-2025-07-04-x86_64-Linux/bin/" + bin
    # binpath = shutil.which(bin)
    call = [binpath,]
    if args is not None: 
        call.extend(args)
    try:
        out = subprocess.run(call,check=True,capture_output=True,encoding='UTF-8').stdout
    except:
        out = "the command {} failed to run, see corresponding asp log".format(call)
    return out


def setup_parallel_jobs(total_jobs: int = None) -> tuple[int, int]:
    """
    Determine the number of parallel jobs to run and threads per job.

    Parameters
    ----------
    total_jobs: int
        The total number of jobs to run (e.g., number of stereo pairs).

    Returns
    -------
    njobs: int
        Number of parallel processes to run.
    threads_per_job: int
        Number of threads to allocate per process.
    """
    total_cpus = multiprocessing.cpu_count()

    if total_jobs <= 1:
        njobs = 1
    elif total_jobs <= 10:
        njobs = min(2, total_jobs)
    elif total_jobs <= 100:
        njobs = min(4, total_jobs)
    else:
        njobs = min(4, total_jobs)

    threads_per_job = max(1, total_cpus // njobs)

    print(f"Will run {total_jobs} jobs across {njobs} CPU with {threads_per_job} threads per CPU")

    return njobs, threads_per_job


def convert_wgs_to_utm(lon: float = None, 
                       lat: float = None) -> str:
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
    # check for Greenland
    if (lat > 58) & (lon < -9) & (lon > -74):
        epsg_code = "EPSG:5938"
        print(f'Optimal CRS = Greenland Polar Stereographic {epsg_code}')

    # check for Antarctica
    elif (lat < -60):
        epsg_code = "EPSG:3031"
        print(f"Optimal CRS = Antarctic Polar Stereographic {epsg_code}")

    # otherwise, calculate UTM zone
    else:
        utm_band = str(int((np.floor((lon + 180) / 6) % 60) + 1))
        if len(utm_band) == 1:
            utm_band = '0' + utm_band
        if lat >= 0:
            epsg_code = 'EPSG:326' + utm_band
            return epsg_code
        epsg_code = 'EPSG:327' + utm_band
        print('Optimal UTM zone = ', epsg_code)

    return epsg_code


def rpc_image_latlon_bounds(img_fn: str = None, 
                            height: float = 0.0) -> tuple[float, float, float, float]:
    """
    Get bounding box (min lon, min lat, max lon, max lat) for image with RPC metadata.

    Parameters
    ----------
    img_fn: str
        Path to image file with RPCs.
    height: float
        Ground height in meters used for projection.

    Returns
    ----------
    tuple: (min_lon, min_lat, max_lon, max_lat)
    """
    with rio.open(img_fn) as src:
        if not src.rpcs:
            raise ValueError("Image does not contain RPC metadata.")

        transformer = RPCTransformer(src.rpcs)

        width = src.width
        height_px = src.height

        # Image corners (col, row)
        pixel_coords = [
            (0, 0),                    # top-left
            (width - 1, 0),            # top-right
            (width - 1, height_px - 1),# bottom-right
            (0, height_px - 1)         # bottom-left
        ]

        cols, rows = zip(*pixel_coords)
        zs = [height] * 4

        lons, lats = transformer.xy(cols, rows, zs)

        min_lon = np.min(lons)
        max_lon = np.max(lons)
        min_lat = np.min(lats)
        max_lat = np.max(lats)

        return min_lon, min_lat, max_lon, max_lat


def calculate_images_footprint(img_list: list = None, 
                               out_folder: str = None) -> tuple[str, str]:
    """
    Save a bounding box geopackage for all images in a folder.

    Parameters
    ----------
    img_list: list
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
    # Make sure output directory exists
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    # Define output file
    bound_buffer_fn = os.path.join(out_folder, 'image_bounds_buffer1km.gpkg')

    # Convert RPCs to image bounds
    n_proc = multiprocessing.cpu_count()
    bounds_list = p_map(rpc_image_latlon_bounds, img_list, num_cpus=n_proc)
    
    # Get the min and max bounds from list
    min_lons = np.array([x[0] for x in bounds_list])
    min_lats = np.array([x[1] for x in bounds_list])
    max_lons = np.array([x[2] for x in bounds_list])
    max_lats = np.array([x[3] for x in bounds_list])
    min_lon, max_lon = np.min(min_lons), np.max(max_lons)
    min_lat, max_lat = np.min(min_lats), np.max(max_lats)
    # convert to polygon
    bounds_poly = Polygon([[min_lon, min_lat], [max_lon, min_lat], [max_lon, max_lat], 
                           [min_lon, max_lat], [min_lon, min_lat]])    
    
    # Convert to GeoDataFrame
    bound_shp = gpd.GeoDataFrame(index=[0], geometry=[bounds_poly], crs="EPSG:4326")

    # Reproject optimal UTM zone
    utm_epsg = convert_wgs_to_utm(bound_shp.geometry[0].centroid.coords.xy[0][0],
                                  bound_shp.geometry[0].centroid.coords.xy[1][0])
    bound_shp = bound_shp.to_crs(utm_epsg)
    
    # Buffer by 1km
    bound_buffer_shp = bound_shp.copy()
    bound_buffer_shp['geometry'] = bound_buffer_shp['geometry'].buffer(1e3)
    
    # Save to file
    bound_buffer_shp.to_file(bound_buffer_fn, driver='GPKG')
    print('Image bounds + 1km buffer saved to file:', bound_buffer_fn)

    return bound_buffer_fn, utm_epsg


def calculate_baseline_to_height_ratio(img1: str = None, 
                                       img2: str = None, 
                                       utm_epsg: str = None) -> float:
    """
    Calculate the baseline to height ratio for a pair of images.

    Parameters
    ----------
    img1: str
        file name of the first image
    img2: str
        file name of the second image
    utm_epsg: str
        EPSG code for the optimal UTM zone, e.g. "EPSG:32601"
    
    Returns
    ----------
    b_h_ratio: float
        baseline to height ratio, where baseline is the distance between camera centers and height is the average height of the two images
    """
    # iterate over images
    cams_list, h_list = [], []
    for img in [img1, img2]:
        # get camera center coordinates and heights
        with rio.open(img) as src:
            h = src.rpcs.height_off
            lat = src.rpcs.lat_off
            lon = src.rpcs.long_off
        # reproject to UTM for distance calculations
        gdf = gpd.GeoDataFrame(index=[0], geometry=[Point(lon, lat)], crs="EPSG:4326")
        gdf = gdf.to_crs(utm_epsg)
        x = gdf.geometry[0].coords.xy[0][0]
        y = gdf.geometry[0].coords.xy[0][0]
        # save in arrays
        cams_list += [[x,y]]
        h_list += [h]
    # calculate baseline
    diff = np.array(cams_list[0]) - np.array(cams_list[1])
    b = np.linalg.norm(diff)
    h_mean = np.nanmean(np.array(h_list))
    # calculate B/H ratio
    return float(b / h_mean)
    

def clip_raster(raster_fn: str = None, 
                crop_shp_fn: str = None, 
                t_crs: str = "EPSG:4326", 
                out_dir: str = None) -> str:
    """
    Trim DEM(s) to the specified footprint and optionally, mask the DEM(s) to a stable surfaces mask. 

    Parameters
    ----------
    raster_fn: str
        file name of the raster
    crop_shp_fn: str
        file name of the geospatial file used for trimming the DEM(s)
    t_crs: str (default="EPSG:4326")
        Target Coordinate Reference System of the outputs
    out_dir: str
        path to the folder where outputs will be saved
    
    Returns
    ----------
    raster_clip_fn: str
        file name of the clipped raster
    """
    # Make output directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Define output file
    raster_clip_fn = os.path.join(out_dir, os.path.splitext(os.path.basename(raster_fn))[0] + '_clipped.tif')
    if os.path.exists(raster_clip_fn):
        print('Clipped raster already exists in directory, skipping.')
        return raster_clip_fn

    # Clip raster
    print('Clipping raster to input polygon')
    raster = gu.Raster(raster_fn)
    clip_shp = gu.Vector.from_file(crop_shp_fn).reproject(raster)
    raster_crop = raster.crop(clip_shp)
    if raster_crop.crs != t_crs:
        print(f'Reprojecting raster to {t_crs}')
        raster_crop = raster_crop.reproject(crs=t_crs)
    raster_crop.save(raster_clip_fn)
    print('Clipped raster saved to file:', raster_clip_fn)

    
    return raster_clip_fn


def apply_masks_to_images(
        image_list: List[str],
        mask_classes: List[str] = ["water_check"],
        confidence_threshold: Optional[int] = None,
        mask_unusable: bool = True,
        out_dir: str = None,
        copy_cams: bool = False
        ) -> List[str]:
    """
    Apply UDM2 and/or check for ice for a list of Planet images.

    Parameters
    ----------
    image_list: List[str]
        List of SkySat image file paths.
    mask_classes: List[str]
        UDM2 classes to mask and/or water check, which removes images with > 95% water.
        Options: "cloud", "cloud_shadow", "snow", "light_haze", "water_check"
    confidence_threshold: int, optional
        Mask pixels with confidence < threshold.
    mask_unusable: bool
        Whether to mask pixels marked as 'unusable' in Band 8.
    out_dir: str
        Output directory for masked images.
    copy_cam: bool
        Whether to copy original cameras to out_dir

    Returns
    -------
    List[str]
        List of masked or copied image filenames.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Define UDM2 bands
    class_to_band = {
        "clear": 0,
        "snow": 1,
        "cloud_shadow": 2,
        "light_haze": 3,
        "heavy_haze": 4,
        "cloud": 5,
        "confidence": 6,
        "unusable": 7
    }

    def process_image(image_fn):
        base = os.path.basename(image_fn)
        udm2_fn = image_fn.replace("_basic_analytic.tif", "_udm2.tif")
        only_water_check = mask_classes == ["water_check"]

        # Load and scale image
        image = rxr.open_rasterio(image_fn, masked=False)
        image_scaled = xr.where(image == 0, np.nan, image / 1e4)

        # NDWI water check
        if "water_check" in mask_classes:
            green = image_scaled.isel(band=1)
            nir = image_scaled.isel(band=3)
            ndwi = (green - nir) / (green + nir)
            water_mask = ((ndwi > 0.3) & (green < 0.8))
            water_fraction = water_mask.sum().data / len(water_mask.data.ravel())
            if water_fraction > 0.99:
                print(f"Skipping {base}, more than 99% water")
                return None

        if only_water_check:
            # Just copy the original image over
            out_fn = os.path.join(out_dir, os.path.basename(image_fn))
            shutil.copy2(image_fn, out_fn)
            return out_fn

        # Otherwise, continue with masking using UDM2
        needs_udm2 = any(cls.lower() in class_to_band for cls in mask_classes if cls.lower() != "water_check")
        if needs_udm2 and not os.path.exists(udm2_fn):
            print(f"Skipping {base}: missing UDM2 file")
            return None

        # Initialize mask
        mask = xr.zeros_like(image.isel(band=0), dtype=bool)

        if needs_udm2:
            udm2 = rxr.open_rasterio(udm2_fn, masked=False)

            for cls in mask_classes:
                cls = cls.lower()
                if cls == "water_check":
                    continue
                if cls not in class_to_band:
                    raise ValueError(f"Unknown mask class: {cls}")
                b = class_to_band[cls]
                if b not in [0, 6, 7]:
                    mask |= (udm2.isel(band=b) == 1)

            if confidence_threshold is not None:
                conf = udm2.isel(band=class_to_band["confidence"])
                mask |= (conf < confidence_threshold)

            if mask_unusable:
                unusable = udm2.isel(band=class_to_band["unusable"])
                mask |= (unusable > 0)

        # Apply mask
        image_masked = xr.where(mask, np.nan, image_scaled)

        # Save result
        image_masked = (image_masked * 1e4).astype("uint16")
        image_masked = image_masked.rio.set_nodata(0)
        if image.rio.crs:
            image_masked = image_masked.rio.write_crs(image.rio.crs)
        image_masked = image_masked.transpose('band', 'y', 'x')
        out_path = os.path.join(out_dir, base.replace('.tif', '_masked.tif'))
        image_masked.rio.to_raster(out_path)

        return out_path

    # Process images in parallel
    processed_image_fns = p_map(process_image, image_list, num_cpus=multiprocessing.cpu_count())
    processed_image_fns = [x for x in processed_image_fns if x]

    # Copy cameras to out_dir
    if copy_cams:
        # determine RPC or TSAI file
        if len(glob(os.path.join(os.path.dirname(image_list[0]), '*RPC.TXT'))) > 0:
            print('Copying filtered RPC cameras to out_dir...')
            cam_fns = [glob(os.path.join(os.path.dirname(image), os.path.basename(image).split('_basic')[0] + '*RPC.TXT'))[0] 
                       for image in image_list]
        elif len(glob(os.path.join(os.path.dirname(image_list[0]), '*.tsai'))) > 0:
            print('Copying filtered TSAI cameras to out_dir...')
            cam_fns = [glob(os.path.join(os.path.dirname(image), os.path.basename(image).split('_basic')[0]) + '*.tsai')[0] 
                       for image in image_list]
        for cam_fn in tqdm(cam_fns):
            cam_out_fn = os.path.join(out_dir, os.path.basename(cam_fn))
            shutil.copy2(cam_fn, cam_out_fn)

    return processed_image_fns


def generate_frame_cameras(img_list: List[str] = None, 
                           dem_fn: str = None, 
                           product_level: str = 'l1b', 
                           out_folder: str = None) -> str:
    """
    Generate ASP camera models and GCPs for a list of images using cam_gen.

    Parameters
    ----------
    img_list: list
        list of image file names
    dem_fn: str
        file name of the reference DEM
    product_level: str
        product level of the images, either 'l1b' or 'l1a'
    out_folder: str
        folder where output camera models and GCPs will be saved
    
    Returns
    ----------
    cam_gen_log: str
        file name of the cam_gen log file, which contains information about the number of GCP
    """
    # Make output directory if it doesn't exist
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    
    # Read overlap data table
    cam_list = img_list

    # Define output camera and GCP files
    frames = [os.path.splitext(os.path.basename(x))[0] for x in img_list] # grab just the image identifier strings
    out_cam_list = [os.path.join(out_folder,'{}.tsai'.format(frame)) for frame in frames]
    out_gcp_list = [os.path.join(out_folder,'{}.gcp'.format(frame)) for frame in frames]

    # Define reference height value where DEM has no data
    ht_datum = np.nanmedian(xdem.DEM(dem_fn).data) 

    # Determine number of jobs and threads per job
    ncpu, threads_per_job = setup_parallel_jobs(total_jobs=len(img_list))
    
    # Set up cam_gen options
    cam_gen_opt = []
    cam_gen_opt.extend(['--focal-length', str(553846.153846)])
    cam_gen_opt.extend(['--optical-center', str(1280), str(540)])
    if product_level=='l1b':
        cam_gen_opt.extend(['--pixel-pitch', str(0.8)])
    else:
        cam_gen_opt.extend(['--pixel-pitch', str(1.0)])
    cam_gen_opt.extend(['--height-above-datum',str(ht_datum)])
    cam_gen_opt.extend(['--gcp-std', str(1)])
    cam_gen_opt.extend(['--datum', 'WGS84'])
    cam_gen_opt.extend(['--reference-dem', dem_fn])
    cam_gen_opt.extend(['--refine-camera'])
    cam_gen_opt.extend(['--threads', str(threads_per_job)])

    # Construct jobs list
    jobs_list = []
    for img, cam, out_fn, out_gcp in zip(img_list, cam_list, out_cam_list, out_gcp_list):
        job = cam_gen_opt.copy()
        job.extend(['--input-camera', cam])
        job.extend(['-o', out_fn])
        job.extend(['--gcp-file', out_gcp])
        job.extend([img])
        jobs_list.append(job)
    print(jobs_list[0])

    # Run cam_gen in parallel
    cam_gen_logs = p_map(run_cmd, ['cam_gen']*len(jobs_list), jobs_list, num_cpus=ncpu)

    # Save compiled cam_gen log
    cam_gen_log = os.path.join(out_folder, 'cam_gen.log')
    print("Saving cam_gen log at {}".format(cam_gen_log))
    with open(cam_gen_log,'w') as f:
        for log in cam_gen_logs:
            f.write(log + '\n')
    
    # Remove basename from GCP file names
    # ASP's cam_gen writes full path for images in the GCP files. This does not play well during bundle adjustment.
    # The function returns a consolidated gcp file with all images paths only containing basenames so that bundle adjustment can roll along
    # See ASP's gcp logic here: https://stereopipeline.readthedocs.io/en/latest/tools/bundle_adjust.html#bagcp
    print("Writing gcp with basename removed")  
    for out_gcp in out_gcp_list:
        df_list = [pd.read_csv(x,header=None,delimiter=r"\s+") for x in out_gcp_list]
    gcp_df = pd.concat(df_list, ignore_index=True)
    def clean_img_in_gcp(row):
        return os.path.basename(row[7])
    gcp_df[7] = gcp_df.apply(clean_img_in_gcp, axis=1)
    gcp_df[0] = np.arange(len(gcp_df))
    print(f"Total number of GCPs found {len(gcp_df)}")
    gcp_df.to_csv(os.path.join(out_folder, 'clean_gcp.csv'), sep=' ', index=False, header=False)
    print(f"Cleaned gcp saved as CSV file at {os.path.join(out_folder,'clean_gcp*')}")

    return cam_gen_log


def find_matching_camera_file(image_fn: str = None, 
                              cam_folder: str = None) -> str:
    """
    Find camera file matching the image file's unique identifier.
    Parameters
    ----------
    image_fn: str
        file name of the image
    cam_folder: str
        folder containing camera files
    Returns
    ----------
    matched_fn: str
        file name of the matching camera file
    """
    # Get the identifying string from the image file
    # File naming convention SkySatScenes (https://developers.planet.com/docs/data/skysat/): 
    # <acquisition date>_<acquisition time>_<satellite_id><camera_id>_<frame_id>_<bandProduct>
    match = re.search(r"\d{8}_\d{6}_[a-zA-Z0-9]+_\d{4}", image_fn)
    if match:
        identifier = match.group(0)
    else:
        identifier = None
    if not identifier:
        raise ValueError(f"Could not extract identifier from image: {image_fn}")

    # Find matching camera file(s)
    cam_list = (glob(os.path.join(cam_folder, "*.tsai")) 
                + glob(os.path.join(cam_folder, '*.TXT')))
    matched_fn = [f for f in cam_list if identifier in f]
    # ideally, only one match, otherwise it's ambiguous
    if len(matched_fn) == 0:
        raise ValueError(f"No matching camera file found for image: {image_fn}")
    elif len(matched_fn) > 1:
        raise ValueError(f"Multiple matching camera files found for image: {image_fn}")
    else:
        matched_fn = matched_fn[0]

    return matched_fn


def run_mapproject(img_list: List[str] = None, 
                   cam_folder: str = None, 
                   out_folder: str = None, 
                   dem: str = 'WGS84', 
                   t_res: float = None, 
                   t_crs: str = None, 
                   session: str = None, 
                   orthomosaic: bool = False) -> None:
    """
    Mapproject images onto a reference DEM and optionally, create median mosaic of mapprojected images. 

    Parameters
    ----------
    img_list: list of str
        list of image file names
    cam_folder: str
        folder containing camera files
    out_folder: str
        path to the folder where mapprojected images and cameras will be saved
    dem: str (default="WGS84")
        reference DEM used for mapprojection. If None, will use the WGS84 datum.
    t_res: float | str
        target spatial resolution of the mapprojected images (meters)
    t_crs: str
        target coordinate reference system of the mapprojected images (e.g., "EPSG:4326")
    session: str
        ASP session type (e.g., "pinhole"). Usually, ASP can determine this automatically based on the inputs. 
    orthomosaic: bool
        whether to create a median mosaic of the mapprojected images, along with count, NMAD, weighted average, 
        and mosaics from different stereo views

    Returns
    ----------
    None
    """
    # Make output directory if it doesn't exist
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    # Set up image specific arguments: output prefixes and cameras
    frames_list = [os.path.splitext(os.path.basename(img))[0] for img in img_list]
    out_list = [os.path.join(out_folder, img + '_map.tif') for img in frames_list]
    cam_list = [find_matching_camera_file(img, cam_folder) for img in img_list]

    # Determine number of CPUs for parallelization and threads per job
    ncpu, threads_per_job = setup_parallel_jobs(total_jobs=len(img_list))
    
    # Set up mapproject arguments
    map_opts = []
    map_opts.extend(['--threads', str(threads_per_job)])
    map_opts.extend(['--tr', str(t_res)])
    map_opts.extend(['--t_srs', str(t_crs)])
    # limit to integer values, with 0 as no-data
    map_opts.extend(['--nodata-value', '0'])
    map_opts.extend(['--ot','UInt16'])
    if session:
        map_opts.extend(['--session', session])

    # Combine image-specific arguments and options into a list of jobs
    jobs_list = []
    for img, cam, out in zip(img_list, cam_list, out_list):
        job = map_opts + [dem, img, cam, out]
        jobs_list.append(job)

    print('\nMapproject arguments for first job:')
    print(jobs_list[0])

    # Run mapproject in parallel
    print("\nMapping images onto DEM")
    ortho_logs = p_map(run_cmd, ['mapproject']*len(jobs_list), jobs_list, num_cpus=ncpu)

    # Remove remaining intermediary tiling folders
    tile_folders = glob(os.path.join(out_folder, '*_tiles'))
    for folder in tile_folders:
        shutil.rmtree(folder)
    
    # Save compiled ortho log
    ortho_log = os.path.join(out_folder, 'ortho.log')
    print("Saving compiled orthorectification log at {}".format(ortho_log))
    with open(ortho_log,'w') as f:
        for log in ortho_logs:
            f.write(log + '\n')
    
    # Create orthomosaic
    if orthomosaic:
        print("Creating orthomosaic")
        # get unique image datetimes
        dt_list = list(set(sorted(['_'.join(os.path.basename(im).split('_')[0:2]) for im in out_list])))

        # define mosaic prefix containing timestamps of inputs
        mos_prefix = '__'.join(dt_list)

        # define output filenames
        mosaic_fn = os.path.join(out_folder, '{}_orthomosaic.tif'.format(mos_prefix))

        # Set up mosaic arguments
        mos_args = ['--threads', str(multiprocessing.cpu_count()),
                    '-ot', 'UInt16',
                    '--tr', str(t_res),
                    '--t_srs', str(t_crs),
                    '--output-nodata-value', '0',
                    '--no-big-tiff']
        mos_args.extend(out_list)
        mos_args.extend(['-o', mosaic_fn])

        # Run image mosaic
        run_cmd('image_mosaic', mos_args)


def identify_overlapping_image_pairs(img_list: List[str] = None, 
                                     overlap_perc: float = 10, 
                                     bh_ratio_range: tuple = None, 
                                     utm_epsg: str = None, 
                                     out_folder: str = None) -> List[tuple]:
    """
    Find overlapping image pairs from a list of georeferenced images.

    Parameters
    ----------
    img_list: list 
        List of image file paths.
    overlap_perc: float 
        Minimum overlap percent (0-100) to include pair.
    bh_ratio_range: tuple
        Minimum and maximum baseline to height ratio (B/H) to include pair.
        If None, will not filter based on B/H ratio.
    utm_epsg: str
        EPSG code for the optimal UTM zone, e.g. "EPSG:32611"
    out_folder: str 
        Folder where output text file will be saved.

    Returns
    ----------
    overlapping_pairs (list of tuple): 
        List of (image1, image2) pairs.
    """
    # Get image bounds polygons
    def get_image_polygon(img_fn):
        # get lat lon bounds
        min_lon, min_lat, max_lon, max_lat = rpc_image_latlon_bounds(img_fn)
        # convert to polygon
        bounds_poly = Polygon([[min_lon, min_lat], [max_lon, min_lat],
                                [max_lon, max_lat], [min_lon, max_lat],
                                [min_lon, min_lat]])
        # reproject to the optimal UTM zone
        bounds_gdf = gpd.GeoDataFrame(index=[0], geometry=[bounds_poly], crs="EPSG:4326")
        bounds_gdf = bounds_gdf.to_crs(utm_epsg)

        return bounds_gdf.geometry[0]
    polygons = {img: get_image_polygon(img) for img in img_list}
    
    # Compare all unique pairs
    overlapping_pairs = []
    overlap_ratios = []
    bh_ratios = []
    for img1, img2 in itertools.combinations(img_list, 2):
        poly1 = polygons[img1]
        poly2 = polygons[img2]

        intersection = poly1.intersection(poly2)
        if not intersection.is_empty:
            area1 = poly1.area
            area2 = poly2.area
            overlap_percent = intersection.area / min(area1, area2) * 100
            if overlap_percent >= overlap_perc:
                bh_ratio = calculate_baseline_to_height_ratio(img1, img2, utm_epsg)
                # check for B/H ratio thresholds if specified
                if bh_ratio_range:
                    if (bh_ratio < bh_ratio_range[0]) | (bh_ratio > bh_ratio_range[1]):
                        continue
                bh_ratios += [bh_ratio]
                overlapping_pairs += [(img1, img2)]
                overlap_ratios += [overlap_percent]
                    
    # Write to file
    out_fn = os.path.join(out_folder, 'overlapping_image_pairs.txt')
    # add the header
    with open(out_fn, 'w') as f:
        f.write(f"img1\timg2\tidentifier_text\toverlap_percent\tbh_ratio\n")
    # iterate over pairs
    for i, (img1, img2) in enumerate(overlapping_pairs):
        date1, time1 = os.path.basename(img1).split('_')[0:2]
        date2, time2 = os.path.basename(img1).split('_')[0:2]
        identifier_text = date1 + '_' + time1 + '__' + date2 + '_' + time2
        with open(out_fn, 'a') as f:
            f.write(f"{img1}\t{img2}\t{identifier_text}\t{overlap_ratios[i]}\t{bh_ratios[i]}\n")

    print('Overlapping stereo pairs saved to file:', out_fn)

    return overlapping_pairs


def get_stereo_opts(session: str = None, 
                    threads: int = None, 
                    texture: str = 'normal', 
                    correlator_mode: bool = False, 
                    unalign_disparity: bool = False) -> List[str]:
    """
    Get the stereo options for the ASP parallel_stereo command.

    Parameters
    ----------
    session: str (default=None)
        The session type to use for stereo matching. Options include 'rpc', 'pinhole', etc. 
        ASP can often figure this out automatically. 
    threads: int (default=None)
        Number of threads to use for parallel processing. If None, will automatically determine based on CPU count.
    texture: str (default='normal')
        This is used for determining the correlation and refinement kernel. Options = "low", "normal".
    correlator_mode: bool (default=False)
        Whether to run in correlator mode (no point cloud generation). This is useful for debugging or testing purposes.
    unalign_disparity: bool (default=False)
        Whether to generate disparity maps without alignment. This can be used for debugging or testing purposes.
    
    Returns
    ----------
    stereo_opt: list
        A list of command line options for the ASP parallel_stereo command.
    """
    stereo_opts = []
    # session_args
    if session:
        stereo_opts.extend(['-t', session])
    stereo_opts.extend(['--threads-multiprocess', str(threads)])
    stereo_opts.extend(['--threads-singleprocess', str(threads)])
    # stereo_pprc args : This is for preprocessing (adjusting image dynamic range, 
    # alignment using ip matches, etc.)
    stereo_opts.extend(['--individually-normalize'])
    stereo_opts.extend(['--ip-per-tile', '8000'])
    stereo_opts.extend(['--ip-num-ransac-iterations','2000'])
    stereo_opts.extend(['--force-reuse-match-files'])
    stereo_opts.extend(['--skip-rough-homography'])
    stereo_opts.extend(['--alignment-method', 'None'])
    # mask out completely feature less area using a std filter, to avoid gross MGM errors
    # this is experimental and needs more testing
    stereo_opts.extend(['--stddev-mask-thresh', '0.5'])
    stereo_opts.extend(['--stddev-mask-kernel', '-1'])
    # stereo_corr_args
    stereo_opts.extend(['--stereo-algorithm', 'asp_mgm'])
    # correlation kernel size depends on the texture
    if texture=='low':
        stereo_opts.extend(['--corr-kernel', '9', '9'])
    elif texture=='normal':
        stereo_opts.extend(['--corr-kernel', '7', '7'])
    stereo_opts.extend(['--corr-tile-size', '1024'])
    stereo_opts.extend(['--cost-mode', '4'])
    stereo_opts.extend(['--corr-max-levels', '5'])
    # stereo_rfne_args:
    stereo_opts.extend(['--subpixel-mode', '9'])
    if texture=='low':
        stereo_opts.extend(['--subpixel-kernel', '21', '21'])
    elif texture=='normal':
        stereo_opts.extend(['--subpixel-kernel', '15', '15'])
    stereo_opts.extend(['--xcorr-threshold', '2'])
    stereo_opts.extend(['--num-matches-from-disparity', '10000'])
    # limit to just correlation (no point cloud generation)
    if correlator_mode:
        stereo_opts.extend(['--correlator-mode'])
    # get the disparity map without any alignment
    if unalign_disparity:
        stereo_opts.extend(['--unalign-disparity'])
    
    return stereo_opts


def run_stereo(stereo_pairs_fn: str = None, 
               cam_folder: str = None, 
               out_folder: str = None, 
               session: str = None,
               texture: str = 'normal', 
               correlator_mode: bool = False) -> None:
    """
    Execute stereo matching for SkySat images using the ASP parallel_stereo command.

    Parameters
    ----------
    stereo_pairs_fn: str (default=None)
        Path to the text file containing overlapping image pairs.
    cam_folder: str or Path (default=None)
        Path to the folder containing camera files. Required if using 'pinhole' session.
    out_folder: str or Path
        Path to the folder where the output stereo results will be saved.
    session: str (default=None)
        The session type to use for stereo matching. Options include 'rpc', 'pinhole', etc. ASP can often figure this out automatically.
    texture: str (default='normal')
        How much relative texture there is in your images. This is used for determining the correlation and refinement kernel. 
        Options = "low", "normal". For example, a flat, snowy landscape likely has "low" texture. 
    correlator_mode: bool (default=False)
        Whether to run in correlator mode (no point cloud generation).
    
    Returns
    ----------
    None
    """
    # Check if output folder exists
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    # Load the stereo pairs
    stereo_pairs_df = pd.read_csv(stereo_pairs_fn, delimiter='\t', header=0)

    # Determine number of CPUs for parallelization and threads per job
    ncpu, threads_per_job = setup_parallel_jobs(total_jobs=len(stereo_pairs_df))
    
    # Define stereo arguments
    stereo_opts = get_stereo_opts(session=session, threads=threads_per_job, texture=texture, correlator_mode=correlator_mode)
    
    # Create jobs list for each stereo pair
    job_list = []
    for _, row in stereo_pairs_df.iterrows():
        # Determine output folder for stereo job
        IMG1 = os.path.splitext(os.path.basename(row['img1']))[0]
        IMG2 = os.path.splitext(os.path.basename(row['img2']))[0]
        out_prefix = os.path.join(out_folder, row['identifier_text'], IMG1 + '__' + IMG2, 'run')  
        # Construct the stereo job
        if cam_folder:
            # Use the camera files if provided
            cam1 = find_matching_camera_file(row['img1'], cam_folder)
            cam2 = find_matching_camera_file(row['img2'], cam_folder)
            job = stereo_opts + [row['img1'], cam1, row['img2'], cam2, out_prefix]
        else:
            # Otherwise, use the images directly
            stereo_args = [row['img1'], row['img2'], out_prefix]
            job = stereo_opts + stereo_args
        # Add job to list of jobs
        job_list.append(job)
    
    # Run the jobs in parallel
    print('stereo arguments for first job:')
    print(job_list[0])
    stereo_logs = p_map(run_cmd, ['parallel_stereo']*len(job_list), job_list, num_cpus=ncpu)

    # Save the consolidated log
    stereo_log_fn = os.path.join(out_folder, 'stereo_log.log')
    with open(stereo_log_fn, 'w') as f:
        for log in stereo_logs:
            f.write(log + '\n')
    print("Consolidated stereo log saved at {}".format(stereo_log_fn))


def coregister_dems_xdem(dem_fn: str = None, 
                         refdem_fn: str = None, 
                         ortho_fn: str = None, 
                         out_dir: str = None) -> tuple:
    """
    Coregister a DEM to a reference DEM using xdem. Optionally, apply the same translation to an orthomosaic.

    Parameters
    ----------
    dem_fn: str
        file name of the DEM to be coregistered
    refdem_fn: str
        file name of the reference DEM
    ortho_fn: str
        file name of the orthomosaic to be coregistered
    out_dir: str
        path to the folder where outputs will be saved

    Returns
    ----------
    dem_coreg_fn: str
        file name of the coregistered DEM
    ortho_coreg_fn: str
        file name of the coregistered orthomosaic

    """
    # Determine optimal CRS
    dem = gu.Raster(dem_fn).reproject(crs="EPSG:4326")
    dem_cen_lon = (dem.bounds.right + dem.bounds.left) / 2
    dem_cen_lat = (dem.bounds.bottom + dem.bounds.top) / 2
    out_crs = convert_wgs_to_utm(dem_cen_lon, dem_cen_lat)
    
    # Reproject DEMs and save to file
    def reproject_and_save_raster(raster_fn, out_crs):
        raster = gu.Raster(raster_fn)
        if raster.crs != out_crs:
            raster_reproj_fn = os.path.splitext(raster_fn)[0] + f'_{out_crs.replace(':','')}.tif'
            raster = raster.reproject(crs=out_crs)
            raster.save(raster_reproj_fn)
            print('Reprojected raster saved to file:', raster_reproj_fn)
            return raster_reproj_fn
        else:
            return raster_fn
    dem_reproj_fn = reproject_and_save_raster(dem_fn, out_crs)
    refdem_reproj_fn = reproject_and_save_raster(refdem_fn, out_crs)        
    
    # Load DEMs
    refdem = xdem.DEM(refdem_reproj_fn)
    dem = xdem.DEM(dem_reproj_fn)
    dem_reproj = dem.reproject(refdem) # coregister with dem upscaled to refdem

    # Coregister
    print('\nCoregistering DEMs using the Iterative Closest Point method...') 
    # then Nuth and Kaab methods...')
    # coreg = xdem.coreg.CoregPipeline([xdem.coreg.ICP(), 
    #                                   xdem.coreg.NuthKaab(subsample=1)]).fit(refdem, dem_reproj)
    coreg = xdem.coreg.ICP().fit(refdem, dem_reproj)
    coreg_matrix = coreg.to_matrix()
    print('Coregistration matrix:')
    print(coreg_matrix)
    dem_coreg = coreg.apply(dem)

    # Save coregistered DEM to file
    dem_coreg_fn = os.path.join(out_dir,
                                os.path.splitext(os.path.basename(dem_reproj_fn))[0] + '_coregistered.tif')
    dem_coreg.save(dem_coreg_fn)
    print('Coregistered DEM saved to file:', dem_coreg_fn)

    # Apply translation to orthomosaic
    if ortho_fn:
        print('\nApplying the coregistration translation to orthomosaic...')
        # subset just the conregistration matrix to x-shift and y-shift components 
        # (no z-shift needed for the orthomosaic)
        from affine import Affine
        x_shift = coreg_matrix[0, 3]
        y_shift = coreg_matrix[1, 3]
        # reproject orthomosaic
        ortho_reproj_fn = reproject_and_save_raster(ortho_fn, out_crs)
        # load the reprojected orthomosaic
        ortho = rxr.open_rasterio(ortho_reproj_fn)
        crs = ortho.rio.crs # save the CRS for writing later
        # update the coordinates
        new_x_coords = ortho['x'].values + x_shift
        new_y_coords = ortho['y'].values + y_shift
        ortho = ortho.assign_coords(x=new_x_coords, y=new_y_coords)
        # update the transform
        original_transform = ortho.rio.transform()
        new_transform = original_transform * Affine.translation(x_shift, y_shift)
        ortho.rio.write_transform(new_transform, inplace=True)
        ortho.rio.write_crs(crs, inplace=True)
        # save to file
        ortho_coreg_fn = os.path.join(out_dir,
                                      os.path.splitext(os.path.basename(ortho_reproj_fn))[0] + '_coregistered.tif')
        ortho.rio.to_raster(ortho_coreg_fn)
        print('Coregistered orthomosaic saved to file:', ortho_coreg_fn)
    else:
        ortho_coreg_fn = None

    return dem_coreg_fn, ortho_coreg_fn



# -------------------------------


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
            multispec_fns = sorted(glob(os.path.join(multispec_path, '*_SR.tif')))
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


def prep_stereo_df(overlap_pkl, true_stereo=True, cross_track=False):
    """
    Prepare stereo jobs dataframe input pickle file containing overlapping images with percentages.

    Parameters
    ----------
    overlap_pkl: str
        Path to pickle file containing overlapping images produced from skysat_overlap_parallel.py
    true_stereo: bool
        True means output dataframe has only pairs formed by scenes from different views
    cross_track: bool
        True means output dataframe has only pairs formed by scenes from the same date and satellite

    Returns
    ----------
    df: Pandas Dataframe
        dataframe containing list of plausible overlapping stereo pairs
    """
    # Load overlap dataframe
    if type(overlap_pkl)==str:
        df = pd.read_pickle(overlap_pkl)
    elif type(overlap_pkl)==pd.DataFrame:
        df = overlap_pkl
    else:
        raise ValueError("Input overlap_pkl should be a file name or pandas dataframe")
    
    # Check date: if dates not equal, drop
    # Check time: if time equal, drop
    # Check satellite: if unequal, drop
    # Check overlap percent
    df.loc[:, 'sat1'] = [os.path.basename(x).split('_', 15)[2].split('d', 15)[0] for x in df.img1.values]
    df.loc[:, 'sat2'] = [os.path.basename(x).split('_', 15)[2].split('d', 15)[0] for x in df.img2.values]
    df.loc[:, 'date1'] = [os.path.basename(x).split('_', 15)[0] for x in df.img1.values]
    df.loc[:, 'date2'] = [os.path.basename(x).split('_', 15)[0] for x in df.img2.values]
    df.loc[:, 'time1'] = [os.path.basename(x).split('_', 15)[1] for x in df.img1.values]
    df.loc[:, 'time2'] = [os.path.basename(x).split('_', 15)[1] for x in df.img2.values]
    if true_stereo:
        # returned df has only those pairs which form true stereo
        df = df[df['time1'] != df['time2']]
        if not cross_track:
            df = df[df['date1'] == df['date2']]
            df = df[df['sat1'] == df['sat2']]
    # filter to overlap percentage of around 5%
    df['overlap_perc'] = df['overlap_perc'] * 100
    df = df[(df['overlap_perc'] > 5)]
    df['identifier_text'] = df['date1'] + '_' + df['time1'] + '__' + df['date2'] + '_' + df['time2']
    df.reset_index(drop=True, inplace=True)

    return df




def get_bundle_adjust_opts(ba_prefix, session='nadirpinhole', ba_dem=False, dem=None, dem_uncertainty=10,
                           fixed_camera_indices=None, cam_weight=0, num_iter=100, num_pass=2, 
                           threads=multiprocessing.cpu_count()):
    """
    Get the bundle adjustment options for the ASP bundle_adjust command.

    Parameters
    ----------
    ba_prefix: str or Path
        Prefix for the bundle adjustment output files.
    session: str (default='nadirpinhole')
        The session type to use for bundle adjustment. Options include 'nadirpinhole', etc.
    ba_dem: bool (default=False)
        Whether to use a reference DEM for bundle adjustment. If True, will include DEM options.
    dem: str or Path (default=None)
        Path to the reference DEM file. Required if ba_dem is True.
    dem_uncertainty: float (default=10)
        Uncertainty parameter associated with the DEM (~meters).
    fixed_camera_indices: str (default=None)
        Indices of cameras to be fixed during bundle adjustment. If None, all cameras will be adjusted.
    cam_weight: float (default=0)
        Weight of the initial camera positions.
    num_iter: int (default=100)
        Number of bundle adjustment iterations.
    num_pass: int (default=2)
        Number of bundle adjustment passes.
    threads: int (default=multiprocessing.cpu_count())
        Number of threads to use for parallel processing.
    
    Returns
    ----------
    ba_opts: list
        A list of command line options for the ASP bundle_adjust command.
    """
    ba_opts = []
    ba_opts.extend(['--threads', str(threads)])
    ba_opts.extend(['-t', session])
    ba_opts.extend(['-o', ba_prefix])
    # keypoint-finding args
    # relax triangulation error based filters to account for initial camera errors
    ba_opts.extend(['--min-matches', '4'])
    ba_opts.extend(['--disable-tri-ip-filter'])
    ba_opts.extend(['--force-reuse-match-files'])
    ba_opts.extend(['--ip-per-tile', '4000'])
    ba_opts.extend(['--ip-inlier-factor', '0.2'])
    ba_opts.extend(['--ip-num-ransac-iterations', '1000'])
    ba_opts.extend(['--skip-rough-homography'])
    ba_opts.extend(['--min-triangulation-angle', '0.0001'])
    # save control network created from match points
    ba_opts.extend(['--save-cnet-as-csv'])
    # individually normalize images to properly stretch constraint 
    # helpful in keypoint detection
    ba_opts.extend(['--individually-normalize'])
    # this generally assigns weight to penalize movement of camera extrinsics
    ba_opts.extend(['--camera-position-weight', str(cam_weight)])
    # output updated cameras, not just the adjustments (only available for pinhole cameras)
    if session == 'nadirpinhole':
        ba_opts.extend(['--inline-adjustments'])
    # specify number of passes and maximum iterations per pass
    ba_opts.extend(['--num-iterations', str(num_iter)])
    ba_opts.extend(['--num-passes', str(num_pass)])
    # add reference DEM if using
    if ba_dem:
        ba_opts.extend(['--heights-from-dem', dem])
        ba_opts.extend(['--heights-from-dem-uncertainty', str(dem_uncertainty)])
    if fixed_camera_indices:
        ba_opts.extend(['--fixed-camera-indices', fixed_camera_indices])
    
    return ba_opts


def bundle_adjustment(img_folder=None, ba_prefix=None, cam_folder=None, dem=None, ba_dem=False, ba_dem_uncertainty=10, overlap_pkl=None, 
                      session='nadirpinhole', texture='normal', cam_weight=0, num_iter=100, num_pass=2):
    """
    Run stereo correlation and bundle adjustment on a set of images. Initial testing showed that select image pairs had
    fewer features matches compared to others, leading to insufficient camera extrinsics adjustment when adjusting many images at once. 
    Thus, this program adjusts one image pair at a time, starting with the most overlapping pair, fixing that pair, adjusting the next 
    most-overlapping image, and iterating until all images are adjusted. For each pair, stereo is run in correlator mode, bundle adjustment 
    is run, then the next run will use the adjusted cameras from the previous iteration.

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
    # Create output directory if it doesn't exist
    out_dir = os.path.dirname(ba_prefix)
    os.makedirs(out_dir, exist_ok=True)

    # Get list of image files
    img_list = sorted(glob(os.path.join(img_folder, '*.tif')) +
                      glob(os.path.join(img_folder, '*.tiff')))
    # resolve symlink if only one image found and it is a symlink
    if len(img_list) == 1 and os.path.islink(img_list[0]):
        img_list = [os.readlink(x) for x in img_list]     

    # Load image pairs overlap data table
    overlap = pd.read_pickle(overlap_pkl)
    overlap = overlap.loc[(overlap['img1'].isin(img_list)) & (overlap['img2'].isin(img_list))]
    overlap = overlap.sort_values(by='overlap_perc', ascending=False).reset_index(drop=True)

    # Determine camera file extension
    cam_ext = '.tsai' if glob(os.path.join(cam_folder, '*.tsai')) else '.TXT'

    # Map camera files to image pairs
    overlap['cam1'] = overlap['img1'].apply(lambda x: find_matching_camera_files(x, cam_folder))
    overlap['cam2'] = overlap['img2'].apply(lambda x: find_matching_camera_files(x, cam_folder))
    overlap.dropna(inplace=True)

    # Update image list to only those with valid cameras
    img_list = sorted(pd.unique(overlap[['img1', 'img2']].values.ravel()))
    if len(img_list) < 2:
        raise Exception('Less than two images with cameras found. Check file paths and cameras')
    print(f"{len(img_list)} images found with cameras for bundle adjustment.")

    # Define output names for adjusted cameras
    overlap['cam1_out'] = ba_prefix + '-' + overlap['cam1'].map(os.path.basename)
    overlap['cam2_out'] = ba_prefix + '-' + overlap['cam2'].map(os.path.basename)

    # Initialize with first image pair
    img1, img2 = overlap.loc[0, ['img1', 'img2']]
    cam1, cam2 = overlap.loc[0, ['cam1', 'cam2']]
    img_adjusted_list = []
    img_tba_list = img_list.copy() 
    ba_img_list = [img1, img2]
    ba_cam_list = [cam1, cam2]
    fixed_indices = None

    i = 0
    pbar = tqdm(total=len(img_list))
    while img_tba_list:
        if i == 0:
            print(f'\nAdjusting {img1} and {img2}')
            out_dir_round = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(img1))[0]}__{os.path.splitext(os.path.basename(img2))[0]}")
        else:
            print(f"\nAdjusting {img_tba}")
            out_dir_round = os.path.join(out_dir, os.path.splitext(os.path.basename(img_tba))[0])
        os.makedirs(out_dir_round, exist_ok=True)

        # Get stereo pairs to adjust in this round
        if i == 0:
            overlap_round = overlap[((overlap['img1'] == img1) & (overlap['img2'] == img2)) |
                                     ((overlap['img1'] == img2) & (overlap['img2'] == img1))]
        else:
            overlap_round = overlap[((overlap['img1'] == img_tba) & (overlap['img2'].isin(img_adjusted_list))) |
                                     ((overlap['img2'] == img_tba) & (overlap['img1'].isin(img_adjusted_list)))]
        print('Number of stereo pairs in this round:', len(overlap_round))

        # Run stereo correlation
        print('Running stereo correlation')
        execute_skysat_stereo(overlap_pkl=overlap_round, cam_folder=cam_folder, out_folder=out_dir_round,
                            session=session, dem=dem, texture=texture, cross_track=True, correlator_mode=True)

        # Move match files to round directory
        match_files = glob(os.path.join(out_dir_round, '*', '*', '*.match'))
        print(f'Moved {len(match_files)} match files to round directory')
        for match_file in match_files:
            shutil.move(match_file, os.path.join(out_dir_round, os.path.basename(match_file).replace('-disp','')))

        # Set up bundle adjustment parameters
        ba_prefix_round = os.path.join(out_dir_round, 'run')
        ba_opts = get_bundle_adjust_opts(ba_prefix_round, session=session, ba_dem=ba_dem, dem=dem,
                                         dem_uncertainty=ba_dem_uncertainty, fixed_camera_indices=fixed_indices,
                                         cam_weight=cam_weight, num_iter=num_iter, num_pass=num_pass)
        ba_args = ba_opts + ba_img_list + ba_cam_list
        print('Running bundle adjust')
        run_cmd('parallel_bundle_adjust', ba_args)

        # Move output cameras to output directory for reuse in future rounds
        out_cams = glob(ba_prefix_round + '*' + cam_ext)
        j = 0
        for cam in out_cams:
            if ba_prefix_round + '-run-' not in cam:
                shutil.move(cam, os.path.join(out_dir, os.path.basename(cam)))
                j += 1
        print(f'Moved {j} adjusted camera(s) to output directory for future rounds')

        # Update list of adjusted images
        if i == 0:
            img_adjusted_list.extend([img1, img2])
            pbar.update(2)
        else:
            img_adjusted_list.append(img_tba)
            pbar.update(1)

        # Determine images left to be adjusted
        img_tba_list = [x for x in img_list if x not in img_adjusted_list]

        if img_tba_list:
            # Select most overlapping unadjusted image
            overlap_remaining = overlap[((~overlap['img1'].isin(img_adjusted_list)) & (overlap['img2'].isin(img_adjusted_list))) |
                                        ((overlap['img1'].isin(img_adjusted_list)) & (~overlap['img2'].isin(img_adjusted_list)))]
            overlap_remaining = overlap_remaining.sort_values(by='overlap_perc', ascending=False).reset_index(drop=True)

            if overlap_remaining.iloc[0]['img1'] in img_adjusted_list:
                img_tba, cam_tba = overlap_remaining.iloc[0][['img2', 'cam2']]
            else:
                img_tba, cam_tba = overlap_remaining.iloc[0][['img1', 'cam1']]

            # Set up images and cameras for next round
            ba_img_list = [img_tba] + [x for x in img_adjusted_list if (overlap_remaining['img1'] == x).any() or (overlap_remaining['img2'] == x).any()]
            ba_cam_list = [cam_tba] + [find_matching_camera_files(x, os.path.abspath(ba_prefix)) for x in ba_img_list[1:]]
            fixed_indices = ' '.join(list(np.arange(1, len(ba_img_list)).astype(str)))

        i += 1

    pbar.close()
    print('\nBundle adjust runs complete!')

    # Check how many processes converged in the log files
    log_fns = sorted(glob(os.path.join(out_dir, '20*', '*log*.txt')))
    nconv = 0
    for log_fn in log_fns:
        with open(log_fn, 'r') as f:
            log = f.read()
        if 'CONVERGENCE' in log:
            nconv += 1
    print(f'Number of bundle adjust runs that converged = {nconv} / {len(img_list)-1}')


def gridding_wrapper(pc_list,tr,tsrs=None):
    """
    Rasterize a list of point clouds using ASP's point2dem command. 

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
    # print(p2dem_log)


def mosaic_dems(dem_list, out_folder, tr=2, tsrs='EPSG:4326', tile_size=None, 
                threads=multiprocessing.cpu_count(), stats_list=['median', 'count', 'nmad']):
    """
    Mosaic a list of DEMs using ASP's dem_mosaic tool for multiple statistics.

    Parameters
    ----------
    dem_list : list of str or Path
        List of DEM filenames to mosaic.
    out_folder : str or Path
        Folder where mosaics and logs will be saved.
    tr : float or int, optional
        Target resolution of output mosaics (default: 2).
    tsrs : str, optional
        Target spatial reference system (default: 'EPSG:4326').
    tile_size : int, optional
        Tile size for memory-efficient mosaicking.
    stats_list : list of str, optional
        Statistics to use for mosaicking (default: ['median', 'count', 'nmad']).
    
    Returns
    ----------
    None
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    
    # Define base dem_mosaic options
    opts = []
    opts.extend(['--threads', str(threads)])
    if tr:
        opts.extend(['--tr', str(tr)])
    if tsrs:
        opts.extend(['--t_srs', tsrs])
    
    # Iterate over statistics
    for stat in tqdm(stats_list):
        # define output file name
        out_fn = os.path.join(out_folder,  f'dem_{stat}_mos.tif')
        print(f"\nCreating {stat} mosaic")

        # split into tiles if defined
        if tile_size:
            opts.extend(['--tile-size', str(tile_size)])
            temp_dir = os.path.join(out_folder, f'dem_{stat}_tiles')
            opts += ['-o', os.path.join(temp_dir, 'run')]
            out_tile_log = run_cmd('dem_mosaic', list(map(str, dem_list)) + opts)
            tile_files = sorted(glob(str(temp_dir / 'run-*.tif')))
            print(f"Found {len(tile_files)} tile(s)")
            final_log = run_cmd('dem_mosaic', tile_files + ['-o', str(out_fn)])
            shutil.rmtree(temp_dir)
            out_log = out_tile_log + final_log
        else:
            opts.extend(['-o', str(out_fn)])
            out_log = run_cmd('dem_mosaic', list(map(str, dem_list)) + opts)

        # save log
        log_fn = os.path.join(out_folder, f'dem_{stat}_mos.log')
        with open(log_fn, 'w') as f:
            f.write(out_log)
        print(f"Saved log to {log_fn}")


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
        
    print('Aligning the DEMs in two rounds:'
          '\n1) ICP for translation and rotation'
          '\n2) Nuth and Kaab for subpixel translation\n')
    
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
    else:
        print('ICP-aligned DEM already exists in file, skipping.')
    # grid
    if os.path.exists(align_out_fn) and not os.path.exists(grid_out_fn):
        print('Gridding the aligned point cloud')
        grid_cmd = ['--tr', str(tr), align_out_fn]
        out = run_cmd('point2dem', grid_cmd)
        print(out)
    
    # Round 2: Nuth and Kaab
    print('Round 2: Align DEMs using Nuth and Kaab method')
    # define outputs
    dem_nk_out_fn = os.path.join(out_dir, 'run-nk-DEM.tif')
    nk_align_matrix_fn = os.path.join(out_dir, 'run-nk-transform.txt')
    # load inputs
    if not os.path.exists(dem_nk_out_fn):
        dem = xdem.DEM(grid_out_fn, load_data=True)
        refdem = xdem.DEM(refdem_fn, load_data=True).reproject(dem)
        # align
        nk = xdem.coreg.NuthKaab().fit(refdem, dem)
        dem_nk = nk.apply(dem)
        metadata = nk._meta
        # save output
        dem_nk.save(dem_nk_out_fn)
        print('Nuth and Kaab-aligned DEM saved to:', dem_nk_out_fn)   
        # calculate and create the translation matrix
        tx = metadata['offset_east_px'] * metadata['resolution']
        ty = metadata['offset_north_px'] * metadata['resolution']
        tz = metadata['vshift']
        transform_matrix = np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ])
        # save to a text file
        np.savetxt(nk_align_matrix_fn, transform_matrix, fmt="%.6f")
        # print matrix
        print("Transformation Matrix saved to file:", nk_align_matrix_fn)
        print(transform_matrix)

    return dem_nk_out_fn


def align_cameras(camera_list, transform_txt, outfolder, rpc=False, dem=None, img_list=None):
    """
    Align a list of pinhole .tsai cameras using a pc_align transform matrix.
    Write new aligned .tsai and (optionally) RPC models.

    Parameters
    ----------
    camera_list : list of str
        Paths to input pinhole .tsai camera files.
    transform_txt : str
        Path to pc_align 4x4 transformation matrix text file.
    outfolder : str
        Folder to save output aligned cameras and RPC models (if requested).
    rpc : bool
        Whether to compute RPC models after alignment.
    dem : str or None
        Path to DEM for RPC calculation.
    img_list : list of str or None
        List of image paths corresponding to each camera, required for RPC.
    
    Returns
    ----------
    None
    """
    # Check if DEM and image list are provided for RPC generation
    if rpc and (not dem or not img_list):
        raise ValueError("DEM and image list required for RPC generation.")

    # Read transform matrix
    with open(transform_txt, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
    matrix = [list(map(float, line.split())) for line in lines[:3]]
    rot_matrix = np.array([row[:3] for row in matrix])
    translation = np.array([row[3] for row in matrix])

    # Loop over each camera
    pbar = tqdm(total=len(camera_list))
    for i, tsai_path in enumerate(camera_list):
        # Read camera parameters
        with open(tsai_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]

        fu = float(lines[2].split('=')[1])
        fv = float(lines[3].split('=')[1])
        cu = float(lines[4].split('=')[1])
        cv = float(lines[5].split('=')[1])
        cam_cen = np.array([float(x) for x in lines[9].split('=')[1].split()])
        rot_flat = [float(x) for x in lines[10].split('=')[1].split()]
        cam_rot = np.array(rot_flat).reshape((3, 3))
        pitch = float(lines[11].split('=')[1])
        cam_name = os.path.basename(tsai_path)

        # Adjust using pc_align transformation
        new_cen = rot_matrix @ cam_cen + translation
        new_rot = rot_matrix @ cam_rot

        # Write new .tsai
        tsai_fn = os.path.join(outfolder, os.path.splitext(cam_name)[0] + '_adj_pc_align.tsai')
        tsai_txt = (
            f"VERSION_4\nPINHOLE\n"
            f"fu = {fu}\n"
            f"fv = {fv}\n"
            f"cu = {cu}\n"
            f"cv = {cv}\n"
            f"u_direction = 1 0 0\n"
            f"v_direction = 0 1 0\n"
            f"w_direction = 0 0 1\n"
            f"C = {new_cen[0]} {new_cen[1]} {new_cen[2]}\n"
            f"R = {' '.join(map(str, new_rot.flatten()))}\n"
            f"pitch = {pitch}\nNULL"
        )
        with open(tsai_fn + '.tsai', 'w') as f:
            f.write(tsai_txt)

        # Optionally write RPC
        if rpc:
            rpc_fn = os.path.splitext(tsai_fn)[0] + '_rpc_asp.xml'
            cam2rpc_args = [
                "--session", "pinhole",
                "--dem-file", dem,
                "--save-tif-image",
                "--threads", str(multiprocessing.cpu_count()),
                "--num-samples", "50",
                "--gsd", str(30),  # can parameterize
                img_list[i],
                rpc_fn
            ]
            run_cmd('cam2rpc', cam2rpc_args, check=True)

        pbar.update(1)
    pbar.close()

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
    
    p_map(align_cameras, input_camera_list, transform_list, outfolder, write, rpc, dem,
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
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    from matplotlib import colors

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


def dense_match_wrapper(stereo_dir, ba_dir):
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
    triplet_stereo_matches = sorted(glob(os.path.join(stereo_dir,'20*/*/run*-*disp*.match')))
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
    
    # Update overlap list if specified - DOESN'T WORK WITHOUT DENSE_MATCH_PKL 
    # if modify_overlap:
    #     orig_df = pd.read_pickle(overlap_pkl)
    #     dense_df = pd.read_pickle(dense_match_pkl)
    #     dense_img1 = list(dense_df.img1.values)
    #     dense_img2 = list(dense_df.img2.values)
    #     priority_list = list(zip(dense_img1,dense_img2))
    #     regular_img1 = [os.path.basename(x) for x in orig_df.img1.values]
    #     regular_img2 = [os.path.basename(x) for x in orig_df.img2.values]
    #     secondary_list = list(zip(regular_img1,regular_img2))
    #     # adapted from https://www.geeksforgeeks.org/python-extract-unique-tuples-from-list-order-irrespective/
    #     # note that I am using the more inefficient answer on purpose, because I want to use image pair order from the dense match overlap list
    #     total_list = priority_list + secondary_list
    #     final_overlap_set = set()
    #     temp = [final_overlap_set.add((a, b)) for (a, b) in total_list
    #             if (a, b) and (b, a) not in final_overlap_set]
    #     new_img1 = [os.path.join(img_fol,pair[0]) for pair in list(final_overlap_set)]
    #     new_img2 = [os.path.join(img_fol,pair[1]) for pair in list(final_overlap_set)]
    #     if not out_overlap_pkl:
    #         out_overlap = os.path.join(ba_dir, 'overlap_list_adapted_from_dense_matches.txt')
    #     else:
    #         out_overlap = os.path.join(ba_dir, out_overlap_pkl)
        
    #     print("Saving adjusted overlap list at {}".format(out_overlap))
    #     with open(out_overlap,'w') as f:
    #         for idx,img1 in enumerate(new_img1):
    #             out_str = '{} {}\n'.format(img1,new_img2[idx])
    #             f.write(out_str)


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
    dem_list = glob(os.path.join(in_dir, '20*', '20*', 'run-DEM.tif'))
    for dem in dem_list:
        new_dem = os.path.join(out_dir, dem.split('/')[-2] + '_run-DEM.tif')
        if not os.path.exists(new_dem):
            shutil.copy2(dem, new_dem)
    dem_list = glob(os.path.join(out_dir, '20*_map_run-DEM.tif'))

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