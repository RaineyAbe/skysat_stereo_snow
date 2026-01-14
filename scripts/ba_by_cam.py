#! /usr/bin/env python

import os
from glob import glob
import subprocess
from p_tqdm import p_map
import shutil
import numpy as np
import pandas as pd
from pprint import pprint
import json
from tqdm import tqdm
import rasterio as rio
import geopandas as gpd
from shapely.geometry import Point, Polygon
import rioxarray as rxr
import itertools

###############################
# HELPER FUNCTIONS
###############################

def run_cmd(
        bin, 
        args
        ) -> str:
    bin_path = shutil.which(bin)
    call = [bin_path]

    if bin_path.endswith('.py'):
        call.insert(0, 'python')
    
    call.extend(args)
    try:
        result = subprocess.run(
            call, check=True, capture_output=True, text=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Command '{' '.join(call)}' failed with error:\n{e.stderr}"
    

def setup_parallel_jobs(
        total_jobs: int = None,
        total_cpu: int = int(os.cpu_count() / 2),
        verbose: bool = True,
        ) -> tuple[int, int]:
    if total_jobs <= 1:
        njobs = 1
    elif total_jobs <= 10:
        njobs = min(2, total_jobs)
    elif total_jobs <= 100:
        njobs = min(4, total_jobs)
    else:
        njobs = min(4, total_jobs)

    threads_per_job = max(1, total_cpu // njobs)

    if verbose:
        print(f"Will run {total_jobs} jobs across {njobs} CPU with {threads_per_job} threads per CPU")

    return njobs, threads_per_job


def parse_image_specs(image_files: str, out_file: str = None, threads=int(os.cpu_count()/2)) -> pd.DataFrame:
    print(f"Parsing specs from {len(image_files)} image file names.")
    # parse datetime, rig (satellite), camera, and frame from each image file name
    def _parse_image_specs(image_file):
        image_base = os.path.basename(image_file)
        parts = image_base.split('_')
        dt_string = "".join(parts[0:2])
        dt = pd.Timestamp(
            f"{dt_string[0:4]}-{dt_string[4:6]}-{dt_string[6:8]}"
            f"T{dt_string[8:10]}:{dt_string[10:12]}:{dt_string[12:14]}"
        )
        df = pd.DataFrame({
            "image_name": os.path.splitext(image_base)[0],
            "datetime": dt,
            "rig": parts[2].split('d')[0],
            "cam": "".join(parts[2].partition('d')[1:]),
            "frame": parts[3]
        }, index=[0])
        return df
    
    # map function over all image files
    df_list = p_map(_parse_image_specs, image_files, num_cpus=threads)
    df_full = pd.concat(df_list, ignore_index=True)

    # save to file
    if out_file:
        df_full.to_csv(out_file, header=True, index=False)
        print(f"Image specs saved to:\n{out_file}")

    return df_full


def save_single_band_images(image_list, out_folder, band=1, threads=int(os.cpu_count() / 2), overwrite=False):
    os.makedirs(out_folder, exist_ok=True)

    def _save_single_band_image(image_file):
        out_file = os.path.join(out_folder, os.path.basename(image_file))
        args = [
            "-b", str(band),
            image_file, out_file
            ]
        if os.path.exists(out_file) & (overwrite==False):
            return out_file
        out = run_cmd("gdal_translate", args)
        return out_file
    
    print(f"Saving band {band} for {len(image_list)} images to: {out_folder}")
    out_image_list = p_map(_save_single_band_image, image_list, num_cpus=threads)

    return sorted(out_image_list)


def get_rpc_bounds(
        img_fn: str = None, 
        height: float = 0.0
        ) -> tuple[float, float, float, float]:
    
    with rio.open(img_fn) as src:
        if not src.rpcs:
            raise ValueError("Image does not contain RPC metadata.")

        # Create an RPC transformer from the dataset RPCs
        transformer = rio.transform.RPCTransformer(src.rpcs)
        width = src.width
        height_px = src.height

    # Image corner pixel coordinates expressed as (col, row)
    corners = [
        (0, 0),                           # top-left
        (width - 1, 0),                   # top-right
        (width - 1, height_px - 1),       # bottom-right
        (0, height_px - 1)                # bottom-left
    ]

    # unzip into cols, rows but keep in mind transformer expects rows, cols order
    cols, rows = zip(*corners)
    cols = np.array(cols, dtype=float)
    rows = np.array(rows, dtype=float)
    zs = np.full_like(cols, fill_value=float(height), dtype=float)

    # Construct the lon/lat coordinates
    lons, lats = transformer.xy(rows, cols, zs)

    min_lon = float(np.min(lons))
    max_lon = float(np.max(lons))
    min_lat = float(np.min(lats))
    max_lat = float(np.max(lats))

    return min_lon, min_lat, max_lon, max_lat
    

def calculate_baseline_to_height_ratio(
        img1: str = None, 
        img2: str = None, 
        utm_epsg: str = None
        ) -> float:
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


def get_image_polygon(img_fn, height=0.0, out_crs=None):
    # if no CRS, image is likely raw, ungeoregistered. Estimate using RPC.
    crs = rxr.open_rasterio(img_fn).rio.crs
    if not crs:
        min_x, min_y, max_x, max_y = get_rpc_bounds(img_fn, height=height)
        crs = "EPSG:4326"
    # otherwise, use the embedded image bounds.
    else:
        min_x, min_y, max_x, max_y = rxr.open_rasterio(img_fn).rio.bounds()
    # convert bounds to polygon
    bounds_poly = Polygon([[min_x, min_y], [max_x, min_y],
                            [max_x, max_y], [min_x, max_y],
                            [min_x, min_y]])
    # make sure bounds are in UTM projection
    bounds_gdf = gpd.GeoDataFrame(index=[0], geometry=[bounds_poly], crs=crs)
    bounds_gdf = bounds_gdf.to_crs(out_crs)

    return bounds_gdf.geometry[0]


def identify_stereo_pairs(
        img_list: str = None, 
        overlap_perc: float = 1, 
        bh_ratio_range: tuple = None,
        true_stereo: bool = True,
        utm_epsg: str = None,
        out_folder: str = None,
        write_basename: bool = False,
        refdem_file: str = None
        )-> None:
    # Make sure out_folder exists
    os.makedirs(out_folder, exist_ok=True)

    # Get image bounds polygons
    ref_height = float(rxr.open_rasterio(refdem_file).mean().data)
    polygons = {img: get_image_polygon(img, height=ref_height, out_crs=utm_epsg) for img in img_list}
    
    # Compare all unique pairs
    print('Identifying stereo image pairs...')
    print(f'Requirements:')
    print(f'\t- overlap >= {overlap_perc} %')
    if bh_ratio_range:
        print(f'\t- baseline to height ratio = {bh_ratio_range[0]} to {bh_ratio_range[1]}')
    print(f'\t- true stereo = {true_stereo}')
    overlapping_pairs = []
    overlap_ratios = []
    bh_ratios = []
    # number of combos for progress bar
    n = len(img_list)
    total = n * (n - 1) // 2
    for img1, img2 in tqdm(itertools.combinations(img_list, 2), total=total):
        poly1 = polygons[img1]
        poly2 = polygons[img2]

        intersection = poly1.intersection(poly2)
        if not intersection.is_empty:
            area1 = poly1.area
            area2 = poly2.area
            overlap_percent = intersection.area / min(area1, area2) * 100
            if overlap_percent >= overlap_perc:
                # check for B/H ratio thresholds if specified
                bh_ratio = calculate_baseline_to_height_ratio(img1, img2, utm_epsg)
                if bh_ratio_range:
                    if (bh_ratio < bh_ratio_range[0]) | (bh_ratio > bh_ratio_range[1]):
                        continue
                
                # check for true stereo if specified - datetimes must be different
                dt1 = '_'.join(os.path.basename(img1).split('_')[0:2])
                dt2 = '_'.join(os.path.basename(img2).split('_')[0:2])
                if true_stereo & (dt1==dt2):
                    continue

                bh_ratios += [bh_ratio]
                overlapping_pairs += [(img1, img2)]
                overlap_ratios += [overlap_percent]
    print('Number of overlapping stereo pairs identified =', len(overlap_ratios))
                    
    # Write to file
    out_fn = os.path.join(out_folder, 'stereo_image_pairs.txt')
    # add the header
    with open(out_fn, 'w') as f:
        f.write(f"img1 img2 datetime_identifier overlap_percent bh_ratio\n")
    # iterate over pairs
    for i, (img1, img2) in enumerate(overlapping_pairs):
        date1, time1 = os.path.basename(img1).split('_')[0:2]
        date2, time2 = os.path.basename(img2).split('_')[0:2]
        dt_text = date1 + '_' + time1 + '__' + date2 + '_' + time2
        with open(out_fn, 'a') as f:
            if write_basename:
                if i==0:
                    print('\nWriting image pairs with basename only.')
                f.write(f"{os.path.basename(img1)} {os.path.basename(img2)} {dt_text} {overlap_ratios[i]} {bh_ratios[i]}\n")
            else:
                if i==0:
                    print('Writing image pairs with full path name.')
                f.write(f"{img1} {img2} {dt_text} {overlap_ratios[i]} {bh_ratios[i]}\n")

    print('Overlapping stereo pairs saved to file:', out_fn)

    return out_fn


def get_stereo_opts(
        session: str = None, 
        threads: int = None, 
        texture: str = 'normal', 
        stop_point: int = -1, 
        unalign_disparity: bool = False
        ):
    """
    Get the stereo options for the ASP parallel_stereo command with robust outlier filtering.
    """
    stereo_opts = []

    # --- Session-specific arguments ---
    if session:
        # Specify the camera session type (e.g., rpc, pinhole).
        stereo_opts.extend(['-t', session])
    # threads for parallel stages (e.g., correlation).
    stereo_opts.extend(['--threads-multiprocess', str(threads)])
    # threads for serial stages (e.g., preprocessing).
    stereo_opts.extend(['--threads-singleprocess', str(threads)])
    # stop the pipeline before this step begins
    if stop_point != -1:
        stereo_opts.extend(['--stop-point', str(stop_point)])

    # --- Step 0: Preprocessing (stereo_pprc) ---
    # normalize brightness/contrast for each image independently
    stereo_opts.extend(['--individually-normalize'])
    # align images before matching
    stereo_opts.extend(["--alignment-method", "affineepipolar"])
    # sets the max number of interest points to detect per tile
    stereo_opts.extend(['--ip-per-tile', '2000'])
    # number of iterations for the RANSAC outlier rejection algorithm.
    stereo_opts.extend(['--ip-num-ransac-iterations','1000'])

    # Experimental filters to mask low-texture areas
    # masks pixels with a standard deviation below this threshold.
    stereo_opts.extend(['--stddev-mask-thresh', '0.5'])
    # kernel size for the standard deviation calculation (-1 auto-calculates)
    stereo_opts.extend(['--stddev-mask-kernel', '-1'])

    # --- Step 1: Correlation (stereo_corr) ---
    # specify the stereo matching algorithm
    stereo_opts.extend(['--stereo-algorithm', 'asp_mgm'])
    # size of the window used to compare pixels for matching
    if texture == 'low':
        stereo_opts.extend(['--corr-kernel', '9', '9'])
    elif texture == 'normal':
        stereo_opts.extend(['--corr-kernel', '7', '7'])
    # image tile size for correlation
    stereo_opts.extend(['--corr-tile-size', '1024'])
    # cost function for MGM/SGM. Mode 4 = census transform, good for illumination changes
    stereo_opts.extend(['--cost-mode', '4'])
    # number of pyramid levels for coarse-to-fine correlation.
    stereo_opts.extend(['--corr-max-levels', '5'])

    # --- Step 3: Sub-pixel Refinement (stereo_rfne) ---
    # algorithm for refining integer disparity matches to sub-pixel accuracy. Mode 9 = affine-adaptive.
    stereo_opts.extend(['--subpixel-mode', '9'])
    # kernel size for sub-pixel refinement
    if texture == 'low':
        stereo_opts.extend(['--subpixel-kernel', '21', '21'])
    elif texture == 'normal':
        stereo_opts.extend(['--subpixel-kernel', '15', '15'])
    # threshold to reject bad sub-pixel matches
    stereo_opts.extend(['--xcorr-threshold', '2'])
    # generate this many "*.match" points from the final disparity map.
    stereo_opts.extend(['--num-matches-from-disparity', '10000'])
    # save the final disparity map without epipolar alignment.
    if unalign_disparity:
        stereo_opts.extend(['--unalign-disparity'])

    return stereo_opts


def run_stereo(
        stereo_pairs_fn: str = None, 
        cam_list: list[str] = None, 
        dem_file: str = None,
        out_folder: str = None, 
        session: str = None,
        texture: str = 'normal', 
        stop_point: int = -1,
        threads: int = int(os.cpu_count() / 2),
        verbose: bool = True
        ) -> None:
    """
    Execute stereo matching for SkySat images using the ASP parallel_stereo command.

    Parameters
    ----------
    stereo_pairs_fn: str (default=None)
        Path to the text file containing overlapping image pairs.
    cam_folder: str (default=None)
        Path to the folder containing camera files. Required if using 'pinhole' session.
    dem_file: str (default=None)

    out_folder: str
        Path to the folder where the output stereo results will be saved.
    session: str (default=None)
        The session type to use for stereo matching. Options include 'rpc', 'pinhole', etc. ASP can often figure this out automatically.
    texture: str (default='normal')
        How much relative texture there is in your images. This is used for determining the correlation and refinement kernel. 
        Options = "low", "normal". For example, a flat, snowy landscape likely has "low" texture. 
    stop_point: int

    
    Returns
    ----------
    None
    """
    # Check if output folder exists
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    # Load the stereo pairs
    stereo_pairs_df = pd.read_csv(stereo_pairs_fn, sep=' ', header=0)

    # Determine number of CPUs for parallelization and threads per job
    ncpu, threads_per_job = setup_parallel_jobs(
        total_jobs=len(stereo_pairs_df), 
        total_cpu=threads, 
        verbose=verbose
        )
    
    # Define stereo arguments
    stereo_opts = get_stereo_opts(
        session=session, 
        threads=threads_per_job, 
        texture=texture, 
        stop_point=stop_point,
        )
    
    # Create jobs list for each stereo pair
    job_list = []
    for _, row in stereo_pairs_df.iterrows():
        # Determine output folder for stereo job
        IMG1 = os.path.splitext(os.path.basename(row['img1']))[0]
        IMG2 = os.path.splitext(os.path.basename(row['img2']))[0]
        out_prefix = os.path.join(out_folder, row['datetime_identifier'], IMG1 + '__' + IMG2, 'run')  

        # Construct the stereo job
        if cam_list:
            cam1 = [x for x in cam_list if IMG1 in x][0]
            cam2 = [x for x in cam_list if IMG2 in x][0]
            job = stereo_opts + [row['img1'], row['img2'], cam1, cam2, out_prefix]
        else:
            # Otherwise, use the images directly
            stereo_args = [row['img1'], row['img2'], out_prefix]
            job = stereo_opts + stereo_args
        # add DEM last
        if dem_file:
            job += [dem_file]

        # Add job to list of jobs
        job_list.append(job)

    if verbose:
        print('stereo arguments for first job:')
        print(job_list[0])
    
    # Run the jobs in parallel
    stereo_logs = p_map(run_cmd, ['parallel_stereo']*len(job_list), job_list, num_cpus=ncpu)

    # Save the consolidated log
    stereo_log_fn = os.path.join(out_folder, 'stereo_log.log')
    with open(stereo_log_fn, 'w') as f:
        for log in stereo_logs:
            f.write(log + '\n')
    if verbose:
        print("Consolidated stereo log saved at {}".format(stereo_log_fn))

    return


def copy_match_files(stereo_folder, out_folder, threads=int(os.cpu_count() / 2), prefix="run-"):
    os.makedirs(out_folder, exist_ok=True)
    # get match files from all subfolders
    match_files = sorted(
        glob(os.path.join(stereo_folder, "*.match"))
        + glob(os.path.join(stereo_folder, "*", "*.match"))
        + glob(os.path.join(stereo_folder, "*", "*", "*.match"))
        )
    
    # copy to the out_folder
    print(f"\nCopying {len(match_files)} match files to {out_folder}")
    def _copy_match_files(match_file):
        out_match_file = os.path.join(
            out_folder,
            prefix + os.path.basename(os.path.dirname(match_file)) + ".match"
            )
        _ = shutil.copy2(match_file, out_match_file)
        return out_match_file
    out_match_files = p_map(_copy_match_files, match_files, num_cpus=threads)

    # return the list of output match files
    return sorted(list(out_match_files))


def plan_incremental_bundle_adjust(image_specs_df, image_list, init_cam_list, ba_folder):
    print("\nPlanning incremental bundle adjust rounds")

    os.makedirs(ba_folder, exist_ok=True)

    # Get unique cameras and sort
    unique_cams = sorted(image_specs_df["cam"].unique())
    print(f"Found {len(unique_cams)} unique camera types to process sequentially: {unique_cams}")

    # Initialize parameters
    ba_rounds_params = {}
    cumulative_images = []
    cumulative_cams = []

    # Iterate over unique cameras
    for i, cam_to_float in enumerate(unique_cams):
        round_num = i + 1
        
        # Create a dedicated directory for this round's output
        # round_dir = os.path.join(ba_folder, f"round{round_num}_float{cam_to_float}")
        ba_prefix = os.path.join(ba_folder, f"run")

        # Identify cameras to fix (all from previous rounds)
        cams_to_fix = unique_cams[0:i]
        
        # Determine the fixed camera indices for this round
        num_fixed_cameras = image_specs_df[image_specs_df['cam'].isin(cams_to_fix)]['cam'].count()
        fixed_cam_indices = list(range(num_fixed_cameras)) if cams_to_fix else []
        
        # Get the new images and initial cameras for the camera we are floating
        new_images_mask = image_specs_df['cam'] == cam_to_float
        new_images = np.array(image_list)[new_images_mask].tolist()
        new_cams_initial = np.array(init_cam_list)[new_images_mask].tolist()

        # Update the cumulative lists for this round
        # Add the newly floated cameras after the ones we are fixing
        cumulative_images.extend(new_images)
        cumulative_cams.extend(new_cams_initial)

        # Store all the parameters for this round
        ba_rounds_params[round_num] = {
            "cam_to_float": cam_to_float,
            "cams_to_fix": cams_to_fix,
            "ba_prefix": ba_prefix,
            "image_files": list(cumulative_images),
            "cam_files": list(cumulative_cams),
            "fixed_cam_indices": fixed_cam_indices,
        }

        # Update the 'cumulative_cams' list for the next round
        start_index_of_new_cams = len(cumulative_cams) - len(new_cams_initial)
        for j, cam_path in enumerate(new_cams_initial):
            cam_basename = os.path.basename(cam_path)
            adjusted_cam_path = f"{ba_prefix}-{cam_basename}"
            cumulative_cams[start_index_of_new_cams + j] = adjusted_cam_path

    # Save to file
    ba_plan_file = os.path.join(ba_folder, "bundle_adjust_plan.json")
    with open(ba_plan_file, 'w') as json_file:
        json.dump(ba_rounds_params, json_file, indent=4)
    print(f"Bundle adjust plan saved to:\n{ba_plan_file}")

    # Return the dictionary
    return ba_rounds_params


def run_bundle_adjust(
        image_files,
        cam_files,
        gcp_files = None,
        skip_matching = False, 
        fixed_cam_indices: list[int] = None,
        threads = int(os.cpu_count() / 2),
        output_prefix = None,
        overlap_file = None,
        reuse_match_files = True,
        match_files_prefix = None,
        adjustments_prefix = None
        ):
    
    os.makedirs(os.path.basename(output_prefix), exist_ok=True)
    
    ba_args = [
        "--threads", str(threads),
        "--num-iterations", "500",
        "--num-passes", "1",
        "--save-cnet-as-csv",
        "--min-matches", "4",
        "--disable-tri-ip-filter",
        "--ip-per-tile", "1000",
        "--ip-inlier-factor", "0.2",
        "--ip-num-ransac-iterations", "1000",
        "--skip-rough-homography",
        "--min-triangulation-angle", "0.0001",
        "--remove-outliers-params", "75 3 5 6",
        "--individually-normalize",
        "-o", output_prefix
        ] + image_files + cam_files
    
    if gcp_files:
        ba_args += gcp_files
        ba_args += ["--fix-gcp-xyz"]

    if skip_matching:
        ba_args += ["--skip-matching"]

    if overlap_file and not skip_matching:
        ba_args += ["--overlap-list", overlap_file]

    if reuse_match_files:
        ba_args += ["--force-reuse-match-files"]

    if match_files_prefix:
        ba_args += ["--match-files-prefix", match_files_prefix]

    if cam_files: 
        # inline adjustments not allowed for RPC files - check for type
        if os.path.splitext(cam_files[0])[1]==".tsai":
            ba_args += ["--inline-adjustments"]

    if fixed_cam_indices:
        ba_args += ["--fixed-camera-indices", " ".join(np.array(fixed_cam_indices).astype(str))]

    if adjustments_prefix:
        ba_args += ["--input-adjustments-prefix", adjustments_prefix]

    # print(f"Bundle adjust arguments:\n{ba_args}")

    out = run_cmd("bundle_adjust", ba_args)
    return out


def main():
    # ==========================================================================
    # --- 0. User inputs and setup ---
    # ==========================================================================
    print("\n============================================================")
    print("\tINITIAL SETUP")
    print("============================================================")
    # Define paths in directory
    DATA_DIR = "/Users/rdcrlrka/Research/SkySat-Stereo/study-sites/MCS/20240420/ba_by_cam"
    INIT_CAMS_DIR = os.path.join(DATA_DIR, "init_pinhole_cams")
    INIT_MATCHES_FOLDER = os.path.join(DATA_DIR, "init_feature_matches")
    BA_DIR = os.path.join(DATA_DIR, "bundle_adjust")
    BA_PREFIX = os.path.join(BA_DIR, "run")
    STEREO_PAIRS_FILE = os.path.join(DATA_DIR, "stereo_pairs.txt")
    REFDEM_FILE = "/Users/rdcrlrka/Research/SkySat-Stereo/study-sites/MCS/refdem/USGS_3DEP/MCS_USGS_3DEP_merged.tif"

    # Settings
    THREADS = 10

    # Get files
    image_files = sorted(glob(os.path.join(DATA_DIR, "..", "SkySatScene", "*_analytic.tif")))
    init_cam_files = sorted(glob(os.path.join(INIT_CAMS_DIR, "*.tsai")))
    print(f"Located {len(image_files)} images and {len(init_cam_files)} cameras.")

    # Outputs
    image_specs_file = os.path.join(DATA_DIR, "image_specs.csv")
    single_band_folder = os.path.join(DATA_DIR, "single_band_images")

    # Parse image specs from file names 
    image_specs_df = parse_image_specs(image_files, image_specs_file, 10)

    # ==========================================================================
    # --- 1. Feature detection and matching ---
    # ==========================================================================
    print("\n============================================================")
    print("\tFEATURE DETECTION AND MATCHING")
    print("============================================================")

    # Save single-band version of images for bundle_adjust match filtering
    oneband_image_list = save_single_band_images(image_files, single_band_folder, band=1, threads=THREADS)

    # Identify stereo pairs
    stereo_pairs_file = identify_stereo_pairs(
        oneband_image_list,
        overlap_perc=1,
        bh_ratio_range=None,
        true_stereo=True,
        utm_epsg="EPSG:32611",
        out_folder=INIT_MATCHES_FOLDER,
        write_basename=False,
        refdem_file=REFDEM_FILE
    )

    print("\nTesting just with d1")
    stereo_pairs_df = pd.read_csv(stereo_pairs_file, sep=" ", header=0)
    stereo_pairs_df["cam1"] = [os.path.basename(x).split("_")[2] for x in stereo_pairs_df["img1"]]
    stereo_pairs_df["cam2"] = [os.path.basename(x).split("_")[2] for x in stereo_pairs_df["img2"]]
    stereo_pairs_d1_df = stereo_pairs_df.loc[(stereo_pairs_df["cam1"]=="ssc16d1") & (stereo_pairs_df["cam2"]=="ssc16d1")]

    stereo_pairs_d1_file = stereo_pairs_file.replace(".txt", "_d1.txt")
    stereo_pairs_d1_df.to_csv(stereo_pairs_d1_file, sep=" ", index=False)
    print(f"d1 stereo pairs saved to:\n{stereo_pairs_d1_file}")

    # Run stereo preprocessing
    run_stereo(
        stereo_pairs_fn=stereo_pairs_d1_file,
        cam_list=sorted([x for x in init_cam_files if "d1" in os.path.basename(x)]),
        out_folder = INIT_MATCHES_FOLDER,
        stop_point=1,
        threads=THREADS
    )


    # ==========================================================================
    # --- 2. Bundle adjust ---
    # ==========================================================================
    print("\n============================================================")
    print("\tBUNDLE ADJUST")
    print("============================================================")

    # Copy match files to bundle adjust folder
    copy_match_files(INIT_MATCHES_FOLDER, BA_DIR, threads=THREADS, prefix="run-")

    # Plan the rounds
    ba_rounds_params = plan_incremental_bundle_adjust(image_specs_df, oneband_image_list, init_cam_files, BA_DIR)

    # --- Run the rounds ---
    print("\nRunning the bundle adjust rounds...")
    for round_num in list(ba_rounds_params.keys())[0:1]:
        print(f"\nRound {round_num}\n----------")

        # Parse arguments from dictionary
        round_image_files = ba_rounds_params[round_num]["image_files"]
        round_cam_files = ba_rounds_params[round_num]["cam_files"]
        round_cam_to_float = ba_rounds_params[round_num]["cam_to_float"]
        round_cams_to_fix = ba_rounds_params[round_num]["cams_to_fix"]
        round_fixed_cam_indices = ba_rounds_params[round_num]["fixed_cam_indices"]
        round_ba_prefix = ba_rounds_params[round_num]["ba_prefix"]

        os.makedirs(os.path.dirname(round_ba_prefix), exist_ok=True)

        # Display the setup
        print(f"\t- Floating camera: \t\t{round_cam_to_float}")
        print(f"\t- Fixed camera(s): \t\t{round_cams_to_fix}")
        print(f"\t- Number of floating cameras: \t{len(round_image_files) - len(round_fixed_cam_indices)}")
        print(f"\t- Number of fixed cameras: \t{len(round_fixed_cam_indices)}")
        print(f"\t- Output prefix: \t\t{ba_rounds_params[round_num]["ba_prefix"]}")

        # Run! 
        run_bundle_adjust(
            image_files = round_image_files,
            cam_files = round_cam_files,
            fixed_cam_indices = round_fixed_cam_indices,
            threads = THREADS,
            output_prefix = round_ba_prefix,
            skip_matching = False,
            reuse_match_files = True,
            match_files_prefix = None
        )

    # Clean up: copy final cameras to root directory, delete redundant camera files
    # print("\nClean up: moving final cameras to root bundle adjust directory, removing redundant files.")
    # for round_num in list(ba_rounds_params.keys()):
    #     # get all camera files
    #     output_cams = sorted(glob(os.path.join(ba_rounds_params[round_num]["ba_prefix"] + '*.tsai')))

    #     # split between floated and fixed
    #     output_floated_cams = [x for x in output_cams if ba_rounds_params[round_num]["cam_to_float"] in os.path.basename(x)]
        
    #     # copy floated cams to root bundle adjust directory
    #     for c in tqdm(output_floated_cams, desc=f"Copying final {ba_rounds_params[round_num]["cam_to_float"]} cams"):
    #         _ = shutil.copy2(c, os.path.join(BA_DIR, os.path.basename(c)))

    #     # now, delete all camera files in subfolder
    #     for c in output_cams:
    #         _ = os.remove(c)


if __name__=="__main__":
    main()