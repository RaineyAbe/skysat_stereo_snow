#! /usr/bin/env python

import os
from glob import glob
import subprocess
import shutil
import pandas as pd
from p_tqdm import p_map
import rasterio as rio
import rioxarray as rxr
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pyproj
from tqdm import tqdm 
import itertools
from shapely.geometry import Polygon, Point
import geopandas as gpd


# =========================================== #
# HELPER FUNCTIONS
# =========================================== #

def run_cmd(
    bin: str = None, 
    args: list = None, 
    log_output_folder: str = None
    ) -> str:
    # --- Set-up logging --- 
    def save_log(out, log_output_folder, error=False):
        os.makedirs(log_output_folder, exist_ok=True)
        dt_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_output_folder, f"{os.path.basename(bin)}_{dt_now}.log")
        if error:
            log_file = log_file.replace(".log", "_ERROR.log")
        with open(log_file, "w") as f:
            f.write(out)

    # --- Construct the command --- 
    # Get executable location
    binpath = shutil.which(bin)
    if not binpath:
        error_message = f"Error: Binary '{bin}' not found in system PATH."
        if log_output_folder:
            save_log(error_message, log_output_folder, error=True)
        return error_message

    # Add "python" argument if command is a script
    call = [binpath]
    if binpath.endswith(".py"):
        call.insert(0, "python")

    # Add any arguments
    if args:
        call.extend(args)

    # --- Run the command ---
    out = subprocess.run(
        call,
        capture_output=True,
        text=True,
    )
    # Format output and check for errors
    output = out.stdout
    error = bool(out.returncode)

    # Write the output to log file
    if log_output_folder:
        save_log(output, log_output_folder, error)

    return output


def parse_image_specs_from_file(image_file):
    base_pieces = os.path.basename(image_file).split("_")
    df = pd.DataFrame({
        "image_file": image_file,
        "datetime": "_".join(base_pieces[0:2]),
        "sat": base_pieces[2].split("d")[0],
        "frame": base_pieces[3]
    }, index=[0])
    # check for camera
    if "d" in base_pieces[2]:
        df["cam"] = "d" + base_pieces[2].split("d")[1]
    else:
        df["cam"] = "N/A"
    # reorder columns
    df = df[["image_file", "datetime", "sat", "cam", "frame"]]

    return df


def get_rpc_bounds(
    img_fn: str, 
    height: float = 0.0, 
    output_crs: str = "EPSG:4326"
) -> tuple[float, float, float, float]:
    # Open the image file
    with rio.open(img_fn) as src:
        if not src.rpcs:
            raise ValueError("Image does not contain RPC metadata.")

        # Create an RPC transformer from the dataset RPCs
        transformer = rio.transform.RPCTransformer(src.rpcs)

        width = src.width
        height_px = src.height

    # Image corner pixel coordinates (col, row)
    corners = [
        (0, 0), (width - 1, 0),
        (width - 1, height_px - 1), (0, height_px - 1)
    ]
    cols, rows = zip(*corners)
    
    # Prepare arrays for transformation
    zs = np.full(len(cols), fill_value=float(height))

    # Transform to WGS84 lat/lon
    lons, lats = transformer.xy(np.array(rows), np.array(cols), zs)

    # If output CRS is not WGS84, reproject
    if output_crs.upper() != "EPSG:4326":
        reprojector = pyproj.Transformer.from_crs(
            "EPSG:4326", output_crs, always_xy=True
        )
        xs, ys = reprojector.transform(lons, lats)
    else:
        xs, ys = lons, lats

    # Calculate the min/max from the final coordinates
    min_x, max_x = float(np.min(xs)), float(np.max(xs))
    min_y, max_y = float(np.min(ys)), float(np.max(ys))

    return min_x, min_y, max_x, max_y
            

def setup_parallel_jobs(
        total_jobs: int = None,
        total_threads: int = int(os.cpu_count()/2),
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

    threads_per_job = max(1, total_threads // njobs)

    if verbose:
        print(f"Will run {total_jobs} jobs across {njobs} CPU with {threads_per_job} threads per CPU")

    return njobs, threads_per_job


def match_skysatscene_to_skysatcollect(
        scene_files, 
        collect_files, 
        dem_file
        ):
    # Create dataframe of SkySat Scene specs
    scene_specs_df_list = []
    for scene_file in scene_files:
        scene_specs_df_list += [parse_image_specs_from_file(scene_file)]
    scene_specs_df = pd.concat(scene_specs_df_list, ignore_index=True)

    # Estimate image bounds from RPC model
    with rxr.open_rasterio(dem_file).squeeze() as dem:
        height = float(dem.median().data) # average height from DEM
    with rxr.open_rasterio(collect_files[0]) as ortho:
        crs = f"EPSG:{ortho.rio.crs.to_epsg()}" # output CRS
    scene_specs_df['image_bounds'] = scene_specs_df['image_file'].apply(lambda x: get_rpc_bounds(x, height=height, output_crs=crs))

    # Create dataframe of SkySat Collect specs
    collect_specs_df_list = []
    for collect_file in collect_files:
        collect_specs_df_list += [parse_image_specs_from_file(collect_file)]
    collect_specs_df = pd.concat(collect_specs_df_list, ignore_index=True)

    # Match images to reference ortho images using datetime/sat columns
    image_specs_df = pd.merge(scene_specs_df, collect_specs_df, on=["datetime", "sat"], suffixes=("_img", "_ref"))
    image_specs_df.rename(columns={
        "image_file_img": "SkySatScene", 
        "image_file_ref": "SkySatCollect"
    }, inplace=True)
    image_specs_df.drop(
        columns=["cam_ref", "frame_ref"], inplace=True
        )

    # Count number of SkySatScenes per unique SkySatCollect
    counts = image_specs_df["SkySatCollect"].value_counts()
    print("Number of SkySatScenes per unique SkySatCollect:")
    print(counts)

    return image_specs_df

def asp_match_file_to_dataframe(match_file):
    # Convert binary match file to text file
    match_txt_file = os.path.splitext(match_file)[0] + '_match.txt'
    run_cmd('parse_match_file.py', [match_file, match_txt_file])

    with open(match_txt_file, 'r') as f:
        lines = f.read().splitlines()

    # Get the number of points for the left and right images
    nL, nR = map(int, lines[0].split())

    # Separate the file into its three main components
    left_points_raw = lines[1 : nL + 1]
    right_points_raw = lines[nL + 1 : nL + nR + 1]

    # Helper to convert feature lists to a DataFrame
    def text_list_to_df(text_list):
        data = [line.split()[:2] for line in text_list]
        return pd.DataFrame(data, columns=["col", "row"], dtype=float).reset_index()

    # Create dataframes for left and right image features
    left_df = text_list_to_df(left_points_raw).reset_index()
    right_df = text_list_to_df(right_points_raw).reset_index()

    # Merge dataframes
    matches_df = left_df.merge(right_df, on="index", suffixes=["_img", "_ref"])

    return matches_df[["index", "col_img", "row_img", "col_ref", "row_ref"]]


def create_gcp_from_ortho_and_dem(
        image_file, 
        ortho_file, 
        dem_file, 
        image_bounds,
        output_folder, 
        threads = int(os.cpu_count()/2),
        sigma_X = 0.2,
        sigma_Y = 0.2,
        sigma_Z = 10,
        delete_ortho_clip = True
        ):
    os.makedirs(output_folder, exist_ok=True)
    
    # --- Define file paths ---
    image_base = os.path.splitext(os.path.basename(image_file))[0]
    ortho_clip_file = os.path.join(output_folder, 'ortho_clip.tif')
    ip_file_img = os.path.join(output_folder, f"{image_base}.vwip")
    ip_file_ortho = os.path.join(output_folder, 'ortho_clip.vwip')
    match_prefix = os.path.join(output_folder, 'run')
    gcp_file = os.path.join(output_folder, f"{image_base}.gcp")

    # --- Crop orthoimage to image bounds ---
    if not os.path.exists(ortho_clip_file):
        image_bounds = [str(x) for x in np.array(image_bounds)]
        gdalwarp_args = [
            ortho_file, ortho_clip_file,
            "-te"
        ] + image_bounds
        run_cmd("gdalwarp", gdalwarp_args, log_output_folder=output_folder)

    # --- Find Interest Points ---
    ipfind_args = [
        image_file, ortho_clip_file,
        "--ip-per-tile", "1000",
        "--output-folder", output_folder
    ]
    if threads: 
        ipfind_args += ["--threads", str(int(threads))]
    run_cmd("ipfind", ipfind_args, output_folder)

    # --- Match Interest Points ---
    ipmatch_args = [
        image_file, ortho_clip_file,
        ip_file_img, ip_file_ortho,
        "--output-prefix", match_prefix,
        # Use homography for perspective-to-ortho matching
        "--ransac-constraint", "homography"
    ]
    if threads: 
        ipmatch_args += ["--threads", str(int(threads))]
    run_cmd("ipmatch", ipmatch_args, output_folder)

    # --- Parse Match File ---
    match_file = glob(match_prefix + '*.match')
    if len(match_file) < 1:
        return None
    else:
        match_file = match_file[0]
    match_df = asp_match_file_to_dataframe(match_file)
    
    # --- Extract X,Y,Z Coordinates for Each Match ---
    # Load files (and make sure they're closed when done)
    with rxr.open_rasterio(ortho_clip_file) as ortho:
        with rxr.open_rasterio(dem_file).squeeze() as dem:
    
            # Reproject DEM to orthoimage CRS
            dem = dem.rio.reproject_match(ortho)

            # Get CRS from orthoimage
            crs = f"EPSG:{ortho.rio.crs.to_epsg()}"

            # Extract X/Y from ortho, Z from DEM at each col_ref, row_ref pixel coordinate
            def extract_xyz(row):
                col_ref = row['col_ref']
                row_ref = row['row_ref']
                x = float(ortho.isel(x=int(col_ref), y=int(row_ref)).x.values)
                y = float(ortho.isel(x=int(col_ref), y=int(row_ref)).y.values)
                z = float(dem.isel(x=int(col_ref), y=int(row_ref)).data)
                return pd.Series({'X': x, 'Y': y, 'Z': z})

            xyz_df = match_df.apply(extract_xyz, axis=1).reset_index()

            # Reproject X,Y to lat/lon (required by ASP for GCP)
            transformer = pyproj.Transformer.from_crs(
                crs, "EPSG:4326", always_xy=True
            )
            lons, lats = transformer.transform(xyz_df['X'].values, xyz_df['Y'].values)
            xyz_df['X'] = lons
            xyz_df['Y'] = lats

    # Merge with match_df
    gcp_df = match_df.merge(xyz_df, on='index')

    # --- Construct final GCP DataFrame and save ---
    # Add other required parameters
    gcp_df["image_file"] = os.path.basename(image_file)
    gcp_df["sigma_X"] = sigma_X
    gcp_df["sigma_Y"] = sigma_Y
    gcp_df["sigma_Z"] = sigma_Z
    gcp_df["use_X"] = 1
    gcp_df["use_Y"] = 1

    # Reorder columns to the required ASP .gcp format
    final_gcp = gcp_df[[
        "Y", "X", "Z", "sigma_Y", "sigma_X", "sigma_Z", 
        "image_file", "col_img", "row_img", "use_X", "use_Y"
    ]]

    # Save to CSV
    final_gcp.to_csv(gcp_file, header=False, index=True, sep=',')

    # --- Delete clipped ortho file to save space ---
    if delete_ortho_clip:
        os.remove(ortho_clip_file)
        
    return gcp_file


def generate_frame_cameras(
        img_list = None,
        dem_file: str = None, 
        product_level: str = 'l1b', 
        refine_cameras: bool = False,
        create_gcp: bool = False,
        gcp_std: float = None,
        out_folder: str = None,
        threads: int = int(os.cpu_count()/2)
        ) -> str:
    # Make output directory if it doesn't exist
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    
    cam_list = img_list

    # Define output camera and GCP files
    frames = [os.path.splitext(os.path.basename(x))[0] for x in img_list] # grab just the image identifier strings
    out_cam_list = [os.path.join(out_folder,'{}.tsai'.format(frame)) for frame in frames]
    out_gcp_list = [os.path.join(out_folder,'{}.gcp'.format(frame)) for frame in frames]

    # Define reference height value where DEM has no data
    ht_datum = np.nanmedian(rxr.open_rasterio(dem_file).squeeze().data) 

    # Iterate over images
    log_list = []
    for img, cam, out_cam, out_gcp in zip(img_list, cam_list, out_cam_list, out_gcp_list):
        # construct arguments
        args = [
            '--threads', str(threads),
            '--focal-length', str(553846.153846),
            '--optical-center', str(1280), str(540),
            '--height-above-datum', str(ht_datum),
            '--reference-dem', dem_file,
            '--input-camera', cam,
            '-o', out_cam,
            img
        ]
        if refine_cameras:
            args += ["--refine-camera"]
        if create_gcp:
            args += ["--gcp-file", out_gcp]
        if gcp_std:
            args += ["--gcp-std", str(gcp_std)]
        if product_level=='l1b':
            args += ['--pixel-pitch', str(0.8)]
        else:
            args += ['--pixel-pitch', str(1.0)]

        # run command
        log = run_cmd('cam_gen', args, log_output_folder=out_folder)
        log_list += [log]
    
    # Remove basename from GCP file names
    # ASP's cam_gen writes full path for images in the GCP files. This does not play well during bundle adjustment.
    # The function returns a consolidated gcp file with all images paths only containing basenames so that bundle adjustment can roll along
    # See ASP's gcp logic here: https://stereopipeline.readthedocs.io/en/latest/tools/bundle_adjust.html#bagcp
    if create_gcp:
        print("Writing GCPs with dirname removed")  
        def clean_img_in_gcp(row):
            return os.path.basename(row[7])
        for out_gcp in tqdm(out_gcp_list):
            df = pd.read_csv(out_gcp, header=None,delimiter=r"\s+")
            df[7] = df.apply(clean_img_in_gcp, axis=1)
            df[0] = np.arange(len(df))
            out_fn = os.path.join(out_folder, os.path.basename(out_gcp).replace('.gcp', '_clean.gcp'))
            df.to_csv(out_fn, sep=' ', index=False, header=False)

    return


def run_bundle_adjust(
        image_files,
        cam_files,
        gcp_files = None,
        skip_matching = False, 
        threads = int(os.cpu_count() / 2),
        output_prefix = None,
        overlap_file = None,
        reuse_match_files = True,
        inline_adjustments=True,
        adjustments_prefix = None
        ):
    
    ba_args = [
        "--threads", str(threads),
        "--num-iterations", "500",
        "--num-passes", "1",
        "--save-cnet-as-csv",
        "--min-matches", "4",
        "--disable-tri-ip-filter",
        "--ip-per-tile", "4000",
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

    if inline_adjustments:
        ba_args += ["--inline-adjustments"]

    if adjustments_prefix:
        ba_args += ["--input-adjustments-prefix", adjustments_prefix]

    print(f"Bundle adjust arguments:\n{ba_args}")

    out = run_cmd("bundle_adjust", ba_args)
    return out


def run_mapproject(
        img_list: str = None, 
        cam_list: str = None, 
        ba_prefix: str = None,
        out_folder: str = None, 
        dem: str = 'WGS84', 
        t_res: float = None, 
        t_crs: str = None, 
        session: str = None, 
        orthomosaic: bool = False,
        threads: int = int(os.cpu_count() / 2)
        ) -> None:

    os.makedirs(out_folder, exist_ok=True)

    # Set up image specific arguments: output prefixes and cameras
    frames_list = [os.path.splitext(os.path.basename(img))[0] for img in img_list]
    out_list = [os.path.join(out_folder, img + '.tif') for img in frames_list]

    # Mapproject is automatically splitting images into a single tile, 
    # so only one threads is needed for each image job
    ncpu, threads_per_job = threads, 1

    # Set up mapproject arguments
    map_opts = [
        '--threads', str(threads_per_job),
        # limit to integer values, with 0 as no-data
        '--nodata-value', '0',
        '--ot', 'UInt16'
        ]
    if t_res:
        map_opts += ['--tr', str(t_res)]
    if t_crs:
        map_opts += ['--t_srs', str(t_crs)]
    if session:
        map_opts += ['--session', session]
    if ba_prefix:
        map_opts += ['--bundle-adjust-prefix', ba_prefix]

    # Set up jobs
    jobs_list = []
    for img, cam, out in tqdm(list(zip(img_list, cam_list, out_list))):
        job = map_opts + [dem, img, cam, out]
        jobs_list += [job]
    print('\nmapproject arguments for first job:')
    print(jobs_list[0])
    
    # Run in parallel
    log_list = p_map(run_cmd, ['mapproject']*len(jobs_list), jobs_list, num_cpus=ncpu)
    
    # Save compiled ortho log
    ortho_log = os.path.join(out_folder, 'ortho.log')
    print("Saving compiled orthorectification log at {}".format(ortho_log))
    with open(ortho_log,'w') as f:
        for log in log_list:
            f.write(log + '\n')
    
    # Create orthomosaic
    # if orthomosaic:
    #     print("\nCreating orthomosaic")
    #     # get unique image datetimes
    #     dt_list = list(set(sorted(['_'.join(os.path.basename(im).split('_')[0:2]) for im in out_list])))

    #     # define mosaic prefix containing timestamps of inputs
    #     mos_prefix = '__'.join(dt_list)

    #     # define output filenames
    #     mosaic_fn = os.path.join(out_folder, f'{mos_prefix}_orthomosaic.tif')

    #     run_gdal_merge(
    #         img_list=out_list, 
    #         mosaic_fn = mosaic_fn,
    #         t_res = t_res
    #         )


def get_image_polygon(img_fn, out_crs):
    # if no CRS, image is likely raw, ungeoregistered. Estimate using RPC.
    crs = rxr.open_rasterio(img_fn).rio.crs
    if not crs:
        min_x, min_y, max_x, max_y = get_rpc_bounds(img_fn, height=1500)
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


def calculate_baseline_to_height_ratio(
        img1: str = None, 
        img2: str = None, 
        utm_epsg: str = None
        ) -> float:
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


def identify_overlapping_image_pairs(
        img_list: str = None, 
        overlap_perc: float = 10, 
        bh_ratio_range: tuple = None,
        true_stereo: bool = True,
        utm_epsg: str = None,
        out_folder: str = None,
        )-> None:
    # Make sure out_folder exists
    os.makedirs(out_folder, exist_ok=True)

    # Get image bounds polygons
    polygons = {img: get_image_polygon(img, out_crs=utm_epsg) for img in img_list}
    
    # Compare all unique pairs
    print('Identifying stereo image pairs...')
    print(f'Requirements:')
    print(f'\toverlap >= {overlap_perc} %')
    if bh_ratio_range:
        print(f'\tbaseline to height ratio = {bh_ratio_range[0]} to {bh_ratio_range[1]}')
    print(f'\ttrue stereo = {true_stereo}')
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
                    
    # Write pairs to file (and only pairs, for ASP bundle_adjust)
    pairs_fn = os.path.join(out_folder, "overlapping_image_pairs.txt")
    specs_fn = os.path.join(out_folder, "overlapping_image_pairs_with_specs.txt")
    if os.path.exists(pairs_fn):
        os.remove(pairs_fn)
    if os.path.exists(specs_fn):
        os.remove(specs_fn)
    with open(pairs_fn, "a") as p:
        with open(specs_fn, "a") as s:
            # write header on specs file
            s.write(f"img1 img2 datetime_identifier overlap_percent bh_ratio\n")
            for i, (img1, img2) in enumerate(overlapping_pairs):
                date1, time1 = os.path.basename(img1).split('_')[0:2]
                date2, time2 = os.path.basename(img2).split('_')[0:2]
                dt_text = date1 + '_' + time1 + '__' + date2 + '_' + time2
                s.write(f"{img1} {img2} {dt_text} {overlap_ratios[i]} {bh_ratios[i]}\n")

                p.write(f"{os.path.basename(img1)} {os.path.basename(img2)}\n")
    
    print("Overlapping stereo pairs and specs saved to file.")

    return


def get_stereo_opts(
        session: str = None, 
        threads: int = None, 
        texture: str = 'normal', 
        stop_point: int = -1, 
        unalign_disparity: bool = False
        ):
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
    stereo_opts.extend(['--alignment-method', 'Affineepipolar'])
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
    # add stopping point if specified
    if stop_point!=-1:
        stereo_opts.extend(['--stop-point', str(stop_point)])
    # get the disparity map without any alignment
    if unalign_disparity:
        stereo_opts.extend(['--unalign-disparity'])
    
    return stereo_opts


def run_stereo(
        stereo_pairs_file: str = None, 
        cam_list: list[str] = None, 
        dem_file: str = None,
        out_folder: str = None, 
        session: str = None,
        texture: str = 'normal', 
        stop_point: int = -1,
        threads: int = int(os.cpu_count() / 2)
        ) -> None:
    os.makedirs(out_folder, exist_ok=True)
    
    # Load the stereo pairs
    stereo_pairs_df = pd.read_csv(stereo_pairs_file, sep=' ', header=0)

    # Determine number of CPUs for parallelization and threads per job
    ncpu, threads_per_job = setup_parallel_jobs(total_jobs=len(stereo_pairs_df), total_threads=threads)
    
    # Define stereo arguments
    stereo_opts = get_stereo_opts(
        session=session, 
        threads=threads_per_job, 
        texture=texture, 
        stop_point=stop_point
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
            job = stereo_opts + [row['img1'], cam1, row['img2'], cam2, out_prefix]
        else:
            # Otherwise, use the images directly
            stereo_args = [row['img1'], row['img2'], out_prefix]
            job = stereo_opts + stereo_args
        # add DEM last
        if dem_file:
            job += [dem_file]

        # Add job to list of jobs
        job_list.append(job)

    print('stereo arguments for first job:')
    print(job_list[0])
    
    # Run the jobs in parallel
    stereo_logs = p_map(run_cmd, ['parallel_stereo']*len(job_list), job_list, num_cpus=ncpu)

    # Save the consolidated log
    stereo_log_fn = os.path.join(out_folder, 'stereo_log.log')
    with open(stereo_log_fn, 'w') as f:
        for log in stereo_logs:
            f.write(log + '\n')
    print("Consolidated stereo log saved at {}".format(stereo_log_fn))

    return


# -------------------------------------------- #
# RUN THE WORKFLOW
# -------------------------------------------- #

# ===== Define inputs and outputs =====
data_folder = "/Users/rdcrlrka/Research/SkySat-Stereo/study-sites/MCS/20240420/"

# Inputs
image_files = sorted(glob(os.path.join(data_folder, "SkySatScene", "*_analytic.tif")))
refortho_files = sorted(glob(os.path.join(data_folder, "SkySatCollect_TOAR", "*_analytic.tif")))
refdem_file = os.path.join(data_folder, "..", "refdem", "MCS_refdem_lidar_COPDEM_merged.tif")
print(f"Located {len(image_files)} images.")
print(f"Located {len(refortho_files)} reference ortho images.")

# Outputs
out_folder = os.path.join(data_folder, "proc_out")
os.makedirs(out_folder, exist_ok=True)
gcp_folder = os.path.join(out_folder, "gcp")
init_cams_folder = os.path.join(out_folder, "init_cams")
ba_folder = os.path.join(out_folder, "bundle_adjust")
ba_prefix = os.path.join(ba_folder, "run")
init_ortho_folder = os.path.join(out_folder, "init_ortho")
stereo_folder = os.path.join(out_folder, "stereo")
single_band_folder = os.path.join(out_folder, "single_band_images")

# # ===== Create GCP using orthoimages and DEM =====
# print("\nCreating GCP using reference orthoimages and DEM")
# os.makedirs(gcp_folder, exist_ok=True)

# # Match SkySat Scenes (raw images) to SkySat Collects (ortho images)
# image_specs_df = match_skysatscene_to_skysatcollect(image_files, refortho_files, refdem_file)

# # Construct argument lists
# image_list = list(image_specs_df["SkySatScene"].values)
# refortho_list = list(image_specs_df["SkySatCollect"].values)
# refdem_list = [refdem_file] * len(image_list)
# image_bounds_list = list(image_specs_df["image_bounds"].values)
# output_folders = [os.path.join(gcp_folder, os.path.splitext(os.path.basename(x))[0]) for x in image_list]

# # Determine threads distribution for jobs
# njobs, threads_per_job = setup_parallel_jobs(len(image_list), total_threads=6)
# threads_list = [threads_per_job] * len(image_list)

# # Create GCP
# gcp_files = p_map(create_gcp_from_ortho_and_dem, image_list, refortho_list, refdem_list, image_bounds_list, output_folders, threads_list, num_cpus=njobs)


# # ===== Generate frame cameras =====
# print("\nGenerating frame cameras")
# os.makedirs(init_cams_folder, exist_ok=True)

# generate_frame_cameras(
#     image_files,
#     refdem_file,
#     out_folder = init_cams_folder
# )


# # ===== Run bundle adjust =====
# print("\n--------------------\nBUNDLE ADJUSTMENT\n--------------------\n")
# # Run bundle adjust in image/cam pairs for better optimization

# # Get files
# gcp_list = sorted(glob(os.path.join(gcp_folder, "*", "*.gcp")))
# image_list = [
#     os.path.join(os.path.dirname(image_files[0]), os.path.splitext(os.path.basename(x))[0] + ".tif")
#     for x in gcp_list
#     ]
# cam_list = [
#     os.path.join(init_cams_folder, os.path.splitext(os.path.basename(x))[0] + ".tsai")
#     for x in image_list
# ]

# # Create overlapping pairs file
# identify_overlapping_image_pairs(
#     image_list,
#     overlap_perc = 50,
#     bh_ratio_range = None,
#     true_stereo = True,
#     utm_epsg = "EPSG:32611",
#     out_folder = ba_folder
# )
# overlap_file = os.path.join(ba_folder, "overlapping_image_pairs.txt")
# overlap_specs_file = os.path.join(ba_folder, "overlapping_image_pairs_with_specs.txt")

# Create single-band images (multiband not accepted by bundle adjust)
# print("\nSaving single-band version of images for bundle adjust.")
# os.makedirs(single_band_folder, exist_ok=True)
# logs = []
# for image_file in tqdm(image_list):
#     image_out_file = os.path.join(single_band_folder, os.path.basename(image_file))
#     if os.path.exists(image_out_file):
#         continue
#     args = [
#         "-b", "1", 
#         image_file, 
#         os.path.join(single_band_folder, os.path.basename(image_file))
#     ]
#     logs += [run_cmd("gdalwarp", args)]
# if len(logs) > 0:
#     log_file = os.path.join(single_band_folder, "gdalwarp_logs_merged.log")
#     with open(log_file, "w") as f:
#         f.write("\n\n".join(logs))

# image_list = sorted(glob(os.path.join(single_band_folder, "*.tif")))

# overlap_specs_df = pd.read_csv(overlap_specs_file, sep=" ")

# # Run in camera groups for to decrease number of images in each job
# print("Running bundle adjust in camera groups.")
# image_cams = sorted(list(set([
#     "d" + os.path.basename(image).split('_')[2].split('d')[1] 
#     for image in image_list
#     ])))
# print(f"Detected {len(image_cams)} cameras: {image_cams}")
# for image_cam in image_cams[0:1]:
#     print(f"\nGroup 1: {image_cam}")
#     # get the index in image list of all images with the image_cam
#     indices = [i for i, image in enumerate(image_list) if image_cam in os.path.basename(image)]

#     # get the subsets of images, cameras, and GCP
#     image_subset = [image_list[i] for i in indices]
#     cam_subset = [cam_list[i] for i in indices]
#     gcp_subset = [gcp_list[i] for i in indices]
#     ba_prefix_subset = os.path.join(ba_folder, f"ba_{image_cam}")
#     overlap_specs_subset_df = overlap_specs_df.loc[
#         (overlap_specs_df["img1"].isin(image_subset)) & (overlap_specs_df["img2"].isin(image_subset))
#         ]
#     print("Number of images in group:", len(image_subset))
#     print("Number of image pairs in group:", len(overlap_specs_subset_df))

#     # run bundle adjust in two rounds: relative (feature matching), then absolute (GCP)
#     print("Running bundle adjust")

    # print("Pass 1: relative")
    # run feature detection and matching
    # for i,row in overlap_specs_subset_df.iterrows():
    #     IMG1, IMG2 = row["img1"], row["img2"]

    #     ipfind_args = [
    #         IMG1, IMG2,
    #         "--output-folder", ba_folder,
    #         "--threads", "8"
    #     ]
    #     run_cmd("ipfind", ipfind_args)
    #     ip_file1 = os.path.join(ba_folder, os.path.splitext(os.path.basename(IMG1))[0] + '.vwip')
    #     ip_file2 = os.path.join(ba_folder, os.path.splitext(os.path.basename(IMG2))[0] + '.vwip')
    #     # rename with prefix
    #     ip_file1_new = ba_prefix_subset + "-" + os.path.basename(ip_file1)
    #     os.rename(ip_file1, ip_file1_new)
    #     ip_file1 = ip_file1_new
    #     ip_file2_new = ba_prefix_subset + "-" + os.path.basename(ip_file2)
    #     os.rename(ip_file2, ip_file2_new)
    #     ip_file2 = ip_file2_new

    #     ipmatch_args = [
    #         IMG1, IMG2,
    #         ip_file1, ip_file2,
    #         "--output-prefix", ba_prefix_subset,
    #     ]
    #     run_cmd("ipmatch", ipmatch_args)
    
    # # run bundle adjust
    # run_bundle_adjust(
    #     image_files=image_subset,
    #     cam_files=cam_subset,
    #     gcp_files=None,
    #     skip_matching=True,
    #     threads=8,
    #     output_prefix=ba_prefix_subset,
    #     inline_adjustments=False,
    #     reuse_match_files=True
    #     )

    # print("Pass 2: absolute")
    # # get new cameras
    # run_bundle_adjust(
    #     image_files=image_subset,
    #     cam_files=cam_subset,
    #     gcp_files=gcp_subset,
    #     skip_matching=True,
    #     threads=8,
    #     output_prefix=ba_prefix_subset,
    #     inline_adjustments=True,
    #     adjustments_prefix=ba_prefix_subset
    #     )


# Construct pairs / argument lists
# num_images = len(image_list)
# # if there's an odd number of images, add the last image to the previous pair
# if num_images % 2 == 1:
#     istarts = np.arange(0, num_images-3, 2)
#     iends = istarts + 2
#     istarts = np.concat([istarts, np.array([num_images-3])])
#     iends = np.concat([iends, np.array([num_images])])
# else:
#     istarts = np.arange(0, num_images, 2)
#     iends = istarts + 2
# num_jobs = len(istarts)
# image_subset_list = [image_list[ix:iy] for ix,iy in zip(istarts, iends)]
# cam_subset_list = [cam_list[ix:iy] for ix,iy in zip(istarts, iends)]
# gcp_subset_list = [gcp_list[ix:iy] for ix,iy in zip(istarts, iends)]
# runs = [f"0{x}" if x < 10 else f"{x}" for x in np.arange(1, num_jobs+1)]
# ba_prefix_list = [os.path.join(ba_folder, f"run{x}") for x in runs]
# print(f"Will run bundle adjust in groups of ~2 images/cams, for a total of {len(runs)} runs.")


# # Set up parallel jobs
# ncpu, threads_per_job = setup_parallel_jobs(total_jobs=num_jobs, total_threads=int(os.cpu_count())/2)

# # Run the jobs in parallel
# p_map(
#     run_bundle_adjust, 
#     image_subset_list, 
#     cam_subset_list,
#     gcp_subset_list,
#     [False] * num_jobs,
#     [threads_per_job] * num_jobs,
#     ba_prefix_list,
#     num_cpus=ncpu
#     )

# ===== Initial orthorectification =====
# print("\n--------------------\nINITIAL ORTHORECTIFICATION\n--------------------\n")
# cam_list = sorted(glob(os.path.join(ba_folder, "*.tsai")))
# image_list = [
#     os.path.join(os.path.dirname(image_files[0]), os.path.splitext(os.path.basename(x))[0].split("-")[1] + ".tif")
#     for x in cam_list
#     ]

# run_mapproject(
#     image_list,
#     cam_list,
#     dem = refdem_file,
#     out_folder = init_ortho_folder,
#     t_res = 0.7
# )

# # ===== Stereo =====
# print("\n--------------------\nSTEREO\n--------------------\n")
# cam_list = sorted(glob(os.path.join(ba_folder, "*.tsai")))

# # Create overlapping stereo pairs
# identify_overlapping_image_pairs(
#     img_list = sorted(glob(os.path.join(init_ortho_folder, "*.tif"))), 
#     overlap_perc = 25, 
#     true_stereo = True,
#     utm_epsg = "EPSG:32611",
#     out_folder = stereo_folder,
#     )

# # Run stereo
# run_stereo(
#     stereo_pairs_file = os.path.join(stereo_folder, "overlapping_image_pairs_with_specs.txt"),
#     cam_list = cam_list,
#     dem_file = refdem_file,
#     out_folder = stereo_folder,
# )

# Rasterize point clouds
pc_files = sorted(glob(os.path.join(stereo_folder, '*', '*', '*-PC.tif')))
print(f'Gridding {len(pc_files)} point clouds')
for pc_file in tqdm(pc_files):
    args = [
        '--threads', str(int(os.cpu_count() / 2)),
        '--t_srs', 'EPSG:32611',
        '--tr', '2',
        pc_file
    ]
    log = run_cmd('point2dem', args)

# Mosaic DEMs
dem_files = sorted(glob(os.path.join(stereo_folder, '*', '*', '*-DEM.tif')))
print(f"Mosaicking {len(dem_files)} DEMs")
args = [
    "--threads", str(int(os.cpu_count() / 2)),
    "--t_srs", "EPSG:32611",
    "--tr", "2",
    "--median",
    "-o", os.path.join(stereo_folder, "DEM_mosaic_test.tif")

] + dem_files
run_cmd("dem_mosaic", args, log_output_folder=stereo_folder)

# ===== Coregistration =====

# ===== Final ortho =====
