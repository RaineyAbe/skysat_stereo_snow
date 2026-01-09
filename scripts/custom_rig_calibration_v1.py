#! /usr/bin/env python

import os
import subprocess
import itertools
from collections import defaultdict
import sys
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from p_tqdm import p_map
from shapely.geometry import Point, Polygon
from scipy.spatial.transform import Rotation
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import time
import ast
from datetime import datetime

# Add path to the Ames Stereo Pipeline
asp_path = "/Users/rdcrlrka/Research/StereoPipeline-3.6.0-2025-12-31-arm64-OSX/bin"
sys.path.append(asp_path)

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
        total_jobs: int, 
        num_cpu: int
        ) -> tuple[int, int]:
    if total_jobs <= 1:
        njobs = 1
        threads_per_job = num_cpu
    else:
        njobs = min(num_cpu, total_jobs)
        threads_per_job = max(1, num_cpu // njobs)
    print(f"Distributing {total_jobs} tasks across {njobs} jobs with {threads_per_job} threads each.")
    return njobs, threads_per_job

def parse_image_specs(image_name: str) -> pd.DataFrame:
    image_base = os.path.basename(image_name)
    parts = image_base.split('_')
    dt_string = "".join(parts[0:2])
    dt = pd.Timestamp(
        f"{dt_string[0:4]}-{dt_string[4:6]}-{dt_string[6:8]}"
        f"T{dt_string[8:10]}:{dt_string[10:12]}:{dt_string[12:14]}"
    )
    return pd.DataFrame({
        "image_name": os.path.splitext(image_base)[0],
        "datetime": dt,
        "rig": parts[2].split('d')[0],
        "cam": "".join(parts[2].partition('d')[1:]),
        "frame": parts[3]
    }, index=[0])


def save_cam_specs(df: pd.DataFrame, out_file: str):
    df_to_save = df.copy()

    # Define a function to convert an array to its list string representation
    # e.g., np.array([[1, 2], [3, 4]]) -> '[[1, 2], [3, 4]]'
    def _array_to_list_str(arr):
        return str(arr.tolist())

    # Apply the conversion to each array column
    for col in ['K', 'R', 't']:
        if col in df_to_save.columns:
            df_to_save[col] = df_to_save[col].apply(_array_to_list_str)

    df_to_save.to_csv(out_file, index=False)
    print(f"Saved camera specs to:\n{out_file}")
    return


def load_cam_specs(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    def str_to_array(s):
        try:
            # Safely evaluate the string to a Python list
            py_list = ast.literal_eval(s)
            # Convert the list to a NumPy array
            return np.array(py_list)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing string to array: {s}. Error: {e}")
            return np.array([])
    # Apply the parsing function to each column
    for col in ['K', 'R', 't']:
        if col in df.columns:
            df[col] = df[col].apply(str_to_array)
    return df



###############################
# CAMERA AND RIG PREPROCESSING
###############################

def generate_pinhole_cameras(
    img_list: list, 
    dem_file: str, 
    product_level: str, 
    out_folder: str, 
    threads: int
) -> str:
    print("\n--- Generating Pinhole Cameras ---")
    os.makedirs(out_folder, exist_ok=True)
    
    frames = [os.path.splitext(os.path.basename(x))[0] for x in img_list]
    out_cam_list = [os.path.join(out_folder, f'{frame}.tsai') for frame in frames]
    if all([os.path.exists(x) for x in out_cam_list]):
        print("All pinhole cameras already exist in file, skipping.")
        return
    with rio.open(dem_file) as src:
        ht_datum = np.nanmean(src.read(1, masked=True))
    
    njobs, threads_per_job = setup_parallel_jobs(total_jobs=len(img_list), num_cpu=threads)
    
    job_list = []
    cam_list = img_list
    for img, cam, out_cam in zip(img_list, cam_list, out_cam_list):
        args = [
            '--threads', str(threads_per_job),
            '--focal-length', '553846.153846',
            '--optical-center', '1280', '540',
            '--height-above-datum', str(ht_datum),
            '--datum', 'WGS84',
            '--reference-dem', dem_file,
            '--input-camera', cam,
            '-o', out_cam,
            img
        ]
        args += ['--pixel-pitch', '0.8' if product_level == 'l1b' else '1.0']
        job_list += [args]
    
    log_list = p_map(run_cmd, ['cam_gen']*len(job_list), job_list, num_cpus=njobs, desc="Running cam_gen")
    
    cam_gen_log = os.path.join(out_folder, 'cam_gen.log')
    print(f"Saving cam_gen log to: {cam_gen_log}")
    with open(cam_gen_log, 'w') as f:
        f.write('\n'.join(log_list))

    return cam_gen_log


def asp_pinhole_to_opencv_cameras(cam_list: list) -> pd.DataFrame:
    print("\nConverting TSAI models to OpenCV format...")
    cam_params_list = []
    for cam_file in tqdm(cam_list, desc="Converting cameras"):
        with open(cam_file, "r") as c:
            lines = c.read().splitlines()
            params = {k.strip(): v.strip() for k, v in (line.split(' = ', 1) for line in lines if ' = ' in line)}
            K = np.array([
                [float(params.get('fu', 0)), 0, float(params.get('cu', 0))],
                [0, float(params.get('fv', 0)), float(params.get('cv', 0))],
                [0, 0, 1]
            ])
            C = np.fromstring(params.get('C', '0 0 0'), sep=' ')
            R = np.fromstring(params.get('R', '1 0 0 0 1 0 0 0 1'), sep=' ').reshape(3, 3)
            t = -R @ C.reshape(3, 1)
            cam_params_list.append({
                "image_name": os.path.splitext(os.path.basename(cam_file))[0],
                "K": K, "R": R, "t": t
            })
    return pd.DataFrame(cam_params_list)


def opencv_cameras_to_asp_pinhole(
        cam_specs_df: pd.DataFrame, 
        out_folder: str
        ):
    print(f"\n--- Converting final OpenCV models to ASP TSAI format in '{out_folder}' ---")
    os.makedirs(out_folder, exist_ok=True)

    for _, row in tqdm(cam_specs_df.iterrows(), total=len(cam_specs_df), desc="Saving cameras"):
        image_name = row['image_name']
        K = row['K']
        R = row['R']
        t = row['t'] 

        # --- Conversion from OpenCV's [R|t] to ASP's (R, C) format ---
        # t = -R @ C -> C = -inv(R) @ t
        C = -R.T @ t
        
        # --- Extract intrinsic parameters from K matrix ---
        fu = K[0, 0]
        fv = K[1, 1]
        cu = K[0, 2]
        cv = K[1, 2]

        # --- Format parameters into strings for the TSAI file ---
        # Ensure vectors are flattened for clean string conversion
        C_str = ' '.join(map(str, C.ravel()))
        R_str = ' '.join(map(str, R.ravel()))

        # --- Write content to file ---
        output_filename = os.path.join(out_folder, f"{image_name}.tsai")
        with open(output_filename, 'w') as f:
            f.write(f"VERSION_4\n")
            f.write(f"PINHOLE\n")
            f.write(f"fu = {fu}\n")
            f.write(f"fv = {fv}\n")
            f.write(f"cu = {cu}\n")
            f.write(f"cv = {cv}\n")
            f.write(f"u_direction = 1 0 0\n")
            f.write(f"v_direction = 0 1 0\n")
            f.write(f"w_direction = 0 0 1\n")
            f.write(f"C = {C_str}\n")
            f.write(f"R = {R_str}\n")
            f.write(f"pitch = 0.8\n")
            f.write("NULL\n")

    print(f"\nSaved {len(cam_specs_df)} camera models to:\n{out_folder}")


###############################
# STEREO PAIR IDENTIFICATION
###############################

def get_image_polygon(img_fn: str, out_crs: str, height: float) -> Polygon:
    with rio.open(img_fn) as src:
        if not src.crs:
            min_x, min_y, max_x, max_y = get_rpc_bounds(img_fn, height)
            crs = "EPSG:4326"
        else:
            min_x, min_y, max_x, max_y = src.bounds
            crs = src.crs
        bounds_poly = Polygon([[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]])
        gdf = gpd.GeoDataFrame(geometry=[bounds_poly], crs=crs, index=[0])
        return gdf.to_crs(out_crs).geometry[0]

def get_rpc_bounds(img_fn: str, height: float) -> tuple:
    with rio.open(img_fn) as src:
        if not src.rpcs: raise ValueError("Image must have RPCs for bound estimation.")
        transformer = rio.transform.RPCTransformer(src.rpcs)
        cols, rows = zip(*[(0,0), (src.width, 0), (src.width, src.height), (0, src.height)])
        lons, lats = transformer.xy(rows, cols, np.full(len(cols), height))
        return min(lons), min(lats), max(lons), max(lats)

def calculate_baseline_to_height_ratio(img1: str, img2: str, utm_epsg: str) -> float:
    centers, heights = [], []
    for img in [img1, img2]:
        with rio.open(img) as src:
            if not src.rpcs: return np.nan
            heights.append(src.rpcs.height_off)
            point = Point(src.rpcs.long_off, src.rpcs.lat_off)
            gdf = gpd.GeoDataFrame(geometry=[point], crs="EPSG:4326", index=[0])
            utm_pt = gdf.to_crs(utm_epsg).geometry[0]
            centers.append((utm_pt.x, utm_pt.y))
    baseline = np.linalg.norm(np.array(centers[0]) - np.array(centers[1]))
    return baseline / np.mean(heights) if np.mean(heights) != 0 else 0


def identify_stereo_pairs(
    img_list: list, out_folder: str, overlap_perc: float, bh_ratio_range: tuple,
    true_stereo: bool, utm_epsg: str, write_basename: bool
):
    os.makedirs(out_folder, exist_ok=True)
    out_fn = os.path.join(out_folder, 'stereo_pairs.txt')
    print("\n--- Identifying Stereo Image Pairs ---")
    print(f"Overlap >= {overlap_perc}%, B/H ratio in {bh_ratio_range}, True Stereo: {true_stereo}")
    avg_ground_height = 1500
    polygons = {
        img: get_image_polygon(img, utm_epsg, avg_ground_height)
        for img in tqdm(img_list, desc="Generating image footprints")
    }
    stereo_pairs = []
    combinations = list(itertools.combinations(img_list, 2))
    for img1, img2 in tqdm(combinations, desc="Finding pairs"):
        overlap = polygons[img1].intersection(polygons[img2]).area / min(polygons[img1].area, polygons[img2].area) * 100
        if overlap >= overlap_perc:
            if true_stereo and ('_'.join(os.path.basename(img1).split('_')[0:2]) == '_'.join(os.path.basename(img2).split('_')[0:2])):
                continue
            bh_ratio = calculate_baseline_to_height_ratio(img1, img2, utm_epsg)
            if bh_ratio_range[0] <= bh_ratio <= bh_ratio_range[1]:
                dt1 = "_".join(os.path.splitext(os.path.basename(img1))[0].split("_")[0:2])
                dt2 = "_".join(os.path.splitext(os.path.basename(img2))[0].split("_")[0:2])
                dt_identifier = f"{dt1}__{dt2}"
                stereo_pairs.append({
                    "img1": os.path.basename(img1) if write_basename else img1,
                    "img2": os.path.basename(img2) if write_basename else img2,
                    "datetime_identifier": dt_identifier,
                    "overlap_percent": overlap,
                    "bh_ratio": bh_ratio
                })
    print(f'Identified {len(stereo_pairs)} valid stereo pairs.')
    if stereo_pairs:
        pd.DataFrame(stereo_pairs).to_csv(out_fn, sep=' ', index=False, float_format='%.3f')
        print(f'Stereo pairs list saved to: {out_fn}')
    return out_fn

###############################
# FEATURE EXTRACTION AND MATCHING
###############################

# ASP
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

    # stereo_pprc args : This is for preprocessing (adjusting image dynamic range, etc.)
    stereo_opts.extend(['--individually-normalize'])
    stereo_opts.extend(['--ip-per-tile', '8000'])
    stereo_opts.extend(['--ip-num-ransac-iterations','2000'])
    stereo_opts.extend(['--force-reuse-match-files'])
    stereo_opts.extend(['--skip-rough-homography'])
    stereo_opts.extend(["--alignment-method", "none"])
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
        stereo_pairs_fn: str = None, 
        cam_list: list[str] = None, 
        dem_file: str = None,
        out_folder: str = None, 
        session: str = None,
        texture: str = 'normal', 
        stop_point: int = -1,
        verbose: bool = True,
        threads: int = int(os.cpu_count() * 0.75)
        ) -> None:
    # Check if output folder exists
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    
    # Load the stereo pairs
    stereo_pairs_df = pd.read_csv(stereo_pairs_fn, sep=' ', header=0)

    # Determine number of CPUs for parallelization and threads per job
    ncpu, threads_per_job = setup_parallel_jobs(total_jobs=len(stereo_pairs_df), num_cpu=threads)
    
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


def copy_match_files(match_files, out_folder):
    print(f"Copying {len(match_files)} match files to: {out_folder}")
    os.makedirs(out_folder, exist_ok=True)
    for match_file in tqdm(match_files):
        image_pair_string = os.path.dirname(match_file).split("/")[-1]
        out_file = os.path.join(out_folder, "run-" + image_pair_string + ".match")
        _ = shutil.copy2(match_file, out_file)
    return


# OPENCV
def save_match_coords(filename, pts1, pts2):
    with open(filename, 'w') as f:
        f.write('pt1_x,pt1_y,pt2_x,pt2_y\n')  # Write header
        for i in range(len(pts1)):
            # Format the line as a comma-separated string and write to file
            line = f"{pts1[i, 0]},{pts1[i, 1]},{pts2[i, 0]},{pts2[i, 1]}\n"
            f.write(line)


def load_match_coords(filename):
    pts1_list = []
    pts2_list = []
    with open(filename, 'r') as f:
        next(f)  # skip header line
        for line in f:
            # Split the line by commas and convert parts to floats
            parts = line.strip().split(',')
            pts1_list.append([float(parts[0]), float(parts[1])])
            pts2_list.append([float(parts[2]), float(parts[3])])
    return np.array(pts1_list), np.array(pts2_list)


def detect_and_match_features(
    stereo_pairs_file: str,
    image_folder: str,
    cam_specs_df: pd.DataFrame,
    out_folder: str,
    nfeatures: int = 5000,
    overwrite: bool = False
):
    print("\n--- Detecting and Matching Features ---")
    os.makedirs(out_folder, exist_ok=True)

    # Check that stereo pairs file exists
    if not os.path.exists(stereo_pairs_file):
        raise FileNotFoundError("Stereo pairs file not found. Skipping feature matching.")

    # Read the stereo pairs file
    pairs_df = pd.read_csv(stereo_pairs_file, delim_whitespace=True)

    # Initialize feature detector and matcher
    sift = cv2.SIFT_create(nfeatures=nfeatures)
    bf = cv2.BFMatcher()

    # Iterate over stereo pairs
    all_matches = {}
    for i, row in tqdm(
        pairs_df.iterrows(), total=len(pairs_df), desc="Matching & Filtering"
    ):
        # parse image files
        img1_name_ext = row["img1"]
        img2_name_ext = row["img2"]
        img1_path = os.path.join(image_folder, img1_name_ext)
        img2_path = os.path.join(image_folder, img2_name_ext)
        img1_name = os.path.splitext(img1_name_ext)[0]
        img2_name = os.path.splitext(img2_name_ext)[0]

        # check if match file exists
        match_pair_file = os.path.join(out_folder, f"{img1_name}__{img2_name}_match.csv")

        try:
            if os.path.exists(match_pair_file) & (overwrite==False):
                print(
                    f"\nMatches already exist for pair: {img1_name_ext}, {img2_name_ext}. Loading from file."
                )
                pts1, pts2 = load_match_coords(match_pair_file)
                all_matches[i] = {
                    "img1": img1_name,
                    "img2": img2_name,
                    "pts1": pts1,
                    "pts2": pts2
                    }
                continue

            # get camera intrinsics
            K1 = cam_specs_df[cam_specs_df["image_name"] == img1_name].iloc[0]["K"]

            # load images
            img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
            img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)
            if (img1 is None) or (img2 is None):
                print(
                    f"Warning: Could not read image pair {img1_name_ext}, {img2_name_ext}"
                )
                continue

            # normalize images for the SIFT detector
            img1_norm = cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
            img2_norm = cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

            # detect keypoints (features)
            kp1, des1 = sift.detectAndCompute(img1_norm, None)
            kp2, des2 = sift.detectAndCompute(img2_norm, None)
            if des1 is None or des2 is None:
                print("No feature matches found. Continuing.")
                continue

            # match features
            matches = bf.knnMatch(des1, des2, k=2)

            # filter based on distance
            good_matches = []
            for match in matches:
                if len(match) == 2:
                    m, n = match
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            if len(good_matches) < 8:
                continue

            # convert to floats for OpenCV
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

            # calculate the essential matrix (i.e., the 3x3 matrix representing the relative pose)
            E, mask = cv2.findEssentialMat(
                pts1,
                pts2,
                cameraMatrix=K1,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0,
            )

            if mask is None:
                print(
                    f"Could not compute relative pose for {img1_name_ext}, {img2_name_ext}. Skipping."
                )
                continue

            # filter points using the RANSAC mask
            inlier_pts1 = pts1[mask.ravel() == 1]
            inlier_pts2 = pts2[mask.ravel() == 1]

            print(
                f"\nPair: {img1_name_ext} | Ratio Test: {len(good_matches)} -> RANSAC Inliers: {len(inlier_pts1)}"
            )

            # Save the final inlier coordinates to the CSV
            save_match_coords(match_pair_file, inlier_pts1, inlier_pts2)

            # Add to list of all matches
            all_matches[i] = {
                "img1": img1_name,
                "img2": img2_name,
                "pts1": pts1,
                "pts2": pts2
                }

        except Exception as e:
            print(f"An error occurred while processing pair {img1_name_ext}-{img2_name_ext}: {e}")

    return all_matches


###############################
# BUNDLE ADJUSTMENT
###############################

def estimate_initial_rig_geometry(
        cam_specs_df: pd.DataFrame, 
        ref_cam_id: str
        ) -> dict:
    print(f"\nEstimating initial rig geometry from camera poses relative to '{ref_cam_id}'")
    relative_poses = defaultdict(lambda: {'rotations': [], 'translations': []})

    # Identify all other camera IDs that need a relative pose
    other_cam_ids = [c for c in cam_specs_df['cam'].unique() if c != ref_cam_id]

    for dt, group in tqdm(cam_specs_df.groupby('datetime'), desc="Calculating relative poses"):
        try:
            # Find the reference camera for this datetime group
            ref_cam = group[group['cam'] == ref_cam_id].iloc[0]
            R_ref, t_ref = ref_cam['R'], ref_cam['t']

            # Calculate relative pose for all other cameras present in this group
            for other_cam_id in other_cam_ids:
                if other_cam_id in group['cam'].values:
                    other_cam = group[group['cam'] == other_cam_id].iloc[0]
                    R_other, t_other = other_cam['R'], other_cam['t']

                    # Pose of other_cam relative to ref_cam
                    relative_R = R_ref.T @ R_other
                    relative_t = R_ref.T @ (t_other - t_ref)

                    relative_poses[other_cam_id]['rotations'].append(relative_R)
                    relative_poses[other_cam_id]['translations'].append(relative_t)
        except (IndexError, KeyError):
            # This can happen if the reference camera is missing for a specific datetime
            print(f"Warning: Reference camera '{ref_cam_id}' not found at datetime {dt}. Skipping relative pose calculation for this group.")

    final_rig_geometry = {}
    print("Averaging measurements to get final rig geometry")
    for cam_id, poses in relative_poses.items():
        if not poses['rotations']: continue
        
        quats = Rotation.from_matrix(poses['rotations']).as_quat()
        # Handle single rotation case where quats is not a 2D array
        if quats.ndim == 1:
            avg_quat = quats
        else:
            avg_quat = np.mean(quats, axis=0)
        
        avg_R = Rotation.from_quat(avg_quat / np.linalg.norm(avg_quat)).as_matrix()
        avg_t = np.mean(np.array(poses['translations']), axis=0).reshape(3, 1)

        final_rig_geometry[cam_id] = {'R': avg_R, 't': avg_t}

    return final_rig_geometry


def build_observation_data(matches, cam_specs_df):
    print("\nBuilding initial 3D points and observation list")

    # Create lookup dictionaries for camera parameters and indices
    cam_map = {name: idx for idx, name in enumerate(cam_specs_df['image_name'])}
    cam_params_map = cam_specs_df.set_index('image_name').to_dict('index')

    observations = []
    points_3d = []
    point_map = {}  # Maps (cam_idx, pt_idx_in_cam) to point_3d_idx

    # Corrected loop to iterate over the list of match dictionaries
    for i in tqdm(matches.keys(), desc="Triangulating points"):
        cam1_name, cam2_name = matches[i]['img1'], matches[i]['img2']
        pts1, pts2 = matches[i]['pts1'], matches[i]['pts2']

        cam1_idx, cam2_idx = cam_map[cam1_name], cam_map[cam2_name]

        # Construct projection matrices P = K @ [R|t]
        R1, t1, K1 = cam_params_map[cam1_name]['R'], cam_params_map[cam1_name]['t'], cam_params_map[cam1_name]['K']
        R2, t2, K2 = cam_params_map[cam2_name]['R'], cam_params_map[cam2_name]['t'], cam_params_map[cam2_name]['K']
        P1 = K1 @ np.hstack((R1, t1))
        P2 = K2 @ np.hstack((R2, t2))

        # Triangulate points
        points4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points3d_new = (points4d_hom[:3, :] / points4d_hom[3, :]).T

        # Add observations and new 3D points
        for j in range(len(pts1)):
            # Create a unique key for the point in the first image
            pt1_key = (cam1_idx, j)
            if pt1_key in point_map:
                point_3d_idx = point_map[pt1_key]
            else:
                point_3d_idx = len(points_3d)
                points_3d.append(points3d_new[j])
                point_map[pt1_key] = point_3d_idx
                observations.append({'cam_idx': cam1_idx, 'pt_3d_idx': point_3d_idx, 'pt_2d': pts1[j]})

            # Link the observation from cam2 to the same 3D point
            pt2_key = (cam2_idx, j)
            point_map[pt2_key] = point_3d_idx
            observations.append({'cam_idx': cam2_idx, 'pt_3d_idx': point_3d_idx, 'pt_2d': pts2[j]})

    return np.array(points_3d), observations


def pack_params(base_poses, relative_poses, points_3d):
    # Base poses (6 params per base camera)
    base_params = np.array([
        np.concatenate((Rotation.from_matrix(pose['R']).as_rotvec(), pose['t'].ravel()))
        for key, pose in sorted(base_poses.items())
    ]).ravel()

    # Relative rig poses: 6 params per non-base camera
    relative_params = np.array([
        np.concatenate((Rotation.from_matrix(pose['R']).as_rotvec(), pose['t'].ravel()))
        for cam, pose in sorted(relative_poses.items())
    ]).ravel()

    return np.concatenate((base_params, relative_params, points_3d.ravel()))


def unpack_params(params, n_base_poses, n_relative, n_points):
    base_offset = n_base_poses * 6
    relative_offset = base_offset + n_relative * 6

    base_params = params[:base_offset].reshape((n_base_poses, 6))
    relative_params = params[base_offset:relative_offset].reshape((n_relative, 6))
    points_3d = params[relative_offset:].reshape((n_points, 3))

    return base_params, relative_params, points_3d


def fun(
        params, 
        n_base_poses, 
        n_relative, 
        n_points, 
        observations, 
        cam_specs_df, 
        ref_cam_id, 
        base_pose_map, 
        relative_cam_map
        ):
    """
    The cost function for the bundle adjustment, to be minimized by least_squares.

    This function calculates the reprojection error for all observations. For a given
    set of camera and point parameters, it projects every 3D point into every camera
    that sees it and calculates the pixel distance between the projected point and
    the originally observed feature point. The optimizer's goal is to adjust the
    input `params` to make the sum of squares of these distances as small as possible.
    """
    # === 1. Unpack Parameters ===
    # The optimizer provides one long flat vector of parameters. This step
    # reshapes that vector back into structured, usable arrays for base poses,
    # relative rig poses, and 3D point coordinates.
    base_params, relative_params, points_3d = unpack_params(
        params, n_base_poses, n_relative, n_points
    )

    all_residuals = []

    # === 2. Iterate Through Every Observation ===
    for obs in observations:
        # Observation = 2D feature point seen in an image
        cam_idx = obs["cam_idx"]      # The index of the camera
        pt_3d_idx = obs["pt_3d_idx"]  # The index of the 3D point
        pt_2d_obs = obs["pt_2d"]      # The measured (x, y) coordinates

        # --- Get Camera Information ---
        cam_spec = cam_specs_df.iloc[cam_idx]
        cam_name = cam_spec["cam"] 
        K = cam_spec["K"] 
        base_key = cam_spec["ref_base_key"]

        # Skip if this camera couldn't be associated with a base camera
        if pd.isna(base_key) or (base_key not in base_pose_map):
            all_residuals.append(np.array([0, 0]))
            continue

        # Get the current estimate for the 3D point's coordinates.
        point_3d = points_3d[pt_3d_idx]

        # === 3. Calculate the Final Absolute Pose of the Camera ===
        
        # --- Start with the pose of the reference camera for this time step ---
        # Get the 6 pose parameters for the base/reference camera
        base_pose_params = base_params[base_pose_map[base_key]]
        # Convert the 6 params into a standard 3x3 rotation matrix and 3x1 translation vector.
        R_base = Rotation.from_rotvec(base_pose_params[:3]).as_matrix()
        t_base = base_pose_params[3:].reshape(3, 1)

        # --- Combine with relative rig pose if this is not a reference camera ---
        if cam_name == ref_cam_id:
            # If this is a reference camera ('d1'), its pose is just the base pose.
            R_cam, t_cam = R_base, t_base
        else:
            # If this is a non-reference camera, apply the rig geometry.
            # Get the 6 parameters for the shared relative pose of this camera type.
            relative_pose_params = relative_params[relative_cam_map[cam_name]]
            R_rel = Rotation.from_rotvec(relative_pose_params[:3]).as_matrix()
            t_rel = relative_pose_params[3:].reshape(3, 1)

            # Compose the camera's absolute pose
            R_cam = R_base @ R_rel
            t_cam = R_base @ t_rel + t_base

        # === 4. Project the 3D Point into the Camera ===
        # x = K * [R|t] * X
        pt_proj = (K @ (R_cam @ point_3d.reshape(3, 1) + t_cam)).ravel()
        
        # Cconvert from homogeneous to 2D coordinates
        pt_proj_2d = pt_proj[:2] / pt_proj[2]

        # === 5. Calculate and Store the Residual (Reprojection Error) ===
        # Reprojection error = difference between where the model projects the point
        # and where the feature was actually observed in the image
        all_residuals.append(pt_proj_2d - pt_2d_obs)

    return np.concatenate(all_residuals)


def calculate_and_save_stats(raw_residuals, observations, cam_specs_df, output_filename):

    # Reshape the (N*2,) flat array into an (N, 2) array of (dx, dy) vectors
    residual_vectors = raw_residuals.reshape(-1, 2)
    
    # Calculate the magnitude of each error vector
    scalar_errors = np.linalg.norm(residual_vectors, axis=1)

    # Group the scalar errors by camera index
    errors_by_cam = defaultdict(list)
    for i, obs in enumerate(observations):
        cam_idx = obs['cam_idx']
        errors_by_cam[cam_idx].append(scalar_errors[i])
        
    # Calculate stats for each camera
    stats_list = []
    for cam_idx, errors in errors_by_cam.items():
        image_name = cam_specs_df.iloc[cam_idx]['image_name']
        stats = {
            'image_name': image_name,
            'residuals_mean': np.mean(errors),
            'residuals_median': np.median(errors),
            'residuals_std': np.std(errors),
            'num_observations': len(errors)
        }
        stats_list.append(stats)
        
    if not stats_list:
        print("Warning: No residual stats were generated to save.")
        return
        
    stats_df = pd.DataFrame(stats_list).sort_values(by='residuals_mean', ascending=False)
    stats_df.to_csv(output_filename, index=False, float_format='%.4f')
    print(f"Residual stats saved to: {output_filename}")


def build_jacobian(
    n_base_poses, 
    n_relative, 
    n_points, 
    observations, 
    cam_specs_df, 
    ref_cam_id, 
    base_pose_map, 
    relative_cam_map
    ):
    """
    Builds the sparsity structure of the Jacobian matrix.

    The Jacobian matrix represents the derivatives of each residual (reprojection error)
    with respect to each optimization parameter. For large-scale bundle adjustment,
    this matrix is enormous but also very sparse (mostly zeros).
    """
    # === 1. Calculate offsets and dimensions ===
    n_params_base = n_base_poses * 6
    n_params_relative = n_relative * 6
    n_params_total = n_params_base + n_params_relative + n_points * 3
    n_residuals = len(observations) * 2  

    # Initialize a sparse matrix
    A = lil_matrix((n_residuals, n_params_total), dtype=int)

    # === 2. Iterate over observations ===
    # Each observation's reprojection error depends on:
    #   - Absolute pose of the observing camera
    #   - 3D coordinates of the observed point
    for i, obs in enumerate(observations):
        cam_idx = obs["cam_idx"]
        pt_3d_idx = obs["pt_3d_idx"]

        cam_spec = cam_specs_df.iloc[cam_idx]
        base_key = cam_spec["ref_base_key"]

        if pd.isna(base_key) or base_key not in base_pose_map:
            continue
        
        row_start = i * 2

        # --- 3D Point ---
        pt_3d_start_idx = n_params_base + n_params_relative + pt_3d_idx * 3
        A[row_start : row_start + 2, pt_3d_start_idx : pt_3d_start_idx + 3] = 1

        # --- Base Camera Pose ---
        base_pose_idx = base_pose_map[base_key]
        base_start_idx = base_pose_idx * 6
        A[row_start : row_start + 2, base_start_idx : base_start_idx + 6] = 1

        # --- Relative Rig Pose ---
        # (only if not observed by the reference camera)
        if cam_spec["cam"] != ref_cam_id:
            if cam_spec["cam"] in relative_cam_map:
                relative_pose_idx = relative_cam_map[cam_spec["cam"]]
                relative_start_idx = n_params_base + relative_pose_idx * 6
                A[row_start : row_start + 2, relative_start_idx : relative_start_idx + 6] = 1

    return A


def print_relative_poses(poses_dict, ref_cam_id, title):
    """Prints the relative camera poses in a human-readable format."""
    print(f"\n--- {title} (relative to '{ref_cam_id}') ---")
    if not poses_dict:
        print("No relative poses to display.")
        return

    for cam_id, pose in sorted(poses_dict.items()):
        # Extract rotation and translation
        R = pose['R']
        t = pose['t'].ravel() # Flatten translation vector for printing

        # Convert rotation matrix to Euler angles (in degrees) for readability
        # The 'xyz' order is common, but can be changed if needed
        euler_angles = Rotation.from_matrix(R).as_euler('xyz', degrees=True)

        print(f"  Camera: '{cam_id}'")
        print(f"    Translation (m): [x: {t[0]:.4f}, y: {t[1]:.4f}, z: {t[2]:.4f}]")
        print(f"    Rotation (deg):  [roll: {euler_angles[0]:.4f}, pitch: {euler_angles[1]:.4f}, yaw: {euler_angles[2]:.4f}]")


def bundle_adjust(
    matches,
    cam_specs_df,
    out_folder,
    max_nfev=None,
    threads: int = int(os.cpu_count() * 0.75),
):    
    # --- Setup (largely unchanged) ---
    print("\nMapping all cameras to a reference camera")
    ref_cam_id = cam_specs_df["cam"].value_counts().idxmax()
    print(f"Using '{ref_cam_id}' as the reference camera (most frames).")
    initial_rig_poses = estimate_initial_rig_geometry(cam_specs_df, ref_cam_id)
    ref_cams_df = cam_specs_df[cam_specs_df["cam"] == ref_cam_id].copy()
    ref_frames_by_dt = {
        dt: sorted(group["frame"].astype(int).tolist())
        for dt, group in ref_cams_df.groupby("datetime")
    }
    ref_name_lookup = {
        (row["datetime"], row["frame"]): row["image_name"]
        for _, row in ref_cams_df.iterrows()
    }
    cam_specs_df["ref_base_key"] = None
    for idx, row in tqdm(cam_specs_df.iterrows(), total=len(cam_specs_df), desc="Mapping frames"):
        if row["cam"] == ref_cam_id:
            cam_specs_df.at[idx, "ref_base_key"] = row["image_name"]
        else:
            dt, frame = row["datetime"], row["frame"]
            if dt in ref_frames_by_dt:
                ref_frames = ref_frames_by_dt[dt]
                closest_ref_frame = min(ref_frames, key=lambda x: abs(x - int(frame)))
                closest_ref_frame_str = str(closest_ref_frame).zfill(4)
                master_key = (dt, closest_ref_frame_str)
                if master_key in ref_name_lookup:
                    cam_specs_df.at[idx, "ref_base_key"] = ref_name_lookup[master_key]
    
    points_3d, observations = build_observation_data(matches, cam_specs_df)
    unique_base_keys = sorted(cam_specs_df["ref_base_key"].dropna().unique())
    base_pose_map = {key: i for i, key in enumerate(unique_base_keys)}
    relative_cams = sorted([cam for cam in cam_specs_df['cam'].unique() if cam != ref_cam_id])
    relative_cam_map = {cam: i for i, cam in enumerate(relative_cams)}
    base_initial_poses = {name: {'R': cam['R'], 't': cam['t']} for name, cam in ref_cams_df.set_index("image_name").to_dict('index').items()}
    
    for cam in relative_cams:
        if cam not in initial_rig_poses:
            print(f"  WARNING: No initial relative pose calculated for '{cam}'. Using identity as starting guess.")
            initial_rig_poses[cam] = {"R": np.eye(3), "t": np.zeros((3, 1))}

    # --- Calculate the initial centroid ---
    initial_centers = []
    for pose in base_initial_poses.values():
        initial_centers.append((-pose['R'].T @ pose['t']).ravel())
    initial_centroid = np.mean(np.array(initial_centers), axis=0)
    print(f"\nInitial trajectory centroid: {initial_centroid}")

    n_base_poses = len(unique_base_keys)
    n_relative = len(relative_cams)
    n_points = len(points_3d)
    initial_params = pack_params(base_initial_poses, initial_rig_poses, points_3d)
    
    print_relative_poses(initial_rig_poses, ref_cam_id, "Initial Relative Rig Geometry")
    
    A = build_jacobian(n_base_poses, n_relative, n_points, observations, cam_specs_df, ref_cam_id, base_pose_map, relative_cam_map)

    # --- Run the optimization ---
    print("\nStarting optimization...")
    start_time = time.time()
    res = least_squares(
        fun, initial_params, jac_sparsity=A, verbose=2, x_scale="jac", ftol=1e-4, method="trf",
        max_nfev=max_nfev, workers=threads,
        args=(n_base_poses, n_relative, n_points, observations, cam_specs_df, ref_cam_id, base_pose_map, relative_cam_map),
    )
    end_time = time.time()
    print(f"Optimization Finished. Total time = {np.round(end_time - start_time, 2)} seconds")

    final_base_params_drifted, final_relative_params, final_points_3d_drifted = unpack_params(res.x, n_base_poses, n_relative, n_points)

    # --- Calculate the final centroid and the required shift ---
    final_centers_drifted = []
    for i in range(n_base_poses):
        R_drifted = Rotation.from_rotvec(final_base_params_drifted[i, :3]).as_matrix()
        t_drifted = final_base_params_drifted[i, 3:].reshape(3, 1)
        final_centers_drifted.append((-R_drifted.T @ t_drifted).ravel())
    final_centroid_drifted = np.mean(np.array(final_centers_drifted), axis=0)
    
    shift_vector = initial_centroid - final_centroid_drifted
    print(f"\nFinal drifted centroid:    {final_centroid_drifted}")
    print(f"Alignment shift vector:    {shift_vector}")

    # --- Apply the alignment shift to ALL cameras and points ---
    final_points_3d_aligned = final_points_3d_drifted + shift_vector
    
    final_base_params_aligned = final_base_params_drifted.copy()
    for i in range(n_base_poses):
        R = Rotation.from_rotvec(final_base_params_drifted[i, :3]).as_matrix()
        C_drifted = final_centers_drifted[i]
        C_aligned = C_drifted + shift_vector
        t_aligned = -R @ C_aligned.reshape(3, 1)
        final_base_params_aligned[i, 3:] = t_aligned.ravel()

    # --- Use ALIGNED parameters for all subsequent steps ---
    final_cam_specs_df = cam_specs_df.copy()
    for idx, row in final_cam_specs_df.iterrows():
        base_key = row["ref_base_key"]
        if pd.isna(base_key) or base_key not in base_pose_map:
            continue
        
        base_pose_idx = base_pose_map[base_key]
        base_pose_params = final_base_params_aligned[base_pose_idx]
        R_base = Rotation.from_rotvec(base_pose_params[:3]).as_matrix()
        t_base = base_pose_params[3:].reshape(3, 1)

        if row["cam"] == ref_cam_id:
            R_final, t_final = R_base, t_base
        else:
            if row["cam"] in relative_cam_map:
                relative_pose_idx = relative_cam_map[row["cam"]]
                relative_pose_params = final_relative_params[relative_pose_idx]
                R_rel = Rotation.from_rotvec(relative_pose_params[:3]).as_matrix()
                t_rel = relative_pose_params[3:].reshape(3, 1)
                R_final = R_base @ R_rel
                t_final = R_base @ t_rel + t_base
            else:
                R_final, t_final = R_base, t_base
        
        final_cam_specs_df.at[idx, "R"] = R_final
        final_cam_specs_df.at[idx, "t"] = t_final

    # Print final relative poses for comparison
    final_rig_poses_printable = {}
    for cam, idx in relative_cam_map.items():
        params = final_relative_params[idx]
        R_rel = Rotation.from_rotvec(params[:3]).as_matrix()
        t_rel = params[3:].reshape(3, 1)
        final_rig_poses_printable[cam] = {'R': R_rel, 't': t_rel}        
    print_relative_poses(final_rig_poses_printable, ref_cam_id, "Final Relative Rig Geometry")

    # Save final camera files
    final_cam_specs_file = os.path.join(out_folder, "final_cam_specs.csv")
    save_cam_specs(final_cam_specs_df, final_cam_specs_file)
    opencv_cameras_to_asp_pinhole(final_cam_specs_df, out_folder)

    # Plot the results
    plot_rig_comparison(
        initial_df=cam_specs_df,
        final_df=final_cam_specs_df,
        title=f"Camera Poses Before and After Optimization (Ref: '{ref_cam_id}')",
    )
    return



###############################
# VISUALIZATION
###############################

def plot_rig_comparison(initial_df, final_df, title="Camera Pose Drift (X/Y Plane)"):
    print("\n--- Plotting initial vs. final camera pose drift ---")

    # Helper function to calculate camera centers from a dataframe
    def get_centers(df):
        centers = {}
        for _, cam in df.iterrows():
            # Ensure R and t are numpy arrays
            R = np.array(cam['R'])
            t = np.array(cam['t'])
            # Calculate camera center in world coordinates: C = -R.T @ t
            center = -R.T @ t.reshape(3, 1)
            centers[cam['image_name']] = center.ravel()
        return pd.DataFrame.from_dict(centers, orient='index', columns=['x', 'y', 'z'])

    # Get initial and final centers
    initial_centers = get_centers(initial_df)
    final_centers = get_centers(final_df)

    # Merge the dataframes to align initial and final poses
    merged_df = initial_centers.join(final_centers, lsuffix='_init', rsuffix='_final')

    # Calculate the components for the quiver plot
    x_init = merged_df['x_init']
    y_init = merged_df['y_init']
    
    # U and V are the vector components of the drift
    u = merged_df['x_final'] - x_init
    v = merged_df['y_final'] - y_init
    
    # Color the arrows by the change in the Z coordinate
    delta_z = merged_df['z_final'] - merged_df['z_init']

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 12))

    quiver = ax.quiver(x_init, y_init, u, v, delta_z,
                       angles='xy', scale_units='xy', scale=1, cmap='coolwarm',
                       headwidth=4, headlength=5)

    # Add a colorbar to explain the Z-drift colors
    cbar = fig.colorbar(quiver)
    cbar.set_label("Change in Z (m)")

    # Add start and end points for clarity
    ax.scatter(x_init, y_init, c='grey', marker='o', alpha=0.7, label='Initial Centers')
    ax.scatter(merged_df['x_final'], merged_df['y_final'], c='black', marker='x', alpha=0.9, label='Final Centers')

    # Formatting the plot
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Ensure the aspect ratio is equal, so drifts in the X and Y directions look the same
    ax.set_aspect('equal')

    plt.show()


def bundle_adjust_one_cam(
        image_list: list[str] = None, 
        cam_list: str = None, 
        output_prefix: str = None,
        refdem_file: str = None, 
        refdem_uncertainty: float = 5, 
        skip_matching: bool = False,
        fixed_cam_indices: list[int] = None,
        threads: int = int(os.cpu_count() * 0.75),
        verbose: bool = True
        ):
    output_folder = os.path.dirname(output_prefix)
    os.makedirs(output_folder, exist_ok=True)

    # construct the arguments
    args = [
        "--threads", str(threads),
        "--num-iterations", "500",
        "--num-passes", "2",
        "--inline-adjustments",
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
        ] + image_list + cam_list

    if skip_matching:
        args += ["--force-reuse-match-files"]
        args += ["--skip-matching"]

    if refdem_file:
        args += ["--heights-from-dem", refdem_file]
        args += ["--heights-from-dem-uncertainty", str(refdem_uncertainty)]

    if fixed_cam_indices:
        args += ["--fixed-camera-indices", " ".join(fixed_cam_indices)]    

    # run bundle adjust
    log = run_cmd('parallel_bundle_adjust', args)

    # write log to file
    dt_now = datetime.now()
    dt_now_string = str(dt_now).replace('-','').replace(' ','').replace(':','').replace('.','')
    log_file = output_prefix + f'-parallel_bundle_adjust_{dt_now_string}.log'
    with open(log_file, 'w') as f:
        f.write(log)
    
    if verbose:
        print('Saved compiled log to file:', log_file)
        print('Bundle adjust complete.')
        
    return


###############################
# MAIN WORKFLOWS
###############################

def main():    
    # -----Define inputs and outputs -----
    # Inputs
    data_folder = "/Users/rdcrlrka/Research/SkySat-Stereo/study-sites/MCS/20240420"
    image_folder = os.path.join(data_folder, "SkySatScene")
    image_list = sorted(glob(os.path.join(image_folder, "*_analytic.tif")))
    refdem_file = os.path.join(data_folder, "..", "refdem", "MCS_refdem_lidar_COPDEM_merged.tif")
    if not image_list:
        raise FileNotFoundError(f"Error: No images found in '{image_folder}'. Please check the path.")
    if not os.path.exists(refdem_file):
        raise FileNotFoundError(f"Cannot locate reference DEM file. Please check the path: {refdem_file}")
    
    # Outputs
    out_folder = os.path.join(data_folder, "custom_rig_calibration")
    init_cam_folder = os.path.join(out_folder, "init_pinhole_cams")
    init_matches_folder = os.path.join(out_folder, "init_feature_matches")
    ba_folder = os.path.join(out_folder, "bundle_adjustment")

    # -----Generate pinhole cameras-----
    print("\n========================================")
    print("PINHOLE CAMERA GENERATION")
    print("========================================")
    generate_pinhole_cameras(
        image_list,
        refdem_file, 
        product_level="l1b",
        out_folder=init_cam_folder,
        threads = 8
    )
    init_cam_list = sorted(glob(os.path.join(init_cam_folder, "*.tsai")))
    if not init_cam_list:
        print(f"\nError: No .tsai camera files found in '{init_cam_folder}'.")
        return

    # Convert to OpenCV-compatible camera params
    os.makedirs(ba_folder, exist_ok=True)
    cam_params_df = asp_pinhole_to_opencv_cameras(init_cam_list)

    # Parse and add image specs
    specs_df = pd.concat(map(parse_image_specs, image_list), ignore_index=True)
    cam_specs_df = pd.merge(cam_params_df, specs_df, on="image_name")

    # Save to file
    cam_specs_file = os.path.join(init_cam_folder, "initial_cam_specs.csv")
    save_cam_specs(cam_specs_df, cam_specs_file)
    
    # Create a unique integer index for each camera for easy lookup
    cam_specs_df.reset_index(inplace=True, drop=True)
    cam_specs_df['cam_idx'] = cam_specs_df.index
    
    # -----Detect and match features-----
    print("\n========================================")
    print("FEATURE DETECTION AND MATCHING")
    print("========================================")
    # Identify stereo pairs
    stereo_pairs_file = identify_stereo_pairs(
        img_list=image_list, 
        out_folder=out_folder, 
        overlap_perc=10, 
        bh_ratio_range=[0.25, 5], 
        true_stereo=True, 
        utm_epsg="EPSG:32611",
        write_basename=False
        )
    
    # Detect and match features
    # run_stereo(
    #     stereo_pairs_fn=stereo_pairs_file,
    #     cam_list=init_cam_list,
    #     out_folder=init_matches_folder,
    #     stop_point=1,
    #     threads=10
    # )
    
    # Copy match files to bundle adjust folder
    match_files = sorted(glob(os.path.join(init_matches_folder, "*", "*", "*.match")))
    copy_match_files(match_files, ba_folder)

    # matches = detect_and_match_features(
    #     stereo_pairs_file=stereo_pairs_file,
    #     image_folder=image_folder,
    #     cam_specs_df=cam_specs_df,
    #     out_folder=init_matches_folder,
    #     nfeatures=5000,
    #     overwrite=False
    # )
    # if not matches:
    #     print("\nNo feature matches found. Exiting before optimization.")
    #     return
    
    # -----Custom bundle adjustment-----
    # print("\n========================================")
    # print("BUNDLE ADJUSTMENT")
    # print("========================================")

    # ROUND 1: best triplet for reference cam
    image_list_ref = [x for x in image_list if "d1" in os.path.basename(x)]
    cam_list_ref = [x for x in init_cam_list if "d1" in os.path.basename(x)]

    # subset stereo pairs to reference pairs
    stereo_pairs = pd.read_csv(stereo_pairs_file, sep=" ", header=0)
    stereo_pairs_ref = stereo_pairs.loc[stereo_pairs["img1"].isin(image_list_ref) & stereo_pairs["img2"].isin(image_list_ref)]
    print(f"{len(stereo_pairs_ref)} reference stereo pairs")

    # Identify the best starting triplet
    def get_dt_from_fname(filename):
        """Helper function to extract datetime from the SkySat filename."""
        base = os.path.basename(filename)
        dt_str = base.split('_')[0] + base.split('_')[1]
        return datetime.strptime(dt_str, '%Y%m%d%H%M%S')

    # Add datetime objects to the dataframe for easy comparison
    stereo_pairs_ref['dt1'] = stereo_pairs_ref['img1'].apply(get_dt_from_fname)
    stereo_pairs_ref['dt2'] = stereo_pairs_ref['img2'].apply(get_dt_from_fname)

    all_valid_triplets = []

    # Iterate through each stereo pair, considering it as the first link (A, B) in a chain
    for _, ab_pair in stereo_pairs_ref.iterrows():
        A, dt_A = ab_pair['img1'], ab_pair['dt1']
        B, dt_B = ab_pair['img2'], ab_pair['dt2']
        overlap_AB = ab_pair['overlap_percent']
        
        # Now, find all pairs that connect to B, which could be the second link (B, C)
        # Exclude the pair we started with to avoid B->A connections
        candidate_bc_pairs = stereo_pairs_ref[
            ((stereo_pairs_ref['img1'] == B) & (stereo_pairs_ref['img2'] != A)) |
            ((stereo_pairs_ref['img2'] == B) & (stereo_pairs_ref['img1'] != A))
        ]
        
        for _, bc_pair in candidate_bc_pairs.iterrows():
            # Determine which image is C and get its datetime
            if bc_pair['img1'] == B:
                C, dt_C = bc_pair['img2'], bc_pair['dt2']
            else:
                C, dt_C = bc_pair['img1'], bc_pair['dt1']
            
            overlap_BC = bc_pair['overlap_percent']

            # Enforce the constraint: all three images must have different datetimes
            if len({dt_A, dt_B, dt_C}) == 3:
                mean_overlap = (overlap_AB + overlap_BC) / 2.0
                all_valid_triplets.append({
                    'triplet': [A, B, C],
                    'mean_overlap': mean_overlap
                })

    if not all_valid_triplets:
        raise ValueError("Could not find any valid triplets with distinct datetimes. Check stereo pair overlaps.")

    # Find the single best triplet from all the valid ones we found
    best_triplet_info = max(all_valid_triplets, key=lambda x: x['mean_overlap'])
    triplet = best_triplet_info['triplet']

    print("Using the following reference triplet for initial fixed cameras:")
    print(f"(Based on a maximum mean overlap of {best_triplet_info['mean_overlap']:.2f}%)")
    for img in triplet:
        print(f"  {os.path.basename(img)}")

    # Identify their indices in the image_list_ref, ensuring a consistent order
    triplet_sorted = [img for img in image_list_ref if img in triplet]
    fixed_cam_indices = [str(image_list_ref.index(img)) for img in triplet_sorted]

    print("Fixed camera indices:", fixed_cam_indices)

    # bundle_adjust_one_cam(
    #     image_list=image_list_ref,
    #     cam_list=cam_list_ref,
    #     output_prefix=os.path.join(ba_folder, "run"),
    #     skip_matching=True,
    #     fixed_cam_indices=fixed_cam_indices
    # )

    # ROUND 2: all reference cams with first triplet fixed

    # bundle_adjust(
    #     matches=matches,
    #     cam_specs_df=cam_specs_df,
    #     out_folder=ba_folder,
    #     max_nfev=None,
    #     threads=10,
    # )

if __name__ == "__main__":
    main()

