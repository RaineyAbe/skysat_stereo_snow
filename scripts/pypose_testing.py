#! /usr/bin/env python

"""
pypose_bundle_adjustment.py

A script to perform rig-constrained bundle adjustment on SkySat imagery using the PyPose library.
"""

import os
import sys
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from collections import defaultdict
import cv2
import ast
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import itertools

import rasterio as rio
from shapely.geometry import Polygon, Point
from p_tqdm import p_map
import shutil
import subprocess
import geopandas as gpd

import torch
import pypose as pp
from torch import nn


# ==============================================================================
# BASE HELPER FUNCTIONS
# ==============================================================================

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


def estimate_initial_rig_geometry(cam_specs_df: pd.DataFrame, ref_cam_id: str) -> dict:
    # This function is unchanged
    print(f"\nEstimating initial rig geometry from camera poses relative to '{ref_cam_id}'")
    relative_poses = defaultdict(lambda: {'rotations': [], 'translations': []})
    other_cam_ids = [c for c in cam_specs_df['cam'].unique() if c != ref_cam_id]
    for dt, group in tqdm(cam_specs_df.groupby('datetime'), desc="Calculating relative poses"):
        try:
            ref_cam = group[group['cam'] == ref_cam_id].iloc[0]
            R_ref, t_ref = ref_cam['R'], ref_cam['t']
            for other_cam_id in other_cam_ids:
                if other_cam_id in group['cam'].values:
                    other_cam = group[group['cam'] == other_cam_id].iloc[0]
                    relative_R = R_ref.T @ other_cam['R']
                    relative_t = R_ref.T @ (other_cam['t'] - t_ref)
                    relative_poses[other_cam_id]['rotations'].append(relative_R)
                    relative_poses[other_cam_id]['translations'].append(relative_t)
        except (IndexError, KeyError):
            pass # Skip if reference camera is missing
    final_rig_geometry = {}
    print("Averaging measurements to get final rig geometry...")
    for cam_id, poses in relative_poses.items():
        if not poses['rotations']: continue
        quats = Rotation.from_matrix(poses['rotations']).as_quat()
        avg_quat = quats[0] if quats.ndim == 1 else np.mean(quats, axis=0)
        avg_R = Rotation.from_quat(avg_quat / np.linalg.norm(avg_quat)).as_matrix()
        avg_t = np.mean(np.array(poses['translations']), axis=0).reshape(3, 1)
        final_rig_geometry[cam_id] = {'R': avg_R, 't': avg_t}
    return final_rig_geometry


def build_observation_data(matches, cam_specs_df):
    """
    Triangulate 3D points from 2D feature matches for bundle adjustment
    """
    print("\nBuilding initial 3D points and observations list...")

    # Create helper dictionaries for faster camera lookup
    cam_map = {name: idx for idx, name in enumerate(cam_specs_df['image_name'])}
    cam_params_map = cam_specs_df.set_index('image_name').to_dict('index')

    # Initialize lists of 3D points and 2D observations
    all_points_3d = []
    all_observations = []

    # Iterate over matches
    for match in tqdm(matches.values(), desc="Triangulating points"):
        
        # --- Get 2D observations (feature match pixel coordinates) ---
        cam1_name, cam2_name = match['img1'], match['img2']
        pts1, pts2 = match['pts1'], match['pts2']
        cam1_idx = cam_map[cam1_name]
        cam2_idx = cam_map[cam2_name]

        # --- Triangulate points for image pair ---
        # Get camera intrinsics
        R1, t1, K1 = cam_params_map[cam1_name]['R'], cam_params_map[cam1_name]['t'], cam_params_map[cam1_name]['K']
        R2, t2, K2 = cam_params_map[cam2_name]['R'], cam_params_map[cam2_name]['t'], cam_params_map[cam2_name]['K']

        # Form projection matrices
        P1 = K1 @ np.hstack((R1, t1))
        P2 = K2 @ np.hstack((R2, t2))

        # Triangulate points to find the 3D position of each feature match
        points4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        
        # Convert from 4D homogeneous coordinates to 3D
        points3d_for_this_pair = (points4d_hom[:3, :] / points4d_hom[3, :]).T

        # --- Add new points and observations ---
        point_index_offset = len(all_points_3d)
        for j, new_point_3d in enumerate(points3d_for_this_pair):
            
            # Calculate the globally unique index for this 3D point
            global_point_idx = point_index_offset + j
            all_points_3d.append(new_point_3d)

            # Create observation for the FIRST camera
            observation1 = {
                'cam_idx': cam1_idx,
                'pt_3d_idx': global_point_idx,
                'pt_2d': pts1[j]
            }
            all_observations.append(observation1)

            # Create observation for the SECOND camera
            observation2 = {
                'cam_idx': cam2_idx,
                'pt_3d_idx': global_point_idx,
                'pt_2d': pts2[j]
            }
            all_observations.append(observation2)

    return np.array(all_points_3d), all_observations



# ==============================================================================
# CAMERA GENERATION AND PROCESSING
# ==============================================================================

def generate_pinhole_cameras(
    img_list: list, 
    dem_file: str, 
    product_level: str, 
    out_folder: str, 
    threads: int,
    overwrite: bool = False
) -> str:
    print("\n--- Generating pinhole cameras ---")
    os.makedirs(out_folder, exist_ok=True)
    
    frames = [os.path.splitext(os.path.basename(x))[0] for x in img_list]
    out_cam_list = [os.path.join(out_folder, f'{frame}.tsai') for frame in frames]
    if all([os.path.exists(x) for x in out_cam_list]) & (not overwrite):
        print("All pinhole cameras already exist in file, skipping.")
        return out_cam_list
    
    with rio.open(dem_file) as src:
        ht_datum = np.nanmean(src.read(1, masked=True))
    
    njobs, threads_per_job = setup_parallel_jobs(total_jobs=len(img_list), num_cpu=threads)
    
    job_list = []
    cam_list = img_list
    for img, cam, out_cam in zip(img_list, cam_list, out_cam_list):
        if os.path.exists(out_cam) & (not overwrite):
            continue

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

    # Get output cameras
    out_cam_list = [x for x in out_cam_list if os.path.exists(x)]

    return out_cam_list


def asp_pinhole_to_opencv_cameras(cam_list: list) -> pd.DataFrame:
    # This function is unchanged
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


# ==============================================================================
# FEATURE DETECTION AND MATCHING
# ==============================================================================

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
    img_list: list, 
    out_folder: str, 
    overlap_perc: float, 
    bh_ratio_range: tuple,
    true_stereo: bool, 
    utm_epsg: str, 
    write_basename: bool
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


def parse_image_specs(image_name: str) -> pd.DataFrame:
    # This function is unchanged
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
    def array_to_list_str(arr):
        return str(arr.tolist())

    # Apply the conversion to each array column
    for col in ['K', 'R', 't']:
        if col in df_to_save.columns:
            df_to_save[col] = df_to_save[col].apply(array_to_list_str)

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


def copy_match_files(match_files, out_folder, prefix="run-"):
    print(f"\nCopying {len(match_files)} match files to: {out_folder}")
    os.makedirs(out_folder, exist_ok=True)

    out_match_files = []
    for match_file in tqdm(match_files):
        image_pair_string = os.path.dirname(match_file).split("/")[-1]
        out_match_file = os.path.join(out_folder, prefix + image_pair_string + ".match")
        _ = shutil.copy2(match_file, out_match_file)
        out_match_files += [out_match_file]
    return out_match_files


def parse_match_files(match_files, threads=int(os.cpu_count() * 0.75), overwrite=False):
    print(f"\nConverting {len(match_files)} match files from binary to TXT using {threads} CPU.")
    def _process_match_file(match_file):
        # Check if output file exists
        out_file = match_file.replace(".match", "_match.txt")
        if os.path.exists(out_file) and (not overwrite):
            return out_file
        # Convert from binary to TXT
        log = run_cmd("parse_match_file.py", [match_file, out_file])
        # Reformat from the default setup
        # Get the total number of feature matches from the header line
        with open(out_file, 'r') as f:
            first_line = f.readline()
            n_matches = int(first_line.split()[0])
        # Read all pixel coordinates, skipping the header row
        init_matches = pd.read_csv(
            out_file, sep=" ", skiprows=1, usecols=[0,1], names=["px", "py"]
        )
        # Split pixel coordinates into img1 and img2
        matches_img1 = init_matches.iloc[0:n_matches].rename(
            columns={"px": "img1_px", "py": "img1_py"}
        )
        matches_img2 = init_matches.iloc[n_matches:].rename(
            columns={"px": "img2_px", "py": "img2_py"}
        ).reset_index(drop=True)     
        # Concatenate (N,4)   
        matches = pd.concat([matches_img1, matches_img2], axis=1)
        matches.to_csv(out_file, sep=" ", header=True, index=False)
        return out_file
    # Process match files in parallel
    out_files = p_map(_process_match_file, match_files, num_cpus=threads)
    return out_files


def load_match_files(match_folder: str, prefix: str = "run-") -> dict:
    print("\nLoading all feature matches...")
    
    match_files = glob(os.path.join(match_folder, "*_match.txt"))

    def _process_file(match_file: str) -> dict:
        img1_name, img2_name = (
            os.path.basename(match_file)
            .replace("_match.txt", "")
            .replace(prefix, "")
            .split('__')
        )
        matches = pd.read_csv(match_file, sep=" ", header=0)
        return {
            "img1": img1_name,
            "img2": img2_name,
            "pts1": np.array(matches[["img1_px", "img1_py"]].values),
            "pts2": np.array(matches[["img2_px", "img2_py"]].values)
        }

    mapped_dict = map(
        lambda t: (t[0], _process_file(t[1])),
        enumerate(match_files)
    )
    
    return dict(mapped_dict)


# ==============================================================================
# PYPOSE BUNDLE ADJUSTMENT
# ==============================================================================

class RigBundleAdjustmentModel(nn.Module):
    """
    A self-contained PyTorch module for rig-constrained bundle adjustment.
    
    This model holds all necessary information as parameters or buffers, and the
    forward pass only requires the observation data.
    """
    def __init__(self, base_poses, relative_poses, points, intrinsics, lookup_indices):
        super().__init__()
        # 1. Learnable Parameters: Tensors the optimizer is allowed to change.
        self.base_poses = pp.Parameter(base_poses)
        self.relative_poses = pp.Parameter(relative_poses)
        self.points = nn.Parameter(points)
        
        # 2. Static Buffers: Tensors that are part of the model's state but are not
        #    changed by the optimizer. They are moved to the GPU automatically.
        self.register_buffer('intrinsics', intrinsics)
        self.register_buffer('lookup_indices', lookup_indices)

    def forward(self, observations):
        """
        The simplified cost function. It now only takes the observation tensor as input.
        """
        cam_indices = observations[:, 0].long()
        pt_indices = observations[:, 1].long()
        pixels_measured = observations[:, 2:]

        # --- Use internal buffers for all lookups ---
        # Get the pre-computed base and relative pose indices for each camera.
        base_pose_indices = self.lookup_indices[cam_indices, 0]
        relative_pose_indices = self.lookup_indices[cam_indices, 1]
        
        selected_base_poses = self.base_poses[base_pose_indices]
        selected_points = self.points[pt_indices]
        selected_intrinsics = self.intrinsics[cam_indices]

        # --- Reconstruct final camera poses (This logic is now robust) ---
        # Create clones to ensure no inplace modification errors.
        R_cam_final = selected_base_poses.rotation().matrix().clone()
        t_cam_final = selected_base_poses.translation().clone()

        is_relative_mask = (relative_pose_indices != -1)
        if torch.any(is_relative_mask):
            rel_indices = relative_pose_indices[is_relative_mask]
            selected_relative_poses = self.relative_poses[rel_indices]
            
            R_base_masked = R_cam_final[is_relative_mask]
            t_base_masked = t_cam_final[is_relative_mask]
            
            R_rel = selected_relative_poses.rotation().matrix()
            t_rel = selected_relative_poses.translation()
            
            R_final_masked = R_base_masked @ R_rel
            t_final_masked = t_base_masked + pp.bmv(R_base_masked, t_rel)
            
            R_cam_final[is_relative_mask] = R_final_masked
            t_cam_final[is_relative_mask] = t_final_masked
        
        # --- Project points and calculate error ---
        points_in_cam_frame = pp.bmv(R_cam_final, selected_points) + t_cam_final
        projected_points = pp.bmv(selected_intrinsics, points_in_cam_frame)
        
        px = projected_points[..., 0] / projected_points[..., 2]
        py = projected_points[..., 1] / projected_points[..., 2]
        pixels_predicted = torch.stack([px, py], dim=-1)
        
        error = pixels_predicted - pixels_measured
        return error.flatten()



def plot_rig_comparison(initial_df, final_df, title="Change in Camera Pose (X/Y Plane)"):
    print("\n--- Plotting initial vs. final camera poses ---")

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

    # Add a colorbar for Z-drift
    cbar = fig.colorbar(quiver, shrink=0.5)
    cbar.set_label("Change in Z (m)")

    # Add start and end points
    ax.scatter(x_init, y_init, c='grey', marker='o', alpha=0.7, label='Initial Centers')
    ax.scatter(merged_df['x_final'], merged_df['y_final'], c='black', marker='x', alpha=0.9, label='Final Centers')

    # Formatting the plot
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    ax.set_aspect('equal')

    plt.show()

# ==============================================================================
# MAIN WORKFLOW
# ==============================================================================

def main():
    # --- 1. Configuration ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_folder = "/Users/rdcrlrka/Research/SkySat-Stereo/study-sites/MCS/20240420"
    epochs = 20
    threads = 10
    image_folder = os.path.join(data_folder, "SkySatScene")
    out_folder = os.path.join(data_folder, "pypose_rig_calibration")
    init_cam_folder = os.path.join(out_folder, "init_pinhole_cams")
    init_matches_folder = os.path.join(out_folder, "init_feature_matches")
    ba_folder = os.path.join(out_folder, "bundle_adjustment")

    print(f"Starting PyPose Bundle Adjustment. Using device: {device}")
    os.makedirs(ba_folder, exist_ok=True)

    # --- 2. Load All Initial Data ---
    image_list = sorted(glob(os.path.join(image_folder, "*_analytic.tif")))
    init_cam_list = sorted(glob(os.path.join(init_cam_folder, "*.tsai")))
    if not init_cam_list: 
        raise FileNotFoundError(f"No .tsai files in '{init_cam_folder}'. Please run `generate_pinhole_cameras` first.")
    
    cam_params_df = asp_pinhole_to_opencv_cameras(init_cam_list)
    specs_df = pd.concat(map(parse_image_specs, image_list), ignore_index=True)
    cam_specs_df = pd.merge(cam_params_df, specs_df, on="image_name").sort_values("image_name").reset_index(drop=True)
    cam_specs_df['cam_idx'] = cam_specs_df.index

    matches = load_match_files(init_matches_folder)
    if not matches: 
        raise ValueError("No feature matches found.")

    # --- 3. Create Initial Rig, Points, and Mappings ---
    ref_cam_id = cam_specs_df["cam"].value_counts().idxmax()
    initial_rig_poses = estimate_initial_rig_geometry(cam_specs_df, ref_cam_id)
    points_3d, observations = build_observation_data(matches, cam_specs_df)

    unique_base_keys = sorted(cam_specs_df[cam_specs_df['cam'] == ref_cam_id]['image_name'].unique())
    base_pose_map = {key: i for i, key in enumerate(unique_base_keys)}
    relative_cams = sorted([cam for cam in cam_specs_df['cam'].unique() if cam != ref_cam_id])
    relative_cam_map = {cam: i for i, cam in enumerate(relative_cams)}

    # Create the reference frames DataFrame
    ref_frames = cam_specs_df[cam_specs_df['cam'] == ref_cam_id].copy()
    cam_specs_df["ref_base_key"] = ""
    print("\nAssigning a reference base camera to each image...")
    for idx, row in tqdm(cam_specs_df.iterrows(), total=len(cam_specs_df)):
        if row['cam'] == ref_cam_id:
            cam_specs_df.at[idx, "ref_base_key"] = row['image_name']
        else:
            time_diffs = (ref_frames['datetime'] - row['datetime']).abs()
            closest_ref = ref_frames.loc[time_diffs.idxmin()]
            cam_specs_df.at[idx, "ref_base_key"] = closest_ref['image_name']
    if cam_specs_df['ref_base_key'].isnull().any() or (cam_specs_df['ref_base_key'] == "").any():
        print("WARNING: Some cameras were not assigned a base key!")
        print(cam_specs_df[cam_specs_df['ref_base_key'] == ""])

    # --- 4. Prepare Data for PyTorch/PyPose ---
    print("\nPreparing all data into PyTorch Tensors...")
    dtype = torch.float64
    device = device

    # Create the lookup tensor for the model's buffer
    # Each row `i` corresponds to camera `i`.
    # Column 0: index into the base_poses parameter tensor.
    # Column 1: index into the relative_poses parameter tensor (-1 if none).
    lookup_indices = torch.full((len(cam_specs_df), 2), -1, dtype=torch.long)
    for i, row in cam_specs_df.iterrows():
        lookup_indices[i, 0] = base_pose_map[row['ref_base_key']]
        if row['cam'] in relative_cam_map:
            lookup_indices[i, 1] = relative_cam_map[row['cam']]
    
    # Create the learnable parameters
    base_initial_poses_df = cam_specs_df[cam_specs_df['image_name'].isin(unique_base_keys)].set_index('image_name').loc[unique_base_keys]
    base_poses_list = [pp.from_matrix(torch.tensor(np.hstack([row['R'], row['t']]), dtype=dtype), ltype=pp.SE3_type) for _, row in base_initial_poses_df.iterrows()]
    batched_base_poses = torch.stack(base_poses_list)
    base_poses = pp.Parameter(pp.SE3(batched_base_poses).to(device))

    relative_poses_list = [pp.from_matrix(torch.tensor(np.hstack([initial_rig_poses[cam]['R'], initial_rig_poses[cam]['t']]), dtype=dtype), ltype=pp.SE3_type) for cam in relative_cams]
    batched_relative_poses = torch.stack(relative_poses_list)
    relative_poses = pp.Parameter(pp.SE3(batched_relative_poses).to(device))

    points = nn.Parameter(torch.tensor(points_3d, dtype=dtype, device=device))
    intrinsics = torch.tensor(np.stack(cam_specs_df['K'].values), dtype=dtype, device=device)
    
    obs_list = [[obs['cam_idx'], obs['pt_3d_idx'], obs['pt_2d'][0], obs['pt_2d'][1]] for obs in observations]
    observations_tensor = torch.tensor(obs_list, dtype=dtype, device=device)
    
    print(f"    Data prepared. Observations tensor shape: {observations_tensor.shape}")

    # --- 5. Run PyPose Optimization using the Scheduler ---
    # Initialize the model
    model = RigBundleAdjustmentModel(base_poses, relative_poses, points, intrinsics, lookup_indices).to(device)

    strategy = pp.optim.strategy.Constant(damping=1e-6)
    optimizer = pp.optim.LM(model, strategy=strategy, vectorize=False)

    scheduler = pp.optim.scheduler.StopOnPlateau(
        optimizer, steps=epochs, patience=3, decreasing=1e-5, verbose=True
    )

    print("\nStarting PyPose Levenberg-Marquardt Optimization with Scheduler...")
    
    # The 'target' for a least-squares problem is zero.
    target = torch.zeros(len(observations_tensor) * 2, device=device, dtype=dtype)
    
    # Run the full optimization with a single, clean command.
    scheduler.optimize(input=observations_tensor, target=target)

    # --- 6. Post-Hoc Alignment and Saving Results ---
    # (This section is now correct and should work as intended)
    print("\nPerforming post-hoc alignment and saving results...")
    
    final_base_poses_drifted = model.base_poses.detach().clone()
    final_relative_poses = model.relative_poses.detach().clone()
    final_points_drifted = model.points.detach().clone()

    initial_centroid = torch.mean(model.base_poses.data.translation(), dim=0) # Get original centroid
    final_centroid_drifted = torch.mean(final_base_poses_drifted.translation(), dim=0)
    shift_vector = initial_centroid - final_centroid_drifted
    
    print(f"    Alignment shift vector: {shift_vector.cpu().numpy()}")
    
    final_base_poses_aligned = final_base_poses_drifted.translation_add(shift_vector)
    
    final_cam_specs_df = cam_specs_df.copy()
    for i in range(len(final_cam_specs_df)):
        row = final_cam_specs_df.iloc[i]
        base_pose_idx = base_pose_map[row["ref_base_key"]]
        T_base = final_base_poses_aligned[base_pose_idx]
        
        if row['cam'] == ref_cam_id:
            T_final = T_base
        else:
            relative_pose_idx = relative_cam_map[row['cam']]
            T_rel = final_relative_poses[relative_pose_idx]
            T_final = T_base @ T_rel
        
        final_cam_specs_df.at[i, "R"] = T_final.rotation().matrix().cpu().numpy()
        final_cam_specs_df.at[i, "t"] = T_final.translation().unsqueeze(-1).cpu().numpy()

    # --- 7. Save and Visualize ---
    final_cam_specs_file = os.path.join(ba_folder, "pypose_final_cam_specs.csv")
    print(f"\\nFinal camera specs ready to be saved.")
    # save_cam_specs(final_cam_specs_df, final_cam_specs_file) 
    
    opencv_cameras_to_asp_pinhole(final_cam_specs_df, ba_folder)
    
    # plot_rig_comparison(cam_specs_df, final_cam_specs_df)



if __name__ == "__main__":
    main()


############################################

from torch import nn, tensor
from pypose.optim import LM
from pypose.optim.strategy import TrustRegion
from pypose.optim.scheduler import StopOnPlateau
import pypose as pp
import torch


class Residual(nn.Module):
    def __init__(self, cameras, points):
        super().__init__()
        cameras = pp.SE3(cameras)
        self.poses = nn.Parameter(cameras)
        self.points = nn.Parameter(points)
    def forward(self, observes, K, cidx, pidx):
        poses = self.poses[cidx]
        points = self.points[pidx]
        # FIX: Pass arguments in the correct order using keywords
        projs = pp.point2pixel(points=points, intrinsics=K, extrinsics=poses)
        
        return projs - observes


torch.set_default_device("cpu")
C, P, fx, fy, cx, cy = 1, 8, 200, 200, 100, 100
K = tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
cameras = pp.randn_SE3(C)
points = torch.randn(P, 3)
observes = torch.randn(P, 2)
cidx = torch.zeros(P, dtype=torch.int64)
pidx = torch.arange(P, dtype=torch.int64)

# The input tuple remains the same
input = (observes, K, cidx, pidx)

model = Residual(cameras, points)
strategy = TrustRegion()
optimizer = LM(model, strategy=strategy)
scheduler = StopOnPlateau(optimizer, steps=1)

s=0
print("Step\tLoss")
print("="*20)

while scheduler.continual():
    loss = optimizer.step(input)
    scheduler.step(loss)
    print(f"{s}\t{loss.item()}")
    s+=1


print("\nOptimization finished!")
print("Final Poses (SE3 LieTensor):\n", model.poses)