#! /usr/bin/env python

import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import coo_matrix
from datetime import datetime
import pandas as pd
from glob import glob
import os
import ast
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
from shapely.geometry import Point
import geopandas as gpd


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
    def _array_to_list_str(arr): return str(arr.tolist())
    for col in ['K', 'R', 't']:
        if col in df_to_save.columns: df_to_save[col] = df_to_save[col].apply(_array_to_list_str)
    df_to_save.to_csv(out_file, index=False)
    print(f"Saved camera specs to:\n{out_file}")
    return


def load_cam_specs(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename)
    def _str_to_array(s):
        try: return np.array(ast.literal_eval(s))
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing string to array: {s}. Error: {e}"); return np.array([])
    for col in ['K', 'R', 't']:
        if col in df.columns: df[col] = df[col].apply(_str_to_array)
    return df


def asp_pinhole_to_opencv_cameras(cam_list: list):
    print("\nConverting TSAI models to OpenCV format...")
    cam_params_list = []
    for cam_file in tqdm(cam_list, desc="Converting cameras"):
        with open(cam_file, "r") as c:
            lines = c.read().splitlines()
        
        params = {k.strip(): v.strip() for k, v in (line.split(' = ', 1) for line in lines if ' = ' in line)}

        # Intrinsics are correct
        K = np.array([
            [float(params.get('fu', 0)), 0, float(params.get('cu', 0))],
            [0, float(params.get('fv', 0)), float(params.get('cv', 0))],
            [0, 0, 1]
        ])

        # Load ASP's camera-to-world pose
        C_cam_center = np.fromstring(params.get('C', '0 0 0'), sep=' ').reshape(3, 1)
        R_cam_to_world = np.fromstring(params.get('R', '1 0 0 0 1 0 0 0 1'), sep=' ').reshape(3, 3)

        # The rotation for OpenCV is the transpose of ASP's R
        R_world_to_cam = R_cam_to_world.T
        
        # The translation for OpenCV is -R.T @ C
        t_world_to_cam = -R_world_to_cam @ C_cam_center

        # Add to list of cameras
        cam_params_list.append({
            "image_name": os.path.splitext(os.path.basename(cam_file))[0],
            "K": K,
            "R": R_world_to_cam,
            "t": t_world_to_cam 
        })    

    return pd.DataFrame(cam_params_list) 


def prepare_ba_data(cam_df: pd.DataFrame = None, anchor_triplet: bool = False, stereo_pairs_file: str = None):
    print("\nAugmenting DataFrame and preparing initial parameters...")
    df = cam_df.copy()

    # Select the reference camera
    ref_cam_id = df['cam'].value_counts().idxmax()
    other_cam_ids = sorted([c for c in df['cam'].unique() if c != ref_cam_id])
    K = df['K'].iloc[0]
    print(f"Selected '{ref_cam_id}' as reference. Using one shared K matrix.")

    # Identify rig instances (unique datetime/frame combinations)
    instance_tuples = [tuple(x) for x in df[['datetime', 'frame']].to_numpy()]
    unique_instances = sorted(list(set(instance_tuples)))
    instance_to_idx = {instance: i for i, instance in enumerate(unique_instances)}
    other_cam_id_to_rel_pose_idx = {name: i for i, name in enumerate(other_cam_ids)}

    # add info to the camera dataframe
    df['is_ref'] = (df['cam'] == ref_cam_id)
    df['instance_idx'] = df.apply(lambda row: instance_to_idx.get((row['datetime'], row['frame'])), axis=1)
    df['rel_pose_idx'] = df['cam'].map(other_cam_id_to_rel_pose_idx).fillna(-1).astype(int)

    # Select the anchor triplet if defined by user
    if anchor_triplet:
        print(f"Identifying the most-overlapping reference triplet pair to anchor during bundle adjust.")
        # Parse image basenames from stereo pairs file
        pairs_df = pd.read_csv(stereo_pairs_file, sep=' ')
        pairs_df['img1_name'] = pairs_df['img1'].apply(lambda x: os.path.basename(x).replace('.tif', ''))
        pairs_df['img2_name'] = pairs_df['img2'].apply(lambda x: os.path.basename(x).replace('.tif', ''))

        # Identify rows with the reference camera
        ref_cam_names = set(df[df['is_ref']]['image_name'])
        name_to_instance = df.set_index('image_name')['instance_idx']

        # Get valid pair candidates: both images captured by reference camera, different rig instances
        valid_pairs = pairs_df[
            pairs_df['img1_name'].isin(ref_cam_names) &
            pairs_df['img2_name'].isin(ref_cam_names)
        ].copy()
        valid_pairs['instance1'] = valid_pairs['img1_name'].map(name_to_instance)
        valid_pairs['instance2'] = valid_pairs['img2_name'].map(name_to_instance)
        valid_pairs = valid_pairs[valid_pairs['instance1'] != valid_pairs['instance2']]

        overlap_map = {
            frozenset([row.img1_name, row.img2_name]): row.overlap_percent
            for _, row in valid_pairs.iterrows()
        }

        # Identify the most overlapping valid triplet pair
        best_triplet = None
        max_overlap_mean = -1
        unique_ref_images = pd.unique(valid_pairs[['img1_name', 'img2_name']].values.ravel('K'))
        for triplet_candidate in combinations(unique_ref_images, 3):
            pair1 = frozenset(triplet_candidate[:2])
            pair2 = frozenset([triplet_candidate[0], triplet_candidate[2]])
            pair3 = frozenset(triplet_candidate[1:])
            if all(p in overlap_map for p in [pair1, pair2, pair3]):
                current_mean = np.mean([overlap_map[p] for p in [pair1, pair2, pair3]])
                if current_mean > max_overlap_mean:
                    max_overlap_mean, best_triplet = current_mean, triplet_candidate

        if not best_triplet:
            raise RuntimeError("Could not find a valid triplet of overlapping reference cameras from different instances.")

        fixed_pose_names = set(best_triplet)
        df['is_fixed_pose'] = df['image_name'].isin(fixed_pose_names)

        print(f"\nIdentified anchor triplet with mean overlap {np.round(max_overlap_mean,2)}%:")
        for name in fixed_pose_names:
            print(f"{name}")

    else:
        df['is_fixed_pose'] = False

    # Estimate initial camera poses for all cameras
    initial_camera_poses = np.array([
        np.hstack([cv2.Rodrigues(R)[0].flatten(), t.flatten()]) for R, t in zip(df['R'], df['t'])
    ])

    n_rig_instances = len(unique_instances)
    initial_ref_poses = np.zeros((n_rig_instances, 6))

    # Create a clean mapping from instance_idx to the corresponding camera's original index
    ref_cam_df = df[df['is_ref']]
    instance_to_cam_idx_map = pd.Series(ref_cam_df.index, index=ref_cam_df.instance_idx).to_dict()

    for i in range(n_rig_instances):
        if i in instance_to_cam_idx_map:
            # Use the map to find the correct camera index
            cam_idx = instance_to_cam_idx_map[i]
            initial_ref_poses[i] = initial_camera_poses[cam_idx]
        elif i > 0:
            # If a reference camera is missing for an instance, copy from the previous one
            initial_ref_poses[i] = initial_ref_poses[i-1]

    # Estimate initial relative camera poses / rig geometry
    initial_rig_geometry = estimate_initial_rig_geometry(df, ref_cam_id)
    initial_rig_poses = np.zeros((len(other_cam_ids), 6))
    for i, cam_id in enumerate(other_cam_ids):
        if cam_id in initial_rig_geometry:
            geom = initial_rig_geometry[cam_id]
            rvec, _ = cv2.Rodrigues(geom['R'])
            initial_rig_poses[i] = np.hstack([rvec.flatten(), geom['t'].flatten()])

    print(f"\nPrepared {len(df)} cameras across {n_rig_instances} unique instances.")
    return df, initial_ref_poses, initial_rig_poses, K


def estimate_initial_rig_geometry(cam_specs_df: pd.DataFrame, ref_cam_id: str) -> dict:
    print(f"\nEstimating initial rig geometry from camera poses relative to '{ref_cam_id}'")
    relative_poses = defaultdict(lambda: {'rotations': [], 'translations': []})
    other_cam_ids = [c for c in cam_specs_df['cam'].unique() if c != ref_cam_id]

    # Iterate over rig instances
    for (dt, frame), group in tqdm(cam_specs_df.groupby(['datetime', 'frame']), desc="Averaging Poses"):
        try:
            ref_cam = group[group['cam'] == ref_cam_id].iloc[0]
            R_ref, t_ref = ref_cam['R'], ref_cam['t']
            for other_cam_id in other_cam_ids:
                if other_cam_id in group['cam'].values:
                    other_cam = group[group['cam'] == other_cam_id].iloc[0]
                    R_other, t_other = other_cam['R'], other_cam['t']
                    relative_R, relative_t = R_ref.T @ R_other, R_ref.T @ (t_other - t_ref)
                    relative_poses[other_cam_id]['rotations'].append(relative_R)
                    relative_poses[other_cam_id]['translations'].append(relative_t)
        except (IndexError, KeyError):
            print(f"Warning: Ref camera '{ref_cam_id}' not found for instance ({dt}, {frame}). Skipping.")
            
    # Averaging the rig poses for initial guess
    initial_rig_geometry = {}
    print("Averaging measurements to get final rig geometry...")
    for cam_id, poses in relative_poses.items():
        if not poses['rotations']: continue
        quats = Rotation.from_matrix(poses['rotations']).as_quat()
        avg_quat = np.mean(quats, axis=0) if quats.ndim > 1 else quats
        avg_R = Rotation.from_quat(avg_quat / np.linalg.norm(avg_quat)).as_matrix()
        avg_t = np.mean(np.array(poses['translations']), axis=0).reshape(3, 1)
        initial_rig_geometry[cam_id] = {'R': avg_R, 't': avg_t}

    return initial_rig_geometry


def project_points(points_3d, K, pose):
    rvec, tvec = pose[:3], pose[3:]
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, K, distCoeffs=None)
    return points_2d.reshape(-1, 2)


def bundle_adjustment_residuals(params, K, cam_df, camera_indices, point_indices, observed_pixels,
                               fixed_ref_poses, instance_idx_to_variable_idx, fixed_instance_indices,
                               n_variable_ref_poses, n_points, n_rig_rel_poses,
                               initial_variable_ref_poses, initial_rig_poses, initial_points_3d,
                               pose_regularization_weight, tri_regularization_weight):
    
    # --- Unpack the parameter vector ---
    variable_ref_poses = params[:n_variable_ref_poses * 6].reshape((n_variable_ref_poses, 6))
    
    point_start = n_variable_ref_poses * 6
    points_3d = params[point_start : point_start + n_points * 3].reshape((n_points, 3))
    
    rig_start = point_start + n_points * 3
    rig_relative_poses = params[rig_start : rig_start + n_rig_rel_poses * 6].reshape((n_rig_rel_poses, 6))

    # --- Calculate reprojection error ---
    # Reconstruct the full list of reference poses
    n_rig_instances = len(cam_df['instance_idx'].unique())
    full_ref_poses = np.zeros((n_rig_instances, 6))
    
    for i in range(n_rig_instances):
        if i in fixed_instance_indices:
            fixed_idx = np.where(fixed_instance_indices == i)[0][0]
            full_ref_poses[i] = fixed_ref_poses[fixed_idx]
        else:
            variable_idx = instance_idx_to_variable_idx[i]
            full_ref_poses[i] = variable_ref_poses[variable_idx]
            
    # Calculate all camera poses
    all_camera_poses = np.zeros((len(cam_df), 6))
    for cam_idx, cam_row in cam_df.iterrows():
        instance_idx, is_ref, rel_pose_idx = cam_row.instance_idx, cam_row.is_ref, cam_row.rel_pose_idx
        ref_pose = full_ref_poses[instance_idx]
        R_world_ref, t_world_ref = Rotation.from_rotvec(ref_pose[:3]), ref_pose[3:]
        if is_ref:
            all_camera_poses[cam_idx] = ref_pose
        else:
            rel_pose = rig_relative_poses[rel_pose_idx]
            R_rig_canonical, t_rig_canonical = Rotation.from_rotvec(rel_pose[:3]), rel_pose[3:]
            R_world_other, t_world_other = R_world_ref * R_rig_canonical, t_world_ref + R_world_ref.apply(t_rig_canonical)
            all_camera_poses[cam_idx, :3], all_camera_poses[cam_idx, 3:] = R_world_other.as_rotvec(), t_world_other
    
    # Calculate reprojection errors
    reprojection_residuals = [(project_points(points_3d[point_idx].reshape(1,3), K, all_camera_poses[cam_idx]).flatten() - observed_pixels[i]) 
                              for i, (cam_idx, _, point_idx) in enumerate(zip(camera_indices, observed_pixels, point_indices))]
    
    # --- Calculate regularization residuals ---
    # Penalty for deviation from initial variable reference poses
    variable_ref_pose_residuals = (variable_ref_poses - initial_variable_ref_poses) * pose_regularization_weight
    
    # Penalty for deviation from initial relative rig poses
    rig_pose_residuals = (rig_relative_poses - initial_rig_poses) * pose_regularization_weight

    # Penalty for deviation from initial 3D point locations
    point_residuals = (points_3d - initial_points_3d) * tri_regularization_weight

    # --- Combine all residuals into one vector ---
    return np.concatenate([
        np.concatenate(reprojection_residuals),
        variable_ref_pose_residuals.flatten(),
        rig_pose_residuals.flatten(),
        point_residuals.flatten()
    ])


def create_sparsity_matrix_vectorized(n_points, cam_df, camera_indices, point_indices,
                                      instance_idx_to_variable_idx, fixed_instance_indices,
                                      n_variable_ref_poses, n_rig_rel_poses):
    n_obs = len(camera_indices)
    obs_instance_indices = cam_df['instance_idx'].to_numpy()[camera_indices]
    obs_is_ref = cam_df['is_ref'].to_numpy()[camera_indices]
    obs_rel_pose_indices = cam_df['rel_pose_idx'].to_numpy()[camera_indices]
    
    all_rows, all_cols = [], []

    # --- Jacobian for reprojection residuals ---
    row_indices_base = np.arange(n_obs) * 2

    # Point Dependency
    point_param_start = 6 * n_variable_ref_poses
    rows_for_points = np.repeat(row_indices_base, 6).reshape(-1, 6) 
    rows_for_points[:, 3:] += 1
    cols_for_points = point_param_start + 3 * np.repeat(point_indices, 6) + np.tile([0,1,2,0,1,2], n_obs)
    all_rows.append(rows_for_points.flatten()); all_cols.append(cols_for_points)

    # Reference camera poses (reference cam only)
    is_variable_mask = np.isin(obs_instance_indices, list(instance_idx_to_variable_idx.keys()))
    variable_obs_indices = np.where(is_variable_mask)[0]
    if len(variable_obs_indices) > 0:
        rows_for_ref_cam = np.repeat(row_indices_base[variable_obs_indices], 12).reshape(-1, 12); rows_for_ref_cam[:, 6:] += 1
        variable_instance_col_indices = np.array([instance_idx_to_variable_idx[i] for i in obs_instance_indices[variable_obs_indices]])
        cols_for_ref_cam = 6 * np.repeat(variable_instance_col_indices, 12) + np.tile(list(range(6))*2, len(variable_obs_indices))
        all_rows.append(rows_for_ref_cam.flatten()); all_cols.append(cols_for_ref_cam)

    # Relative camera poses (non-reference cams only)
    non_ref_mask = ~obs_is_ref
    if np.any(non_ref_mask):
        non_ref_obs_indices = np.where(non_ref_mask)[0]
        rows_for_rel_pose = np.repeat(row_indices_base[non_ref_obs_indices], 12).reshape(-1, 12); rows_for_rel_pose[:, 6:] += 1
        rig_param_start = 6 * n_variable_ref_poses + 3 * n_points
        relevant_rel_pose_indices = obs_rel_pose_indices[non_ref_mask]
        cols_for_rel_pose = rig_param_start + 6 * np.repeat(relevant_rel_pose_indices, 12) + np.tile(list(range(6))*2, len(non_ref_obs_indices))
        all_rows.append(rows_for_rel_pose.flatten()); all_cols.append(cols_for_rel_pose)

    # --- Jacobian for regularization residuals ---
    row_offset = 2 * n_obs
    
    n_var_ref_pose_params = 6 * n_variable_ref_poses
    rows_for_var_reg = row_offset + np.arange(n_var_ref_pose_params)
    cols_for_var_reg = np.arange(n_var_ref_pose_params)
    all_rows.append(rows_for_var_reg); all_cols.append(cols_for_var_reg)
    row_offset += n_var_ref_pose_params

    n_rig_pose_params = 6 * n_rig_rel_poses
    rows_for_rig_reg = row_offset + np.arange(n_rig_pose_params)
    cols_for_rig_reg = (6 * n_variable_ref_poses + 3 * n_points) + np.arange(n_rig_pose_params)
    all_rows.append(rows_for_rig_reg); all_cols.append(cols_for_rig_reg)
    
    row_offset += n_rig_pose_params

    # --- Jacobian for 3D Point Regularization ---
    n_point_params = 3 * n_points
    rows_for_point_reg = row_offset + np.arange(n_point_params)
    cols_for_point_reg = (6 * n_variable_ref_poses) + np.arange(n_point_params)
    all_rows.append(rows_for_point_reg)
    all_cols.append(cols_for_point_reg)

    # --- Assemble final matrix ---
    final_rows = np.concatenate(all_rows)
    final_cols = np.concatenate(all_cols)
    data = np.ones_like(final_rows, dtype=int)

    n_rows = 2 * len(camera_indices) + n_var_ref_pose_params + n_rig_pose_params + n_point_params
    n_cols = 6 * n_variable_ref_poses + 3 * n_points + 6 * n_rig_rel_poses
    
    return coo_matrix((data, (final_rows, final_cols)), shape=(n_rows, n_cols))


def build_initial_observations(
    match_files: list,
    K: np.ndarray,
    cam_df: pd.DataFrame,
    max_points_per_pair: int,
    max_initial_reproj_error: float = None,
    local_origin: np.ndarray = None
):
    print(f"Found {len(match_files)} feature match files. Loading and triangulating 3D points...")
    observations, initial_points_3d, point_counter = [], [], 0
    image_name_to_idx = pd.Series(cam_df.index, index=cam_df.image_name)

    # Iterate over match files
    for i, match_file in tqdm(enumerate(match_files), total=len(match_files), desc="Triangulating"):
        # parse image and camera names
        img1_name, img2_name = os.path.splitext(os.path.basename(match_file))[0].replace("_match", "").split('__')
        if not all(name in image_name_to_idx for name in [img1_name, img2_name]):
            continue
        cam1_idx, cam2_idx = image_name_to_idx[img1_name], image_name_to_idx[img2_name]
        if cam_df.loc[cam1_idx, 'instance_idx'] == cam_df.loc[cam2_idx, 'instance_idx']:
            continue

        # get camera projection matrices
        R1, t1 = cam_df.loc[cam1_idx, ['R', 't']]
        P1 = K @ np.hstack((R1, t1.reshape(3, 1)))
        R2, t2 = cam_df.loc[cam2_idx, ['R', 't']]
        P2 = K @ np.hstack((R2, t2.reshape(3, 1)))

        # iterate over matches for pair
        match_df = pd.read_csv(match_file, sep=" ", header=0)
        potential_points_for_pair = []
        for _, row in match_df.iterrows():
            # triangulate points
            px1, px2 = np.array([row['img1_px'], row['img1_py']]), np.array([row['img2_px'], row['img2_py']])
            point_4d_hom = cv2.triangulatePoints(P1, P2, px1, px2)
            w = point_4d_hom[3, 0]
            if np.isclose(w, 0):
                continue
            
            point_3d_orig = (point_4d_hom[:3] / w).flatten()
            if not np.all(np.isfinite(point_3d_orig)):
                continue

            # Center the triangulated point around the local origin if needed
            if local_origin is not None:
                point_3d = point_3d_orig - local_origin.flatten()
            else:
                point_3d = point_3d_orig

            # Calculate initial reprojection error
            rvec1, _ = cv2.Rodrigues(R1)
            rvec2, _ = cv2.Rodrigues(R2)
            px1_reproj, _ = cv2.projectPoints(point_3d.reshape(1, 3), rvec1, t1, K, None)
            px2_reproj, _ = cv2.projectPoints(point_3d.reshape(1, 3), rvec2, t2, K, None)
            error1 = np.linalg.norm(px1 - px1_reproj.flatten())
            error2 = np.linalg.norm(px2 - px2_reproj.flatten())

            # Filter for max initial error
            if (error1 + error2) / 2 > max_initial_reproj_error:
                continue

            potential_points_for_pair.append({
                'p3d': point_3d,
                'err': (error1 + error2) / 2.0,
                'obs1': (cam1_idx, px1),
                'obs2': (cam2_idx, px2)
            })

        if not potential_points_for_pair:
            continue

        # Save max_points_per_pair points with the lowest initial reprojection error
        potential_points_for_pair.sort(key=lambda p: p['err'])
        for point_data in potential_points_for_pair[:max_points_per_pair]:
            initial_points_3d.append(point_data['p3d'])
            observations.append(point_data['obs1'] + (point_counter,))
            observations.append(point_data['obs2'] + (point_counter,))
            point_counter += 1

    if not initial_points_3d:
        raise RuntimeError("Failed to generate any valid 3D points.")

    print(f"Generated {len(initial_points_3d)} valid initial 3D points from {len(observations)} observations.")
    return np.array(initial_points_3d), observations


def center_cameras(cams_df):
    print("\nCentering camera coordinates")
    all_t_vectors = np.array(list(cams_df['t']))
    local_origin = np.mean(all_t_vectors, axis=0)
    print(f"Calculated scene centroid (local origin) at: {local_origin.ravel()}")
    centered_cam_df = cams_df.copy()
    centered_cam_df['t'] = centered_cam_df['t'].apply(lambda t: t - local_origin)
    return centered_cam_df, local_origin
    
def opencv_cameras_to_asp_pinhole(cam_specs_df: pd.DataFrame, out_folder: str):
    """
    Correctly converts a DataFrame with OpenCV-compatible extrinsics
    back to ASP .tsai pinhole models.
    """
    print(f"\n--- Converting final camera models to ASP TSAI format in '{out_folder}' ---")
    os.makedirs(out_folder, exist_ok=True)
    
    for _, row in tqdm(cam_specs_df.iterrows(), total=len(cam_specs_df), desc="Saving cameras"):
        # The K, R, and t from the DataFrame are OpenCV-compatible (world-to-camera)
        K, R_world_to_cam, t_world_to_cam, image_name = row['K'], row['R'], row['t'], row['image_name']

        # The Rotation for ASP is the transpose of OpenCV's R
        R_cam_to_world = R_world_to_cam.T

        # The Camera Center for ASP is -R.T @ t
        C_cam_center = -R_cam_to_world @ t_world_to_cam
        
        # Prepare strings for saving
        C_str = ' '.join(map(str, C_cam_center.ravel()))
        R_str = ' '.join(map(str, R_cam_to_world.ravel()))
        fu, fv, cu, cv = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        
        output_filename = os.path.join(out_folder, f"{image_name}.tsai")
        with open(output_filename, 'w') as f:
            f.write(
                f"VERSION_4\n"
                f"PINHOLE\n"
                f"fu = {fu}\n"
                f"fv = {fv}\n"
                f"cu = {cu}\n"
                f"cv = {cv}\n"
                f"u_direction = 1 0 0\n"
                f"v_direction = 0 1 0\n"
                f"w_direction = 0 0 1\n"
                f"C = {C_str}\n"
                f"R = {R_str}\n"
                f"pitch = 0.8\n"
                f"NULL\n"
            )

    print(f"\nSaved {len(cam_specs_df)} camera models to:\n{out_folder}")


def analyze_and_save_residuals(
    residuals: np.ndarray,
    camera_indices: np.ndarray,
    cam_df: pd.DataFrame,
    output_file: str
):
    # The reprojection residuals are at the beginning of the full residual vector.
    num_reproj_residuals = len(camera_indices) * 2
    reproj_residuals = residuals[:num_reproj_residuals].reshape(-1, 2)

    # Calculate per-observation error magnitude (Euclidean distance)
    per_obs_error = np.linalg.norm(reproj_residuals, axis=1)
    
    # --- Overall Statistics ---
    rms = np.sqrt(np.mean(per_obs_error**2))
    mean_err = np.mean(per_obs_error)
    median_err = np.median(per_obs_error)

    print(f"Overall RMS Error:    {rms:.4f} pixels")
    print(f"Overall Mean Error:   {mean_err:.4f} pixels")
    print(f"Overall Median Error: {median_err:.4f} pixels")

    # --- Per-Camera Statistics ---
    error_df = pd.DataFrame({
        'cam_idx': camera_indices,
        'error': per_obs_error
    })
    
    # Map camera index to image name
    idx_to_name = cam_df['image_name']
    error_df['image_name'] = error_df['cam_idx'].map(idx_to_name)
    
    # Group by image name and calculate stats
    per_camera_stats = error_df.groupby('image_name')['error'].agg(['mean', 'median', 'count']).reset_index()
    per_camera_stats = per_camera_stats.sort_values(by='mean', ascending=False)

    # --- Save to File ---
    per_camera_stats.to_csv(output_file, index=False, float_format='%.4f')
    print(f"Saved per-camera error statistics to: {output_file}")

    return reproj_residuals, per_camera_stats


def plot_camera_positions(plot_df, fig_file):
    """
    Plots the initial and final camera center positions (top-down XY view).
    Both input dataframes are expected to be in the same centered coordinate system.
    """
    print("\nPlotting initial vs. final camera positions in local coordinates...")

    # Extract XY coordinates
    initial_xy = np.array([t[:2].flatten() for t in plot_df['t_initial']])
    final_xy = np.array([t[:2].flatten() for t in plot_df['t_final']])

    fig = plt.figure(figsize=(10, 6))
    
    # Plot initial and final positions
    plt.scatter(initial_xy[:, 0], initial_xy[:, 1], c='blue', alpha=0.6, label='Initial Positions', s=40)
    plt.scatter(final_xy[:, 0], final_xy[:, 1], c='red', marker='x', label='Final Positions', s=40)
    
    # Draw lines connecting initial to final positions
    for i in range(len(initial_xy)):
        plt.plot([initial_xy[i, 0], final_xy[i, 0]], 
                 [initial_xy[i, 1], final_xy[i, 1]], 
                 'k-', alpha=0.3, linewidth=0.7)
    
    plt.xlabel("X Position (meters, local centered frame)")
    plt.ylabel("Y Position (meters, local centered frame)")
    plt.title("Initial vs. Final Camera Positions (Top-Down View)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    if fig_file:
        fig.savefig(fig_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Figure of initial vs. final camera positions saved to:\n{fig_file}")
    else:
        plt.show()

    return

def save_pointmap_stats(
    points_3d: np.ndarray,
    reproj_residuals: np.ndarray,
    point_indices: np.ndarray,
    output_file: str,
    plot_results: bool = True,
    fig_file: str = None
):
    # Calculate error magnitudes per observation
    errors = np.linalg.norm(reproj_residuals, axis=1)

    # Compile in a DataFrame
    point_error_df = pd.DataFrame({
        'point_id': point_indices,
        'reproj_error': errors,
    })

    # Group by point_id and calculate the mean error for each point
    mean_errors_per_point = point_error_df.groupby('point_id').mean().reset_index()

    # Create the output DataFrame with coordinates
    point_coords_df = pd.DataFrame({
        'point_id': np.arange(len(points_3d)),
        'x': points_3d[:, 0],
        'y': points_3d[:, 1],
        'z': points_3d[:, 2],
    })

    # Merge coordinates with errors
    final_df = pd.merge(point_coords_df, mean_errors_per_point, on='point_id')

    # Reproject to WGS84
    final_df["geometry"] = [Point(x,y,z) for x,y,z in final_df[["x", "y", "z"]].values]
    final_gdf = gpd.GeoDataFrame(final_df, geometry=final_df["geometry"], crs="EPSG:4978")
    final_gdf = final_gdf.to_crs("EPSG:4326")
    
    # Reorder columns for clarity
    final_df = final_gdf[['point_id', 'x', 'y', 'z', 'reproj_error']]

    # Save to CSV
    final_df.to_csv(output_file, index=False, float_format='%.6f')
    print(f"\nSaved 3d points and reprojection errors to: {output_file}")

    # Plot
    if plot_results:
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        im = ax.scatter(final_df["x"], final_df["y"], c=final_df["reproj_error"], s=2, cmap="seismic")
        fig.colorbar(im, ax=ax, shrink=0.5, label="Reprojection error")
        if not fig_file:
            fig_file = os.path.splitext(output_file)[0] + ".png"
        fig.savefig(fig_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"\nSaved residuals pointmap to: {fig_file}")

    return


# ==============================================================================
# MAIN SCRIPT
# ==============================================================================

def main():
    # ==========================================================================
    # --- 0. User Inputs and Setup ---
    # ==========================================================================
    # --- Input paths in directory ---
    DATA_DIR = "/Users/rdcrlrka/Research/SkySat-Stereo/study-sites/MCS/20240420/custom_rig_calibration"
    INIT_CAMS_DIR = os.path.join(DATA_DIR, "init_pinhole_cams")
    FEATURE_MATCH_DIR = os.path.join(DATA_DIR, "init_feature_matches")
    STEREO_PAIRS_FILE = os.path.join(DATA_DIR, "stereo_pairs.txt")

    image_files = sorted(glob(os.path.join(DATA_DIR, "..", "SkySatScene", "*_analytic.tif")))
    init_cam_files = sorted(glob(os.path.join(INIT_CAMS_DIR, "*.tsai")))
    match_files = sorted(glob(os.path.join(FEATURE_MATCH_DIR, "*_match.txt")))
    print(f"Located {len(image_files)} images, {len(init_cam_files)} cameras, and {len(match_files)} match files.")

    # --- Optimization hyperparameters ---
    ANCHOR_TRIPLET = False
    CENTER_POINTS = False
    MAX_POINTS_PER_PAIR = 500
    MAX_INITIAL_REPROJ_ERROR = 3e3
    FTOL = 1e-6
    POSE_REGULARIZATION_WEIGHT = 0.0
    TRI_POINT_REGULARIZATION_WEIGHT = 0.1
    THREADS = 10

    # --- Output files ---
    BA_DIR = os.path.join(DATA_DIR, "bundle_adjust")
    os.makedirs(BA_DIR, exist_ok=True)
    initial_cameras_file = os.path.join(BA_DIR, "initial_camera_specs.csv")
    initial_residuals_file = os.path.join(BA_DIR, "initial_reprojection_errors.csv")
    initial_pointmap_file = os.path.join(BA_DIR, "initial_pointmap_reprojection_errors.csv")
    final_residuals_file = os.path.join(BA_DIR, "final_reprojection_errors.csv")
    final_pointmap_file = os.path.join(BA_DIR, "final_pointmap_reprojection_errors.csv")
    cam_pos_fig_file = os.path.join(BA_DIR, "initial_vs_final_cam_positions.png")

    # ==========================================================================
    # --- 1. Load and Prepare Data ---
    # ==========================================================================
    print("\n--- Step 1: Loading and preparing camera data ---")
    # Convert to OpenCV-compatible camera params and merge with image metadata
    cam_params_df = asp_pinhole_to_opencv_cameras(init_cam_files)
    specs_df = pd.concat(map(parse_image_specs, image_files), ignore_index=True)
    orig_cam_df = pd.merge(cam_params_df, specs_df, on="image_name")
    orig_cam_df['image_name'] = orig_cam_df['image_name'].str.replace('.tif', '', regex=False)
    # save to file
    save_cam_specs(orig_cam_df, initial_cameras_file)

    # Center points
    if CENTER_POINTS:
        centered_cam_df, local_origin = center_cameras(orig_cam_df)
    else:
        centered_cam_df, local_origin = orig_cam_df.copy(), None

    # Prepare bundle adjust data
    cam_df, initial_ref_poses, initial_rig_poses, K = prepare_ba_data(
        centered_cam_df, ANCHOR_TRIPLET, STEREO_PAIRS_FILE
    )

    # Build initial observations from feature matches
    initial_points_3d, observations = build_initial_observations(
        match_files, K, cam_df, MAX_POINTS_PER_PAIR, MAX_INITIAL_REPROJ_ERROR, local_origin
    )

    # ==========================================================================
    # --- 2. Partition Parameters for Optimization ---
    # ==========================================================================
    print("\n--- Step 2: Partitioning parameters into fixed and variable sets ---")
    fixed_instance_indices = np.sort(cam_df[cam_df['is_fixed_pose']]['instance_idx'].unique())
    all_instance_indices = np.sort(cam_df['instance_idx'].unique())
    variable_instance_indices = np.setdiff1d(all_instance_indices, fixed_instance_indices, assume_unique=True)
    instance_idx_to_variable_idx = {original_idx: new_idx for new_idx, original_idx in enumerate(variable_instance_indices)}

    fixed_ref_poses = initial_ref_poses[fixed_instance_indices]
    variable_ref_poses = initial_ref_poses[variable_instance_indices]

    n_variable_ref_poses, n_points, n_rig_rel_poses = len(variable_ref_poses), len(initial_points_3d), len(initial_rig_poses)
    print(f"Total reference poses: {len(all_instance_indices)}")
    print(f"Fixed poses: {len(fixed_ref_poses)} (Instance IDs: {fixed_instance_indices})")
    print(f"Variable poses: {n_variable_ref_poses}")

    # --- Assemble the parameter vector for the optimizer ---
    params_0 = np.concatenate([variable_ref_poses.flatten(), initial_points_3d.flatten(), initial_rig_poses.flatten()])
    camera_indices = np.array([o[0] for o in observations])
    point_indices = np.array([o[2] for o in observations])
    observed_pixels = np.array([o[1] for o in observations])

    # --- Build the Jacobian sparsity structure ---
    sparsity_matrix = create_sparsity_matrix_vectorized(
        n_points, cam_df, camera_indices, point_indices,
        instance_idx_to_variable_idx, fixed_instance_indices,
        n_variable_ref_poses, n_rig_rel_poses
    )

    # ==========================================================================
    # --- 3. Save Initial State Stats ---
    # ==========================================================================
    print("\n--- Step 3: Analyzing and saving initial state stats ---")
    # Calculate and save initial residuals by camera
    initial_residuals_full = bundle_adjustment_residuals(
        params_0, K, cam_df, camera_indices, point_indices, observed_pixels,
        fixed_ref_poses, instance_idx_to_variable_idx, fixed_instance_indices,
        n_variable_ref_poses, n_points, n_rig_rel_poses,
        variable_ref_poses, initial_rig_poses, initial_points_3d,
        POSE_REGULARIZATION_WEIGHT, TRI_POINT_REGULARIZATION_WEIGHT
    )
    initial_reproj_residuals, _ = analyze_and_save_residuals(
        initial_residuals_full, camera_indices, cam_df, initial_residuals_file
    )

    # Calculate and save initial pointmap residual stats
    save_pointmap_stats(
        initial_points_3d,
        initial_reproj_residuals,
        point_indices,
        initial_pointmap_file
    )

    # ==========================================================================
    # --- 4. Run Optimization ---
    # ==========================================================================
    print("\n--- Step 4: Starting the SciPy least_squares optimizer ---")
    res = least_squares(
        fun=bundle_adjustment_residuals, 
        x0=params_0, 
        jac_sparsity=sparsity_matrix,
        loss='cauchy', 
        f_scale=0.5, 
        verbose=2, 
        x_scale='jac', 
        ftol=FTOL, 
        method='trf', 
        workers=THREADS,
        args=(K, cam_df, camera_indices, point_indices, observed_pixels,
              fixed_ref_poses, instance_idx_to_variable_idx, fixed_instance_indices,
              n_variable_ref_poses, n_points, n_rig_rel_poses,
              variable_ref_poses, initial_rig_poses, initial_points_3d,
              POSE_REGULARIZATION_WEIGHT, TRI_POINT_REGULARIZATION_WEIGHT)
    )

    # ==========================================================================
    # --- 5. Save Final State Stats ---
    # ==========================================================================
    print("\n--- Step 5: Analyzing and saving final state stats ---")
    # Calculate final residuals by camera
    final_reproj_residuals, _ = analyze_and_save_residuals(
        res.fun, camera_indices, cam_df, final_residuals_file
    )

    # Save pointmap stats
    point_start_idx = n_variable_ref_poses * 6
    final_points_3d = res.x[point_start_idx : point_start_idx + n_points * 3].reshape((n_points, 3))
    save_pointmap_stats(
        final_points_3d,
        final_reproj_residuals,
        point_indices,
        final_pointmap_file
    )

    # ==========================================================================
    # --- 6. Reconstruct Final Poses and Save ---
    # ==========================================================================
    print("\n--- Step 6: Reconstructing and saving final camera models ---")
    optimized_variable_poses = res.x[:n_variable_ref_poses * 6].reshape((n_variable_ref_poses, 6))
    final_ref_poses_local = np.zeros_like(initial_ref_poses)
    final_ref_poses_local[fixed_instance_indices] = fixed_ref_poses
    final_ref_poses_local[variable_instance_indices] = optimized_variable_poses

    rig_params_offset = n_variable_ref_poses * 6 + n_points * 3
    final_rig_poses = res.x[rig_params_offset : rig_params_offset + n_rig_rel_poses * 6].reshape((n_rig_rel_poses, 6))
    final_cam_data = []
    for cam_idx, cam_row in cam_df.iterrows():
        ref_pose_local = final_ref_poses_local[cam_row.instance_idx]
        if cam_row.is_ref:
            final_pose_vec = ref_pose_local
        else:
            rel_pose_opt = final_rig_poses[cam_row.rel_pose_idx]
            R_world_ref = Rotation.from_rotvec(ref_pose_local[:3])
            R_rig_rel = Rotation.from_rotvec(rel_pose_opt[:3])
            R_final = R_world_ref * R_rig_rel
            t_final = R_world_ref.apply(rel_pose_opt[3:]) + ref_pose_local[3:]
            final_pose_vec = np.hstack([R_final.as_rotvec(), t_final])

        final_R_local, _ = cv2.Rodrigues(final_pose_vec[:3])
        final_t_local = final_pose_vec[3:].reshape(3,1)

        # Uncenter coords if needed (local -> global origin)
        if local_origin is not None:
            final_t_global = final_t_local + local_origin.reshape(3,1)
        else:
            final_t_global = final_t_local

        final_cam_data.append({
            'image_name': cam_row.image_name, 'K': K, 'R': final_R_local,
            't_global': final_t_global, 't_local': final_t_local
        })

    final_df_global = pd.DataFrame(final_cam_data)
    final_df_global['t'] = final_df_global['t_global']
    opencv_cameras_to_asp_pinhole(final_df_global, BA_DIR)

    # ==========================================================================
    # --- 7. Plot Final Camera Positions ---
    # ==========================================================================
    final_df_centered = pd.DataFrame(final_cam_data)
    final_df_centered['t'] = final_df_centered['t_local']

    plot_df = pd.merge(centered_cam_df, final_df_centered, on="image_name", suffixes=("_initial", "_final"))
    plot_camera_positions(plot_df, cam_pos_fig_file)

    

if __name__ == "__main__":
    main()