import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.sparse import coo_matrix, lil_matrix
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
import ast
from tqdm import tqdm
from collections import defaultdict

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def estimate_initial_rig_geometry(
        cam_specs_df: pd.DataFrame, 
        ref_cam_id: str
        ) -> dict:
    print(f"\nEstimating initial rig geometry from camera poses relative to '{ref_cam_id}'")
    relative_poses = defaultdict(lambda: {'rotations': [], 'translations': []})
    
    # Identify all other camera IDs that need a relative pose
    other_cam_ids = [c for c in cam_specs_df['cam'].unique() if c != ref_cam_id]

    # Group images by frame
    for frame, group in tqdm(cam_specs_df.groupby('frame'), desc="Averaging Poses"):
        try:
            # Find the reference camera for this frame group
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
            print(f"Warning: Reference camera '{ref_cam_id}' not found at frame {frame}. Skipping.")

    final_rig_geometry = {}
    print("Averaging measurements to get final rig geometry...")
    for cam_id, poses in relative_poses.items():
        if not poses['rotations']: continue

        # This quaternion averaging is a very robust method!
        quats = Rotation.from_matrix(poses['rotations']).as_quat()
        if quats.ndim == 1:
            avg_quat = quats
        else:
            avg_quat = np.mean(quats, axis=0)
            
        avg_R = Rotation.from_quat(avg_quat / np.linalg.norm(avg_quat)).as_matrix()
        avg_t = np.mean(np.array(poses['translations']), axis=0).reshape(3, 1)
        final_rig_geometry[cam_id] = {'R': avg_R, 't': avg_t}
    
    return final_rig_geometry

def project_points(points_3d, K, pose):
    rvec, tvec = pose[:3], pose[3:]
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, K, distCoeffs=None)
    return points_2d.reshape(-1, 2)

def bundle_adjustment_residuals(params, n_cameras, n_points, n_rig_rel_poses, camera_indices, point_indices, observed_pixels, K_map, rig_config, rig_def, cam_info):
    n_ref_cameras = len(rig_config)
    ref_cam_poses = params[:n_ref_cameras * 6].reshape((n_ref_cameras, 6))
    points_3d = params[n_ref_cameras * 6 : n_ref_cameras * 6 + n_points * 3].reshape((n_points, 3))
    rig_relative_poses = params[n_ref_cameras * 6 + n_points * 3:].reshape((n_rig_rel_poses, 6))

    all_camera_poses = np.zeros((n_cameras, 6))
    for cam_idx in range(n_cameras):
        info = cam_info[cam_idx]
        instance_idx = info['instance_idx']
        ref_pose = ref_cam_poses[instance_idx]
        R_world_ref = Rotation.from_rotvec(ref_pose[:3])
        t_world_ref = ref_pose[3:]

        if info['is_ref']:
            all_camera_poses[cam_idx] = ref_pose
        else:
            rel_pose_idx = info['rel_pose_idx']
            rel_pose = rig_relative_poses[rel_pose_idx]
            R_rig_canonical = Rotation.from_rotvec(rel_pose[:3])
            t_rig_canonical = rel_pose[3:]
            R_world_other = R_world_ref * R_rig_canonical
            t_world_other = t_world_ref + R_world_ref.apply(t_rig_canonical)
            all_camera_poses[cam_idx, :3] = R_world_other.as_rotvec()
            all_camera_poses[cam_idx, 3:] = t_world_other

    residuals = []
    for i in range(len(observed_pixels)):
        cam_idx, point_idx = camera_indices[i], point_indices[i]
        K = K_map[cam_idx]
        pose, point = all_camera_poses[cam_idx], points_3d[point_idx]
        reprojected_point = project_points(point.reshape(1,3), K, pose)
        error = reprojected_point.flatten() - observed_pixels[i]
        residuals.append(error)
        
    return np.array(residuals).flatten()


def create_sparsity_matrix(n_cameras, n_points, n_rig_rel_poses, camera_indices, point_indices, rig_config, rig_def, cam_info):

    n_ref_cameras = len(rig_config)
    n_obs = len(camera_indices)
    n_rows = 2 * n_obs
    n_cols = 6 * n_ref_cameras + 3 * n_points + 6 * n_rig_rel_poses
    A = lil_matrix((n_rows, n_cols), dtype=int)
    point_param_start = 6 * n_ref_cameras
    rig_param_start = 6 * n_ref_cameras + 3 * n_points

    for i in range(n_obs):
        row_idx = i * 2
        cam_idx, point_idx = camera_indices[i], point_indices[i]
        info = cam_info[cam_idx]
        instance_idx = info['instance_idx']

    for j in range(3):
        A[row_idx:row_idx+2, point_param_start + 3 * point_idx + j] = 1

    for j in range(6):
        A[row_idx:row_idx+2, 6 * instance_idx + j] = 1

    if not info['is_ref']:
        rel_pose_idx = info['rel_pose_idx']

    for j in range(6): 
        A[row_idx:row_idx+2, rig_param_start + 6 * rel_pose_idx + j] = 1

    return A


def create_sparsity_matrix_vectorized(n_cameras, n_points, n_rig_rel_poses, camera_indices, point_indices, rig_config, rig_def, cam_info):
    n_ref_cameras = len(rig_config)
    n_obs = len(camera_indices)
    
    cam_info_array = np.array([
        (ci['instance_idx'], 1 if ci['is_ref'] else 0, ci['rel_pose_idx']) 
        for ci in cam_info
    ])
    obs_cam_info = cam_info_array[camera_indices]
    obs_instance_indices = obs_cam_info[:, 0].astype(int)
    obs_is_ref = obs_cam_info[:, 1].astype(bool)
    obs_rel_pose_indices = obs_cam_info[:, 2].astype(int)

    all_rows, all_cols = [], []

    # --- Point Dependency ---
    # Each obs has 2 residuals. Each point has 3 params. Total 2*3=6 Jacobian entries per obs.
    row_indices_base = np.arange(n_obs) * 2
    # Create rows for both x and y: [0, 1, 2, 3, ...] -> [[0,0,0,1,1,1], [2,2,2,3,3,3], ...]
    rows_for_points = np.repeat(row_indices_base, 3 * 2).reshape(-1, 6)
    rows_for_points[:, 3:] += 1 # Offset the y-residual rows
    rows_for_points = rows_for_points.flatten()

    point_param_start = 6 * n_ref_cameras
    # For each obs, dependencies are [px, py, pz, px, py, pz]
    cols_for_points = point_param_start + 3 * np.repeat(point_indices, 6) + np.tile([0,1,2,0,1,2], n_obs)
    
    all_rows.append(rows_for_points)
    all_cols.append(cols_for_points)

    # --- Reference Camera Dependency ---
    # Each obs has 2 residuals. Each pose has 6 params. Total 2*6=12 Jacobian entries per obs.
    rows_for_ref_cam = np.repeat(row_indices_base, 6 * 2).reshape(-1, 12)
    rows_for_ref_cam[:, 6:] += 1 # Offset the y-residual rows
    rows_for_ref_cam = rows_for_ref_cam.flatten()

    cols_for_ref_cam = 6 * np.repeat(obs_instance_indices, 12) + np.tile([0,1,2,3,4,5,0,1,2,3,4,5], n_obs)

    all_rows.append(rows_for_ref_cam)
    all_cols.append(cols_for_ref_cam)

    # --- Relative Pose Dependency ---
    non_ref_mask = ~obs_is_ref
    non_ref_obs_indices = np.where(non_ref_mask)[0]
    if len(non_ref_obs_indices) > 0:
        n_non_ref_obs = len(non_ref_obs_indices)
        row_indices_base_non_ref = non_ref_obs_indices * 2

        rows_for_rel_pose = np.repeat(row_indices_base_non_ref, 6 * 2).reshape(-1, 12)
        rows_for_rel_pose[:, 6:] += 1 # Offset the y-residual rows
        rows_for_rel_pose = rows_for_rel_pose.flatten()
        
        rig_param_start = 6 * n_ref_cameras + 3 * n_points
        relevant_rel_pose_indices = obs_rel_pose_indices[non_ref_mask]
        
        cols_for_rel_pose = rig_param_start + 6 * np.repeat(relevant_rel_pose_indices, 12) + np.tile([0,1,2,3,4,5,0,1,2,3,4,5], n_non_ref_obs)
        
        all_rows.append(rows_for_rel_pose)
        all_cols.append(cols_for_rel_pose)
    
    # --- Assemble the final sparse matrix ---
    final_rows = np.concatenate(all_rows)
    final_cols = np.concatenate(all_cols)
    data = np.ones_like(final_rows, dtype=int)
    
    n_rows = 2 * n_obs
    n_cols = 6 * n_ref_cameras + 3 * n_points + 6 * n_rig_rel_poses
    
    A = coo_matrix((data, (final_rows, final_cols)), shape=(n_rows, n_cols))
    
    return A


    
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

    # Use ast.literal_eval to convert strings to arrays
    def _str_to_array(s):
        try:
            py_list = ast.literal_eval(s)
            return np.array(py_list)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing string to array: {s}. Error: {e}")
            return np.array([])
        
    # Apply the conversion function to relevant columns
    for col in ['K', 'R', 't']:
        if col in df.columns:
            df[col] = df[col].apply(_str_to_array)
    return df


def center_cameras(cams_df):
    print("\nCentering camera coordinates")
    # Calculate the mean/centroid of all camera translation vectors
    all_t_vectors = np.array(list(cams_df['t']))
    local_origin = np.mean(all_t_vectors, axis=0)
    print(f"Calculated scene centroid (local origin) at: {local_origin.ravel()}")
    
    # Create a new DataFrame with centered coordinates
    centered_cam_df = cams_df.copy()
    centered_cam_df['t'] = centered_cam_df['t'].apply(lambda t: t - local_origin)
    
    return centered_cam_df, local_origin

def build_initial_observations(
        match_files: list, 
        image_name_to_idx: dict, 
        initial_camera_params: list, 
        K_map: dict, 
        cam_df: pd.DataFrame, 
        max_points_per_pair: int
    ):
    """
    Loads feature matches, triangulates initial 3D points, and filters them
    based on initial reprojection error.
    """
    print(f"Found {len(match_files)} feature match files. Loading and triangulating 3D points...")
    observations, initial_points_3d, point_counter = [], [], 0
    idx_to_frame = cam_df.set_index('image_name')['frame'].to_dict()

    for i, match_file in tqdm(enumerate(match_files), total=len(match_files), desc="Triangulating"):
        img1_name, img2_name = os.path.splitext(os.path.basename(match_file))[0].replace("_match","").split('__')
        if (img1_name not in image_name_to_idx) or (img2_name not in image_name_to_idx): continue
        
        cam1_idx, cam2_idx = image_name_to_idx[img1_name], image_name_to_idx[img2_name]
        frame1, frame2 = idx_to_frame.get(img1_name), idx_to_frame.get(img2_name)
        if (frame1 is None) or (frame2 is None) or frame1 == frame2: continue

        cam1_pose, cam2_pose = initial_camera_params[cam1_idx]['pose'], initial_camera_params[cam2_idx]['pose']
        K1, K2 = K_map[cam1_idx], K_map[cam2_idx]
        
        R1, _ = cv2.Rodrigues(cam1_pose[:3]); t1 = cam1_pose[3:].reshape(3,1)
        P1 = K1 @ np.hstack((R1, t1))
        
        R2, _ = cv2.Rodrigues(cam2_pose[:3]); t2 = cam2_pose[3:].reshape(3,1)
        P2 = K2 @ np.hstack((R2, t2))
        
        match_df = pd.read_csv(match_file, sep=" ", header=0).dropna()
        potential_points_for_pair = []
        for _, row in match_df.iterrows():
            px1, px2 = np.array([row['img1_px'], row['img1_py']]), np.array([row['img2_px'], row['img2_py']])
            point_4d_hom = cv2.triangulatePoints(P1, P2, px1, px2)
            if point_4d_hom[3, 0] < 0: point_4d_hom *= -1
            w = point_4d_hom[3, 0]
            if np.isclose(w, 0): continue
            point_3d = (point_4d_hom[:3] / w).flatten()
            if not np.all(np.isfinite(point_3d)): continue

            px1_reproj, _ = cv2.projectPoints(point_3d.reshape(1,3), cam1_pose[:3], cam1_pose[3:], K1, None)
            px2_reproj, _ = cv2.projectPoints(point_3d.reshape(1,3), cam2_pose[:3], cam2_pose[3:], K2, None)
            error1, error2 = np.linalg.norm(px1 - px1_reproj.flatten()), np.linalg.norm(px2 - px2_reproj.flatten())
            mean_error = (error1 + error2) / 2.0
            potential_points_for_pair.append({'point_3d': point_3d, 'error': mean_error, 'obs1': (cam1_idx, px1), 'obs2': (cam2_idx, px2)})

        if not potential_points_for_pair: continue
        potential_points_for_pair.sort(key=lambda p: p['error'])
        best_points_for_pair = potential_points_for_pair[:max_points_per_pair]

        for point_data in best_points_for_pair:
            initial_points_3d.append(point_data['point_3d'])
            cam1_idx, px1 = point_data['obs1']
            cam2_idx, px2 = point_data['obs2']
            observations.append((cam1_idx, point_counter, px1))
            observations.append((cam2_idx, point_counter, px2))
            point_counter += 1

    initial_points_3d = np.array(initial_points_3d)
    if len(initial_points_3d) == 0: raise RuntimeError("Failed to generate any valid 3D points after filtering.")
    print(f"Generated {len(initial_points_3d)} valid initial 3D points from {len(observations)} observations.")
    
    return initial_points_3d, observations


def opencv_cameras_to_asp_pinhole(
        cam_specs_df: pd.DataFrame, 
        out_folder: str
        ):
    print(f"\n--- Converting final camera models to ASP TSAI format in '{out_folder}' ---")
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
# MAIN SCRIPT
# ==============================================================================
def main():
    # --- User inputs ---
    data_path = "/Users/rdcrlrka/Research/SkySat-Stereo/study-sites/MCS/20240420/custom_rig_calibration"
    CAMERA_SPECS_CSV = os.path.join(data_path, "init_pinhole_cams", "initial_cam_specs.csv")
    FEATURE_MATCH_DIR = os.path.join(data_path, "init_feature_matches")
    FINAL_CAM_DIR = os.path.join(data_path, "bundle_adjust")
    MAX_POINTS_PER_PAIR = 100
    FTOL = 1e-3
    THREADS = 10

    # --- Step 0: Center camera coordinates ---
    CENTERED_CAMERA_SPECS_CSV = os.path.join(data_path, "init_pinhole_cams", "initial_cam_specs_centered.csv")
    print(f"0. Pre-processing: Centering camera coordinates for numerical stability.")
    orig_cam_df = load_cam_specs(CAMERA_SPECS_CSV)
    centered_cam_df, local_origin = center_cameras(orig_cam_df)
    save_cam_specs(centered_cam_df, CENTERED_CAMERA_SPECS_CSV)

    # --- Step 1 & 2: Load Data and Triangulate ---
    print("\n1. Loading and preparing camera data...")
    cam_df = load_cam_specs(CENTERED_CAMERA_SPECS_CSV)
    cam_df['frame'] = cam_df['frame'].astype(int) 
    cam_df = cam_df.sort_values(by=['frame', 'cam']).reset_index(drop=True)
    ref_cam_name = cam_df['cam'].value_counts().idxmax()
    print(f"Identified '{ref_cam_name}' as the reference camera (most frequent).")
    cam_names_list = sorted(cam_df['cam'].unique())
    other_cam_names = [name for name in cam_names_list if name != ref_cam_name]
    other_cam_name_to_rel_idx = {name: i for i, name in enumerate(other_cam_names)}
    frame_numbers = sorted(cam_df['frame'].unique())
    frame_to_instance_idx = {frame: i for i, frame in enumerate(frame_numbers)}
    rig_def = [ref_cam_name] + other_cam_names
    n_rig_rel_poses = len(other_cam_names)
    print(f"Rig cameras detected: {rig_def}")
    initial_rig_geometry = estimate_initial_rig_geometry(cam_df, ref_cam_name)
    image_name_to_idx = {name: i for i, name in enumerate(cam_df['image_name'])}
    initial_camera_params, cam_info, ref_cam_indices, K_map = [], [], [], {}
    for idx, row in cam_df.iterrows():
        K, R, t = row['K'], row['R'], row['t']
        rvec, _ = cv2.Rodrigues(R)
        K_map[idx] = K
        initial_camera_params.append({'pose': np.hstack([rvec.flatten(), t.flatten()])})
        is_ref = (row['cam'] == ref_cam_name)
        if is_ref: ref_cam_indices.append(idx)
        cam_info.append({'instance_idx': frame_to_instance_idx[row['frame']], 'is_ref': is_ref, 'rel_pose_idx': -1 if is_ref else other_cam_name_to_rel_idx[row['cam']]})
    NUM_CAMERAS = len(cam_df)
    rig_config = [{} for _ in range(len(frame_numbers))]
    for idx, row in cam_df.iterrows():
        instance_idx = frame_to_instance_idx[row['frame']]
        slot_name = row['cam']
        rig_config[instance_idx][slot_name] = idx
    print(f"Loaded {NUM_CAMERAS} cameras across {len(rig_config)} unique frames.")
    match_files = sorted(glob(os.path.join(FEATURE_MATCH_DIR, "*__*.txt")))
    initial_points_3d, observations = build_initial_observations(match_files, image_name_to_idx, initial_camera_params, K_map, cam_df, MAX_POINTS_PER_PAIR)

    # --- Step 3: Prepare for Bundle Adjustment ---
    print("\n3. Preparing for bundle adjustment...")
    initial_rig_rel_poses = np.zeros((n_rig_rel_poses, 6))
    print("Using estimated geometry as initial guess for optimizer:")
    for i, cam_name in enumerate(other_cam_names):
        if cam_name in initial_rig_geometry:
            avg_R, avg_t = initial_rig_geometry[cam_name]['R'], initial_rig_geometry[cam_name]['t']
            rvec, _ = cv2.Rodrigues(avg_R)
            initial_rig_rel_poses[i] = np.hstack([rvec.flatten(), avg_t.flatten()])
            print(f"\t- Initial guess for {ref_cam_name} -> {cam_name}: t = {avg_t.flatten()}")
    initial_ref_poses_list = []
    for instance in rig_config:
        if ref_cam_name in instance:
            initial_ref_poses_list.append(initial_camera_params[instance[ref_cam_name]]['pose'])
        else:
            initial_ref_poses_list.append(initial_ref_poses_list[-1] if initial_ref_poses_list else np.zeros(6))
    initial_ref_poses = np.array(initial_ref_poses_list).flatten()

    params_0 = np.concatenate([initial_ref_poses, initial_points_3d.flatten(), initial_rig_rel_poses.flatten()])

    camera_indices = np.array([o[0] for o in observations])
    point_indices = np.array([o[1] for o in observations])
    observed_pixels = np.array([o[2] for o in observations])
    
    print("Creating Jacobian sparsity matrix")
    sparsity_matrix = create_sparsity_matrix_vectorized(NUM_CAMERAS, len(initial_points_3d), n_rig_rel_poses, camera_indices, point_indices, rig_config, rig_def, cam_info)
    print("Size of the sparsity matrix:", sparsity_matrix.shape)

    # --- Step 4: Run Optimization ---
    print("\n4. Starting the SciPy least_squares optimizer...")
    start_time = datetime.now()
    res = least_squares(
        fun=bundle_adjustment_residuals, 
        x0=params_0, 
        jac_sparsity=sparsity_matrix, 
        verbose=2, x_scale='jac', 
        ftol=FTOL, 
        method='trf', 
        workers=THREADS,
        args=(
            NUM_CAMERAS, len(initial_points_3d), n_rig_rel_poses, camera_indices,
            point_indices, observed_pixels, K_map, rig_config, rig_def, cam_info
            )
            )
    end_time = datetime.now()
    print(f"Optimization finished after {np.round((end_time-start_time).total_seconds(), 2)} seconds.")

    # --- Step 5: REPORT AND SAVE RESULTS ---
    print("\n5. Reporting and saving results...")
    n_ref_cameras = len(rig_config)
    final_ref_poses_local = res.x[:n_ref_cameras * 6].reshape((n_ref_cameras, 6))
    rig_params_offset = n_ref_cameras * 6 + len(initial_points_3d) * 3
    final_rig_poses = res.x[rig_params_offset:].reshape((n_rig_rel_poses, 6))

    # --- Report Results ---
    final_rms = np.sqrt(np.mean(res.fun**2))
    print(f"\nFinal RMS error: {np.round(final_rms,2)} pixels")
    print(f"\n--- Optimized Rig Geometry (Relative to {rig_def[0]}) ---")
    for i in range(n_rig_rel_poses):
        pose = final_rig_poses[i]
        angle_axis, translation = pose[:3], pose[3:]
        print(f"\n* Relative Pose for {rig_def[i+1]}:")
        print(f"  - Angle-Axis:  [{angle_axis[0]:.4f}, {angle_axis[1]:.4f}, {angle_axis[2]:.4f}]")
        print(f"  - Translation: [{translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f}] (in meters)")

    print("\nReconstructing and saving final camera models...")
    os.makedirs(FINAL_CAM_DIR, exist_ok=True)
    
    # --- Reconstruct, De-center, and Save Final Cameras ---
    final_cam_data = []
    for cam_idx in range(NUM_CAMERAS):
        info = cam_info[cam_idx]
        instance_idx = info['instance_idx']
        
        ref_pose_local = final_ref_poses_local[instance_idx]
        R_world_ref_local = Rotation.from_rotvec(ref_pose_local[:3])
        t_world_ref_local = ref_pose_local[3:]

        if info['is_ref']:
            final_pose_local = ref_pose_local
        else:
            rel_pose_idx = info['rel_pose_idx']
            rel_pose_opt = final_rig_poses[rel_pose_idx]
            R_rig_opt = Rotation.from_rotvec(rel_pose_opt[:3])
            t_rig_opt = rel_pose_opt[3:]
            R_world_other_local = R_world_ref_local * R_rig_opt
            t_world_other_local = t_world_ref_local + R_world_ref_local.apply(t_rig_opt)
            final_pose_local = np.hstack([R_world_other_local.as_rotvec(), t_world_other_local])

        # De-center the camera pose by adding the original ECEF centroid back
        final_t_ecef = np.array(final_pose_local[3:]).reshape(3,1) + np.array(local_origin).reshape(3,1)
        final_R, _ = cv2.Rodrigues(final_pose_local[:3])
        
        image_name = cam_df.loc[cam_idx, 'image_name']
        final_cam_data.append({
            'image_name': image_name,
            'K': K_map[cam_idx],
            'R': final_R,
            't': final_t_ecef
        })

    final_cam_df = pd.DataFrame(final_cam_data)
    opencv_cameras_to_asp_pinhole(final_cam_df, FINAL_CAM_DIR)

if __name__ == "__main__":
    main()