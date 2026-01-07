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

import torch
import pypose as pp
from torch import nn

# ==============================================================================
# DATA LOADING & PREPROCESSING FUNCTIONS
# ==============================================================================

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

def load_all_matches(match_folder: str, image_folder: str) -> dict:
    print("\nLoading all feature matches...")
    all_matches = {}
    match_files = glob(os.path.join(match_folder, "*.csv"))
    for i, match_file in enumerate(tqdm(match_files, desc="Loading matches")):
        img1_name, img2_name = os.path.basename(match_file).replace("_match.csv", "").split('__')
        pts1, pts2 = np.loadtxt(match_file, delimiter=',', skiprows=1, usecols=(0,1,2,3)).reshape(-1, 2, 2).transpose(1,0,2)
        all_matches[i] = {"img1": img1_name, "img2": img2_name, "pts1": pts1, "pts2": pts2}
    print(f"Loaded {len(all_matches)} match files.")
    return all_matches

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
    print("\nBuilding initial 3D points and observation list...")
    cam_map = {name: idx for idx, name in enumerate(cam_specs_df['image_name'])}
    cam_params_map = cam_specs_df.set_index('image_name').to_dict('index')
    observations = []
    points_3d = []
    point_map = {}
    for i in tqdm(matches.keys(), desc="Triangulating points"):
        cam1_name, cam2_name = matches[i]['img1'], matches[i]['img2']
        pts1, pts2 = matches[i]['pts1'], matches[i]['pts2']
        cam1_idx, cam2_idx = cam_map[cam1_name], cam_map[cam2_name]
        R1, t1, K1 = cam_params_map[cam1_name]['R'], cam_params_map[cam1_name]['t'], cam_params_map[cam1_name]['K']
        R2, t2, K2 = cam_params_map[cam2_name]['R'], cam_params_map[cam2_name]['t'], cam_params_map[cam2_name]['K']
        P1 = K1 @ np.hstack((R1, t1)); P2 = K2 @ np.hstack((R2, t2))
        points4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points3d_new = (points4d_hom[:3, :] / points4d_hom[3, :]).T
        for j in range(len(pts1)):
            pt1_key = (cam1_idx, j)
            if pt1_key in point_map: point_3d_idx = point_map[pt1_key]
            else:
                point_3d_idx = len(points_3d)
                points_3d.append(points3d_new[j])
                point_map[pt1_key] = point_3d_idx
                observations.append({'cam_idx': cam1_idx, 'pt_3d_idx': point_3d_idx, 'pt_2d': pts1[j]})
            pt2_key = (cam2_idx, j); point_map[pt2_key] = point_3d_idx
            observations.append({'cam_idx': cam2_idx, 'pt_3d_idx': point_3d_idx, 'pt_2d': pts2[j]})
    return np.array(points_3d), observations


# ==============================================================================
# PYPOSE BUNDLE ADJUSTMENT IMPLEMENTATION
# ==============================================================================

class RigBundleAdjustmentModel(nn.Module):
    """
    A PyTorch module for rig-constrained bundle adjustment using PyPose.

    This model holds the optimizable parameters:
    - Base Poses: The trajectory of the reference camera (e.g., 'd1').
    - Relative Poses: The single, shared transformation for each non-reference camera.
    - 3D Points: The coordinates of all feature points in the world frame.
    """
    def __init__(self, base_poses, relative_poses, points, intrinsics):
        super().__init__()
        # Register all parameters that the optimizer should adjust
        self.base_poses = pp.Parameter(base_poses)
        self.relative_poses = pp.Parameter(relative_poses)
        self.points = nn.Parameter(points)
        # Intrinsics are considered fixed
        self.register_buffer('intrinsics', intrinsics)

    def forward(self, observations, cam_specs_df, ref_cam_id, base_pose_map, relative_cam_map):
        """
        This is the cost function. It calculates the reprojection error for all observations.
        It's a direct PyTorch/PyPose translation of the original 'fun' function.
        """
        # Get indices for all observations in the batch
        cam_indices = observations[:, 0].long()
        pt_indices = observations[:, 1].long()
        pixels_measured = observations[:, 2:]

        # --- Select all necessary data for the batch using tensor indexing ---
        # This is significantly faster than looping in Python
        cam_specs = cam_specs_df.iloc[cam_indices]
        base_keys = cam_specs["ref_base_key"].values
        cam_types = cam_specs["cam"].values
        
        # Map keys to indices
        base_pose_indices = torch.tensor([base_pose_map[key] for key in base_keys], device=self.points.device)
        relative_pose_indices = torch.tensor([relative_cam_map[typ] if typ in relative_cam_map else -1 for typ in cam_types], device=self.points.device)
        
        # Select the parameters using the calculated indices
        selected_base_poses = self.base_poses[base_pose_indices]
        selected_points = self.points[pt_indices]
        selected_intrinsics = self.intrinsics[cam_indices]

        # --- Reconstruct final camera poses for each observation ---
        # Start with the base poses
        R_cam, t_cam = selected_base_poses.rotation().matrix(), selected_base_poses.translation().unsqueeze(-1)

        # Identify which observations are from non-reference cameras
        is_relative_mask = (relative_pose_indices != -1)
        if torch.any(is_relative_mask):
            # Select the shared relative poses for those observations
            selected_relative_poses = self.relative_poses[relative_pose_indices[is_relative_mask]]
            R_rel = selected_relative_poses.rotation().matrix()
            t_rel = selected_relative_poses.translation().unsqueeze(-1)
            
            # Apply the rig geometry: T_final = T_base * T_relative
            # Note: PyTorch requires careful broadcasting for this batch matrix multiplication
            R_cam[is_relative_mask] = R_cam[is_relative_mask] @ R_rel
            t_cam[is_relative_mask] = t_cam[is_relative_mask] + pp.bmv(R_cam[is_relative_mask], t_rel.squeeze(-1)).unsqueeze(-1)
        
        # --- Project points and calculate error ---
        # This part is the same as the simpler model, but now operates on the fully constructed poses
        points_in_cam_frame = pp.bmv(R_cam, selected_points) + t_cam.squeeze(-1)
        projected_points = pp.bmv(selected_intrinsics, points_in_cam_frame)
        
        px = projected_points[..., 0] / projected_points[..., 2]
        py = projected_points[..., 1] / projected_points[..., 2]
        pixels_predicted = torch.stack([px, py], dim=-1)
        
        error = pixels_predicted - pixels_measured
        return error.flatten()


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

# ==============================================================================
# SECTION 3: MAIN WORKFLOW ORCHESTRATION
# ==============================================================================

def main():
    # --- 1. Configuration ---
    # Use a simple class for configuration instead of argparse for notebook compatibility
    class Args:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        data_folder = "/Users/rdcrlrka/Research/SkySat-Stereo/study-sites/MCS/20240420"
        image_folder = os.path.join(data_folder, "SkySatScene")
        out_folder = os.path.join(data_folder, "pypose_rig_calibration")
        init_cam_folder = os.path.join(out_folder, "init_pinhole_cams")
        init_matches_folder = os.path.join(out_folder, "init_feature_matches")
        ba_folder = os.path.join(out_folder, "bundle_adjustment")
        epochs = 20 # Number of optimization iterations

    args = Args()
    print(f"Starting PyPose Bundle Adjustment. Using device: {args.device}")
    os.makedirs(args.ba_folder, exist_ok=True)

    # --- 2. Load All Initial Data (from original script) ---
    image_list = sorted(glob(os.path.join(args.image_folder, "*_analytic.tif")))
    init_cam_list = sorted(glob(os.path.join(args.init_cam_folder, "*.tsai")))
    if not init_cam_list: raise FileNotFoundError(f"No .tsai files in '{args.init_cam_folder}'")
    
    cam_params_df = asp_pinhole_to_opencv_cameras(init_cam_list)
    specs_df = pd.concat(map(parse_image_specs, image_list), ignore_index=True)
    cam_specs_df = pd.merge(cam_params_df, specs_df, on="image_name").sort_values("image_name").reset_index(drop=True)
    cam_specs_df['cam_idx'] = cam_specs_df.index
    
    matches = load_all_matches(args.init_matches_folder, args.image_folder)
    if not matches: raise ValueError("No feature matches found.")

    # --- 3. Create Initial Rig, Points, and Mappings ---
    ref_cam_id = cam_specs_df["cam"].value_counts().idxmax()
    initial_rig_poses = estimate_initial_rig_geometry(cam_specs_df, ref_cam_id)
    points_3d, observations = build_observation_data(matches, cam_specs_df)

    # Create mappings needed for the model's forward pass
    unique_base_keys = sorted(cam_specs_df[cam_specs_df['cam'] == ref_cam_id]['image_name'].unique())
    base_pose_map = {key: i for i, key in enumerate(unique_base_keys)}
    relative_cams = sorted([cam for cam in cam_specs_df['cam'].unique() if cam != ref_cam_id])
    relative_cam_map = {cam: i for i, cam in enumerate(relative_cams)}
    
    # Map every camera to a reference base pose
    for idx, row in cam_specs_df.iterrows():
        if row['cam'] == ref_cam_id:
            cam_specs_df.at[idx, "ref_base_key"] = row['image_name']
        else: # Simple heuristic: find closest reference frame (can be improved)
            ref_frames = cam_specs_df[cam_specs_df['cam'] == ref_cam_id]
            time_diffs = (ref_frames['datetime'] - row['datetime']).abs()
            closest_ref = ref_frames.loc[time_diffs.idxmin()]
            cam_specs_df.at[idx, "ref_base_key"] = closest_ref['image_name']

    print("Camera specs:")
    cam_specs_df.head()
    
    # # --- 4. Prepare Data for PyTorch/PyPose ---
    # print("\nPreparing all data into PyTorch Tensors...")
    # dtype = torch.float64 # Use double precision for numerical stability
    # device = args.device

    # # Base Poses
    # base_initial_poses_df = cam_specs_df[cam_specs_df['image_name'].isin(unique_base_keys)].set_index('image_name').loc[unique_base_keys]
    # base_poses_list = [pp.SE3.from_matrix(torch.tensor(np.hstack([row['R'], row['t']]), dtype=dtype)) for _, row in base_initial_poses_df.iterrows()]
    # base_poses = pp.Parameter(pp.stack(base_poses_list).to(device))

    # # Relative Poses
    # relative_poses_list = [pp.SE3.from_matrix(torch.tensor(np.hstack([initial_rig_poses[cam]['R'], initial_rig_poses[cam]['t']]), dtype=dtype)) for cam in relative_cams]
    # relative_poses = pp.Parameter(pp.stack(relative_poses_list).to(device))

    # # 3D Points
    # points = nn.Parameter(torch.tensor(points_3d, dtype=dtype, device=device))

    # # Intrinsics (fixed)
    # intrinsics = torch.tensor(np.stack(cam_specs_df['K'].values), dtype=dtype, device=device)
    
    # # Observations
    # obs_list = [[obs['cam_idx'], obs['pt_3d_idx'], obs['pt_2d'][0], obs['pt_2d'][1]] for obs in observations]
    # observations_tensor = torch.tensor(obs_list, dtype=dtype, device=device)

    # # --- 5. Run PyPose Optimization ---
    # model = RigBundleAdjustmentModel(base_poses, relative_poses, points, intrinsics).to(device)
    # optimizer = pp.optim.LevenbergMarquardt(model, vectorize=False) # Vectorize=False saves memory

    # print("\nStarting PyPose Levenberg-Marquardt Optimization...")
    # for epoch in range(args.epochs):
    #     loss = optimizer.step(observations_tensor, cam_specs_df, ref_cam_id, base_pose_map, relative_cam_map)
    #     print(f"    Epoch {epoch:02d}: Loss = {loss.item():.4f}")

    # # --- 6. Post-Hoc Alignment and Saving Results ---
    # print("\nPerforming post-hoc alignment and saving results...")
    
    # # Get final optimized parameters from the model
    # final_base_poses = model.base_poses.detach()
    # final_relative_poses = model.relative_poses.detach()
    # final_points_aligned = model.points.detach()

    # # Calculate initial and final centroids for alignment
    # initial_centroid = torch.mean(base_poses.translation(), dim=0)
    # final_centroid_drifted = torch.mean(final_base_poses.translation(), dim=0)
    # shift_vector = initial_centroid - final_centroid_drifted
    
    # print(f"    Alignment shift vector: {shift_vector.cpu().numpy()}")
    
    # # Align the poses and points
    # final_base_poses.translation().add_(shift_vector)
    # final_points_aligned.add_(shift_vector)
    
    # # Create final dataframe with aligned poses
    # final_cam_specs_df = cam_specs_df.copy()
    # base_pose_map_inv = {v: k for k, v in base_pose_map.items()}
    # relative_cam_map_inv = {v: k for k, v in relative_cam_map.items()}

    # for i in range(len(final_cam_specs_df)):
    #     cam_spec = final_cam_specs_df.iloc[i]
    #     base_key = cam_spec["ref_base_key"]
    #     base_pose_idx = base_pose_map[base_key]
        
    #     T_base = final_base_poses[base_pose_idx]
        
    #     if cam_spec['cam'] == ref_cam_id:
    #         T_final = T_base
    #     else:
    #         relative_pose_idx = relative_cam_map[cam_spec['cam']]
    #         T_rel = final_relative_poses[relative_pose_idx]
    #         T_final = T_base @ T_rel
        
    #     # Update DataFrame with NumPy arrays for saving
    #     final_cam_specs_df.at[i, "R"] = T_final.rotation().matrix().cpu().numpy()
    #     final_cam_specs_df.at[i, "t"] = T_final.translation().unsqueeze(-1).cpu().numpy()

    # # --- 7. Save and Visualize ---
    # final_cam_specs_file = os.path.join(args.ba_folder, "pypose_final_cam_specs.csv")
    # save_cam_specs(final_cam_specs_df, final_cam_specs_file) 
    # print(f"Final camera specs ready to be saved to: {final_cam_specs_file}")
    
    # opencv_cameras_to_asp_pinhole(final_cam_specs_df, args.ba_folder)
    
    # plot_rig_comparison(cam_specs_df, final_cam_specs_df)


if __name__ == "__main__":
    main()

