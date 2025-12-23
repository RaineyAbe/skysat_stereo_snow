#! /usr/bin/env python3

# flag to avoid pytorch-related error on imports
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"

import pandas as pd
import json
from pathlib import Path
import rioxarray as rxr
import numpy as np
from tqdm import tqdm
import itertools
import rasterio as rio
import geopandas as gpd
from shapely.geometry import Point, Polygon
import shutil
import subprocess
import datetime
from scipy.spatial.transform import Rotation
import pycolmap
import h5py
from hloc import extract_features, match_features
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------
def run_cmd(bin: str = None, 
            args: list = None,
            log_out_folder: str = None, **kw) -> str:
    # Construct command
    binpath = shutil.which(bin)
    if binpath.endswith(".py"):
        call = ["python", binpath,]
    else:
        call = [binpath,]
    if args is not None: 
        call.extend(args)

    # Run command and capture output
    process = subprocess.Popen(call, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    # Log output
    if log_out_folder is not None:
        log_file = Path(log_out_folder) / (Path(bin).stem + "_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".log")
        with open(log_file, "w") as f:
            f.write(f"stdout:\n{stdout.decode()}\n")
            f.write(f"stderr:\n{stderr.decode()}\n")
    out = stdout.decode()

    return out


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
        (width - 1, height_px - 1),      # bottom-right
        (0, height_px - 1)               # bottom-left
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


def identify_overlapping_image_pairs(
        img_list: str = None, 
        min_overlap: float = 10, 
        bh_ratio_range: tuple = None,
        true_stereo: bool = True,
        utm_epsg: str = None,
        out_folder: str = None,
        write_basename: bool = False
        )-> None:
    print("\nIdentifying stereo image pairs.")
    os.makedirs(out_folder, exist_ok=True)

    # Get image bounds polygons
    polygons = {img: get_image_polygon(img, out_crs=utm_epsg) for img in img_list}
    
    # Compare all unique pairs
    print(f"Requirements:")
    print(f"\toverlap >= {min_overlap} %")
    if bh_ratio_range:
        print(f"\tbaseline to height ratio = {bh_ratio_range[0]} to {bh_ratio_range[1]}")
    print(f"\ttrue stereo = {true_stereo}")
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
            if overlap_percent >= min_overlap:
                # check for B/H ratio thresholds if specified
                bh_ratio = calculate_baseline_to_height_ratio(img1, img2, utm_epsg)
                if bh_ratio_range:
                    if (bh_ratio < bh_ratio_range[0]) | (bh_ratio > bh_ratio_range[1]):
                        continue
                
                # check for true stereo if specified - datetimes must be different
                dt1 = "_".join(os.path.basename(img1).split("_")[0:2])
                dt2 = "_".join(os.path.basename(img2).split("_")[0:2])
                if true_stereo & (dt1==dt2):
                    continue

                bh_ratios += [bh_ratio]
                overlapping_pairs += [(img1, img2)]
                overlap_ratios += [overlap_percent]
    print("Number of stereo pairs identified =", len(overlap_ratios))
                    
    # Write to file
    out_fn = os.path.join(out_folder, "stereo_image_pairs.txt")
    out_specs_fn = os.path.join(out_folder, "stereo_image_pairs_with_specs.txt")
    # add the header to the specs file
    with open(out_specs_fn, "w") as f:
        f.write(f"img1 img2 datetime overlap_percent bh_ratio\n")
    # iterate over pairs
    for i, (img1, img2) in enumerate(overlapping_pairs):
        # get specs from image basenames
        date1, time1 = os.path.basename(img1).split("_")[0:2]
        date2, time2 = os.path.basename(img2).split("_")[0:2]
        dt_text = date1 + "_" + time1 + "__" + date2 + "_" + time2

        # write pairs file
        with open(out_fn, "a") as f:
            if write_basename:
                if i==0:
                    print(f"Writing image pairs with basename only to:\n{out_specs_fn}")
                f.write(f"{os.path.basename(img1)} {os.path.basename(img2)}\n")
            else:
                if i==0:
                    print(f"Writing image pairs with full path name to:\n{out_specs_fn}")
                f.write(f"{img1} {img2}\n")

        # write pairs with specs file
        with open(out_specs_fn, "a") as f:
            if write_basename:
                f.write(f"{os.path.basename(img1)} {os.path.basename(img2)} {dt_text} {overlap_ratios[i]} {bh_ratios[i]}\n")
            else:
                f.write(f"{img1} {img2} {dt_text} {overlap_ratios[i]} {bh_ratios[i]}\n")

    return


def parse_tsai(tsai_path):
    with open(tsai_path, "r") as f:
        lines = f.readlines()
    lines = [x.replace("\n","") for x in lines]

    def get_value(lines, k):
        line = [x for x in lines if k in x][0]
        val_string = line.split(" = ")[1]
        if " " in val_string:
            return np.array(val_string.split(" ")).astype(float)
        else:
            return float(val_string)

    # focal lengths and optical centers
    fx = get_value(lines, "fu =")
    fy = get_value(lines, "fv =")
    cx = get_value(lines, "cu =")
    cy = get_value(lines, "cv =")
    # rotation matrix
    R = get_value(lines, "R =").reshape(3, 3)
    # camera center
    C = get_value(lines, "C =")

    return (fx, fy, cx, cy), R, C


def save_image_as_8bit(image_file, out_image_file=None):
    # Create output directory
    if out_image_file:
        out_dir = Path(out_image_file).parent
        out_dir.mkdir(parents=True, exist_ok=True)

    # Open image
    image = rxr.open_rasterio(image_file).squeeze()

    # Take the mean of all bands if multiband
    if "band" in list(image.dims):
        image = image.mean(dim="band")

    # Calculate min/max for scaling
    imin = image.min().data
    imax = image.max().data

    # Scale image from 0 to 255
    a = 255.0 / (imax - imin)
    b = 255.0 - a * imax
    scaled_image = (a * image + b).astype(np.uint8)

    # Re-attach CRS and other rioxarray-specific metadata
    rds_8bit = scaled_image.copy()
    rds_8bit = rds_8bit.rio.write_nodata(0) 

    # Save to file
    if out_image_file:
        rds_8bit.rio.to_raster(
            out_image_file, 
            driver="PNG", 
            dtype=np.uint8
        )
    return rds_8bit


def get_image_specs(
        image_list,
        out_file=None
        ):
    
    print("\nParsing image specs from file names")
    # Build dataframe with parsed image metadata
    specs_df_list = []
    for i, image_file in enumerate(image_list):
        # parse components from file name
        base = os.path.basename(image_file)
        pieces = base.split("_")

        # build metadata dataframe
        specs_df_list.append(pd.DataFrame({
            "image_path": [image_file],
            "image_file": [base],
            "datetime": ["_".join(pieces[0:2])],
            "sat": [pieces[2].split("d")[0]],
            "cam": ["d" + pieces[2].split("d")[1]],
            "frame": [pieces[3]],
        }, index=[i]))

    specs_df = pd.concat(specs_df_list)
    specs_df = specs_df.sort_values(
        by=["datetime", "sat", "cam", "frame"]
        ).reset_index(drop=True)

    print(f"Detected: "
          f"\n\t{len(specs_df["sat"].unique())} satellite(s)/rig(s)"
          f"\n\t{len(specs_df["cam"].unique())} cameras/sensors"
          f"\n\t{len(specs_df["frame"].unique())} frames"
          )

    # Write to file
    if out_file:
        specs_df.to_csv(out_file, index=False)
        print(f"Image specs file saved to:\n{out_file}")

    return specs_df


def write_rig_config(specs_df, tsai_dir, out_file):
    print("\nConstructing rig configuration file.")
    rigs = []

    # Group by rig ("sat" column)
    for rig_name, rig_df in specs_df.groupby("sat"):
        print(f"\tProcessing rig: {rig_name}")

        # --- Find a reference group of images (same frame, all rig cameras) ---
        print("\tIdentifying a reference image group for relative poses.")
        # Note: Corrected to find required sensors per rig, not from the global df
        required_sensors = set(rig_df["cam"].unique())
        grouper = ["datetime", "sat", "frame"]

        # Filter groups to find one that contains every sensor for this specific rig
        valid_groups = rig_df.groupby(grouper).filter(lambda g: set(g["cam"]) == required_sensors)
        if valid_groups.empty:
            print(f"\tERROR: Could not find a single frame containing all {len(required_sensors)} sensors for rig '{rig_name}'.")
            print("\tCannot calculate relative poses for this rig. Skipping.")
            continue
        
        # Take the first complete group as our reference for poses
        example_key, example_group = next(iter(valid_groups.groupby(grouper)))
        example_group = example_group.sort_values("cam").reset_index(drop=True)
        print(f"\tMatched group for relative poses: {example_key}")

        # --- Get absolute poses (R, C) for the reference group ---
        absolute_poses = {}
        for _, row in example_group.iterrows():
            tsai_path = Path(tsai_dir) / (Path(row["image_file"]).stem + ".tsai")
            if not tsai_path.exists():
                raise FileNotFoundError(f"\tTSAI file not found for {row['image_file']}")
            
            # The R matrix from parse_tsai is a flat array, so reshape it
            _, R_flat, C = parse_tsai(tsai_path)
            absolute_poses[row["cam"]] = (R_flat.reshape(3, 3), C)

        # --- Determine the reference sensor and get its absolute pose ---
        ref_sensor_name = rig_df["cam"].value_counts().idxmax()
        print(f"\tUsing sensor '{ref_sensor_name}' as reference for rig.")
        R_ref, C_ref = absolute_poses[ref_sensor_name]

        # --- Build the camera dictionaries for the rig ---
        rig_cameras = []
        # Use the unique sensors from the rig_df to build the final list
        for sensor_name in sorted(rig_df["cam"].unique()):
            sensor_df = rig_df[rig_df["cam"] == sensor_name]
            first_image = sensor_df.iloc[0]
            tsai_path = Path(tsai_dir) / (Path(first_image["image_file"]).stem + ".tsai")
            
            intrinsics, _, _ = parse_tsai(tsai_path)
            fx, fy, cx, cy = intrinsics
            prefix_path = Path(first_image["sat"]) / first_image["cam"]
            
            cam_dict = {
                "image_prefix": str(prefix_path),
                "ref_sensor": sensor_name == ref_sensor_name,
                "camera_model_name": "PINHOLE",
                "camera_params": [fx, fy, cx, cy],
            }

            # If not the reference sensor, calculate and add its relative pose
            if not cam_dict["ref_sensor"]:
                R_cam, C_cam = absolute_poses[sensor_name]

                # R_rel = R_ref * R_cam^T
                R_rel = R_ref @ R_cam.T
                
                # t_rel = R_ref * (C_cam - C_ref)
                t_rel = R_ref @ (C_cam - C_ref)

                # Convert rotation matrix to quaternion [qw, qx, qy, qz]
                # scipy returns [x, y, z, w], so we need to reorder for COLMAP
                quat_scipy = Rotation.from_matrix(R_rel).as_quat()
                quat_colmap = [quat_scipy[3], quat_scipy[0], quat_scipy[1], quat_scipy[2]]

                cam_dict["cam_from_rig_rotation"] = quat_colmap
                cam_dict["cam_from_rig_translation"] = t_rel.tolist()
            
            rig_cameras.append(cam_dict)

        rigs.append({"cameras": rig_cameras})

    # Write JSON
    with open(out_file, "w") as f:
        json.dump(rigs, f, indent=4)

    print(f"Rig configuration saved to {out_file}")
    return


def configure_reconstruction(specs_df, tsai_dir, image_dir, out_dir):
    print("\nBuilding pycolmap reconstruction object.")

    rec = pycolmap.Reconstruction()
    camera_name_to_id = {}
    rig_name_to_id = {}
    next_cam_id = 1
    next_rig_id = 1

    for rig_name, rig_df in specs_df.groupby("sat"):
        print(f"\tProcessing rig: {rig_name}")

        rig = pycolmap.Rig(rig_id=next_rig_id)
        rig_name_to_id[rig_name] = next_rig_id

        required_sensors = set(rig_df["cam"].unique())
        grouper = ["datetime", "sat", "frame"]
        valid_groups = rig_df.groupby(grouper).filter(lambda g: set(g["cam"]) == required_sensors)

        if valid_groups.empty:
            print(f"\tERROR: Could not find a frame with all sensors for rig '{rig_name}'. Skipping.")
            continue

        _, example_group = next(iter(valid_groups.groupby(grouper)))
        absolute_poses = {
            row["cam"]: parse_tsai(Path(tsai_dir) / (Path(row["image_file"]).stem + ".tsai"))[1:]
            for _, row in example_group.iterrows()
        }

        ref_sensor_name = rig_df["cam"].value_counts().idxmax()
        R_ref, C_ref = absolute_poses[ref_sensor_name]
        all_sensor_names = sorted(list(rig_df["cam"].unique()))

        def process_sensor(sensor_name, is_ref):
            nonlocal next_cam_id
            sensor_df = rig_df[rig_df["cam"] == sensor_name]

            if sensor_name not in camera_name_to_id:
                first_image_row = sensor_df.iloc[0]
                tsai_path = Path(tsai_dir) / (Path(first_image_row["image_file"]).stem + ".tsai")
                (fx, fy, cx, cy), _, _ = parse_tsai(tsai_path)
                image_path = Path(image_dir) / first_image_row["image_path"]

                with rio.open(image_path) as img:
                    height, width = img.shape

                camera = pycolmap.Camera(
                    model="PINHOLE", 
                    width=width, 
                    height=height, 
                    params=[fx, fy, cx, cy], 
                    camera_id=next_cam_id
                )
                rec.add_camera(camera)
                camera_name_to_id[sensor_name] = next_cam_id
                next_cam_id += 1

            camera_id = camera_name_to_id[sensor_name]
            sensor = pycolmap.sensor_t(
                id=camera_id, 
                type=pycolmap.SensorType.CAMERA
                )

            if is_ref:
                rig.add_ref_sensor(sensor)
            else:
                R_cam, C_cam = absolute_poses[sensor_name]
                R_rel = R_ref @ R_cam.T
                t_rel = R_ref @ (C_cam - C_ref)
                relative_pose = pycolmap.Rigid3d(
                    rotation=pycolmap.Rotation3d(R_rel), 
                    translation=t_rel
                    )
                rig.add_sensor(sensor, relative_pose)

        print(f"\tAdding reference sensor: {ref_sensor_name}")
        process_sensor(ref_sensor_name, is_ref=True)

        for sensor_name in all_sensor_names:
            if sensor_name == ref_sensor_name:
                continue
            print(f"\tAdding non-reference sensor: {sensor_name}")
            process_sensor(sensor_name, is_ref=False)
        
        rec.add_rig(rig)
        next_rig_id += 1

    print("\tRegistering images and frame poses.")
    next_image_id = 1
    next_frame_id = 1

    for key, frame_group_df in specs_df.groupby(["datetime", "sat", "frame"]):
        rig_name = key[1]
        rig_id = rig_name_to_id.get(rig_name)
        if rig_id is None:
            continue

        # Validate the data before creating the objects
        ref_sensor_name = specs_df[specs_df.sat == rig_name]["cam"].value_counts().idxmax()
        ref_image_row_series = frame_group_df[frame_group_df.cam == ref_sensor_name]
        if ref_image_row_series.empty:
            print(f"\tSkipping group {key} as it lacks the reference sensor '{ref_sensor_name}'.")
            continue  # No frame or ID is created
        
        # Create and add Frame
        frame = pycolmap.Frame(
            frame_id=next_frame_id, 
            rig_id=rig_id
            )
        rec.add_frame(frame)
        frame_in_rec = rec.frame(frame.frame_id)

        ref_image_row = ref_image_row_series.iloc[0]
        _, R_abs, C_abs = parse_tsai(Path(tsai_dir) / (Path(ref_image_row["image_file"]).stem + ".tsai"))
        R_world_to_cam = R_abs.T
        t_world_to_cam = -R_world_to_cam @ C_abs
        cam_from_world_pose = pycolmap.Rigid3d(
            rotation=pycolmap.Rotation3d(R_world_to_cam), translation=t_world_to_cam
        )
        ref_camera_id = camera_name_to_id[ref_sensor_name]
        frame_in_rec.set_cam_from_world(ref_camera_id, cam_from_world_pose)
        rec.register_frame(frame_in_rec.frame_id)

        for _, image_row in frame_group_df.iterrows():
            camera_id = camera_name_to_id[image_row["cam"]]
            image = pycolmap.Image(
                image_id=next_image_id, 
                name=image_row["image_path_8bit_relative"], 
                camera_id=camera_id, 
                frame_id=next_frame_id
            )
            frame_in_rec.add_data_id(image.data_id)
            rec.add_image(image)
            next_image_id += 1
            
        next_frame_id += 1

    # Save to file
    rec.write(out_dir)
    print(f"Initial reconstruction configuration saved to: \n{out_dir}")

    return rec


def populate_database_from_hloc(
    recon: pycolmap.Reconstruction,
    db: pycolmap.Database,  # <-- MODIFIED: Takes a pycolmap.Database object
    features_path: Path,
    matches_path: Path,
    pairs_path: Path,
):
    """
    Populates an OPEN pycolmap.Database with the model structure, features, 
    and verified matches, mirroring the logic of hloc.triangulation.
    """
    print("Populating database with HLOC features and matches...")

    image_ids = {image.name: image.image_id for image in recon.images.values()}
    
    # The 'with' statement is now REMOVED from this function.
    # The calling function (`main`) is responsible for managing the connection.

    # 1. Write the complete model structure to the database.
    print("Writing complete model structure to the database...")
    for cam_id, camera in recon.cameras.items():
        db.write_camera(camera, use_camera_id=True)
    for rig_id, rig in recon.rigs.items():
        db.write_rig(rig, use_rig_id=True)
    for frame_id, frame in recon.frames.items():
        db.write_frame(frame, use_frame_id=True)
    for img_id, image in recon.images.items():
        db.write_image(image, use_image_id=True)

    # 2. Import Features.
    print("Importing features into the database...")
    with h5py.File(features_path, 'r') as features:
        for image_name, image_id in tqdm(image_ids.items(), desc="Writing features"):
            if image_name not in features: continue
            keypoints = features[image_name]['keypoints'].__array__()
            keypoints += 0.5
            db.write_keypoints(image_id, keypoints)

    # 3. Import Matches.
    print("Importing matches into the database...")
    with open(str(pairs_path), "r") as f:
        pairs = [p.split() for p in f.readlines()]

    matched = set()
    with h5py.File(matches_path, 'r') as matches:
        for name0, name1 in tqdm(pairs, desc="Writing matches"):
            if name0 not in image_ids or name1 not in image_ids: continue
            id0, id1 = image_ids[name0], image_ids[name1]
            if len({(id0, id1), (id1, id0)} & matched) > 0: continue
            
            pair_key = '/'.join(sorted((name0, name1)))
            if pair_key not in matches: continue

            hloc_matches = matches[pair_key]['matches0'].__array__()
            kp_indices0 = np.arange(len(hloc_matches))
            valid_mask = hloc_matches != -1
            db_matches = np.stack([kp_indices0[valid_mask], hloc_matches[valid_mask]], axis=-1)

            db.write_matches(id0, id1, db_matches)
            matched |= {(id0, id1), (id1, id0)}

            tg = pycolmap.TwoViewGeometry(inlier_matches=db_matches)
            db.write_two_view_geometry(id0, id1, tg)

    print("Database population finished.")
    return


def main():

    # ----------------------------------------
    # 1. DEFINE ALL PATHS AND CLEAN UP PREVIOUS OUTPUTS
    # ----------------------------------------
    print("--- 1. Setting up paths and cleaning previous run outputs ---")
    data_folder = Path("/Users/rdcrlrka/Research/SkySat-Stereo/study-sites/MCS/20240420/")

    # --- Define all output directories ---
    out_folder = data_folder / "hloc_proc_out"
    tsai_dir = out_folder / "init_cams"
    new_image_folder = out_folder / "images"
    recon_folder = out_folder / "reconstruction"
    triang_folder = out_folder / "triangulation"

    # --- Define all output files ---
    specs_full_file = out_folder / "image_specs_full.csv"
    specs_filtered_file = out_folder / "image_specs_filtered.csv"
    original_pairs_file = out_folder / "stereo_image_pairs.txt"
    sfm_pairs_file = recon_folder / "pairs-sfm.txt"
    features_file = recon_folder / "features.h5"
    matches_file = recon_folder / "matches.h5"
    database_path = triang_folder / "database.db"

    # --- Clean up files from previous runs to ensure a fresh start ---
    files_to_clean = [
        specs_full_file, specs_filtered_file, original_pairs_file,
        sfm_pairs_file, database_path, #features_file, matches_file
    ]
    for f in files_to_clean:
        if f.exists():
            print(f"Deleting old output file: {f}")
            f.unlink()

    # --- Get original images ---
    image_list_original = sorted(data_folder.glob("SkySatScene/*analytic.tif"))
    print(f"Found {len(image_list_original)} original images.")

    # Create output directories
    for f in [
        out_folder, new_image_folder, recon_folder, triang_folder
    ]:
        f.mkdir(parents=True, exist_ok=True)


    # ----------------------------------------
    # 2. IMAGE PRE-PROCESSING
    # ----------------------------------------
    print("\n--- 2. Pre-processing images ---")
    specs_df = get_image_specs(image_list_original, out_file=specs_full_file)

    def convert_images_to_8bit(df):
        df["image_path_8bit"] = ""
        df["image_path_8bit_relative"] = ""
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Converting to 8-bit"):
            relative_path = Path(row["sat"]) / row["cam"] / f"{row['datetime']}_{row['frame']}.png"
            out_file = new_image_folder / relative_path
            out_file.parent.mkdir(parents=True, exist_ok=True)
            if not out_file.exists():
                save_image_as_8bit(row["image_path"], out_file)
            df.at[i, "image_path_8bit"] = str(out_file)
            df.at[i, "image_path_8bit_relative"] = str(relative_path)
        return df

    specs_df = convert_images_to_8bit(specs_df)

    # ----------------------------------------
    # 3. CONFIGURE RECONSTRUCTION
    # ----------------------------------------
    print("\n--- 3. Configuring reconstruction to find the valid set of images ---")
    recon = configure_reconstruction(specs_df, tsai_dir, new_image_folder, recon_folder)
    print(recon.summary())
    valid_relative_paths = {image.name for image in recon.images.values()}
    print(f"Reconstruction contains {len(valid_relative_paths)} valid images.")

    # ----------------------------------------
    # 4. FILTER AND IDENTIFY PAIRS
    # ----------------------------------------
    print("\n--- 4. Filtering data and identifying stereo pairs ---")
    specs_df_filtered = specs_df[specs_df['image_path_8bit_relative'].isin(valid_relative_paths)].copy()
    specs_df_filtered.to_csv(specs_filtered_file, index=False)
    print(f"Filtered specs DataFrame to {len(specs_df_filtered)} images.")
    valid_original_images = specs_df_filtered['image_path'].tolist()
    identify_overlapping_image_pairs(
        valid_original_images, min_overlap=10, utm_epsg="EPSG:32611", out_folder=out_folder
    )

    # ----------------------------------------
    # 5. CREATE HLOC-COMPATIBLE PAIRS FILE
    # ----------------------------------------
    print("\n--- 5. Creating HLOC-compatible pairs file ---")
    path_map = pd.Series(
        specs_df_filtered['image_path_8bit_relative'].values.astype(str),
        index=specs_df_filtered['image_path'].values.astype(str)
    ).to_dict()
    pairs_df = pd.read_csv(original_pairs_file, header=None, sep=" ", names=["img1_orig", "img2_orig"])
    pairs_df['img1_hloc'] = pairs_df['img1_orig'].map(path_map)
    pairs_df['img2_hloc'] = pairs_df['img2_orig'].map(path_map)
    final_pairs_df = pairs_df[['img1_hloc', 'img2_hloc']].dropna()
    final_pairs_df.to_csv(sfm_pairs_file, sep=" ", header=False, index=False)
    print(f"Saved {len(final_pairs_df)} final pairs to {sfm_pairs_file}")

    # ----------------------------------------
    # 6. FEATURE EXTRACTION AND MATCHING
    # ----------------------------------------
    # print("\n--- 6. Starting feature extraction and matching ---")
    # feature_conf = extract_features.confs["disk"]
    # matcher_conf = match_features.confs["disk+lightglue"]
    # image_list_for_hloc = specs_df_filtered['image_path_8bit_relative'].tolist()

    # extract_features.main(
    #     conf=feature_conf, 
    #     image_dir=new_image_folder, 
    #     image_list=image_list_for_hloc,
    #     feature_path=features_file, 
    #     overwrite=True
    # )
    # match_features.main(
    #     conf=matcher_conf, 
    #     pairs=sfm_pairs_file, 
    #     features=features_file,
    #     matches=matches_file, 
    #     overwrite=True
    # )

    # ----------------------------------------
    # 7. CREATE AND POPULATE THE DATABASE
    # ----------------------------------------
    print("\n--- 7. Creating and populating database from HLOC output ---")
    if database_path.exists():
        database_path.unlink()
    
    # Use a 'with' block here to manage the database connection
    with pycolmap.Database.open(database_path) as db:
        
        populate_database_from_hloc(
            recon=recon,
            db=db,
            features_path=features_file,
            matches_path=matches_file,
            pairs_path=sfm_pairs_file
        )

        # Your debugging prints will now work correctly inside this block
        print("\n--- Verifying database contents ---")
        print("db.num_keypoints:", db.num_keypoints())
        print("db.num_matches:", db.num_matches())
        print("db.num_matched_image_pairs:", db.num_matched_image_pairs())

    # ----------------------------------------
    # 8. TRIANGULATE POINTS
    # ----------------------------------------
    print("\n--- 8. Triangulating points using initial poses and database matches ---")
    
    # The rest of the script is now guaranteed to work because the database
    # was created and populated correctly.
    populated_recon = pycolmap.triangulate_points(
        reconstruction=recon,
        database_path=str(database_path),
        image_path=str(new_image_folder),
        output_path=str(recon_folder / "triangulated"),
        clear_points=True
    )
    
    print("\nAfter triangulation:")
    print(populated_recon.summary())

    # ----------------------------------------
    # 9. RUN BUNDLE ADJUSTMENT
    # ----------------------------------------
    print("\n--- 9. Running Bundle Adjustment to refine cameras and points ---")
    # Define configuration
    ba_config = pycolmap.BundleAdjustmentOptions()
    ba_config.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    ba_config.refine_focal_length = False
    ba_config.refine_principal_point = False
    ba_config.refine_extra_params = False
    ba_config.refine_rig_from_world = True
    ba_config.refine_sensor_from_rig = True
    ba_config.use_gpu = False
    ba_config.print_summary = True
    print(f"Bundle adjust configuration settings:\n{ba_config.todict()}")

    pycolmap.bundle_adjustment(populated_recon, options=ba_config)
    print("\nAfter bundle adjustment:")
    print(populated_recon.summary())

    # ----------------------------------------
    # 10. EXPORT REFINED DATA
    # ----------------------------------------
    # print("\n--- 10. Exporting refined reconstruction and camera poses ---")
    # refined_recon_folder = recon_folder / "refined"
    # refined_recon_folder.mkdir(exist_ok=True, parents=True)
    # populated_recon.write(refined_recon_folder)
    # print(f"Saved final refined model to {refined_recon_folder}")

    # print("\n--- Workflow finished successfully! ---")


if __name__ == "__main__":
    main()