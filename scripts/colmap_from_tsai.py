#!/usr/bin/env python3

import os
import argparse
import numpy as np
import rioxarray as rxr
from glob import glob
import sqlite3
import pandas as pd
from tqdm import tqdm
import subprocess
import json
import pycolmap

# rasterio doesn't like when images are georeferenced, so ignore those warnings.
import warnings
import rasterio
warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)


def run_cmd(
        bin: str = None, 
        args: list = None, **kw
        ) -> str:
    # binpath = shutil.which(bin)
    # if binpath.endswith('.py'):
    #     call = ['python', binpath,]
    # else:
    #     call = [binpath,]
    # if args is not None: 
    #     call.extend(args)
    call = [bin] + args
    try:
        out = subprocess.run(call,check=True,capture_output=True,encoding='UTF-8').stdout
    except:
        out = f"the command {call} failed to run."
    return out


def parse_tsai(tsai_path):
    with open(tsai_path, 'r') as f:
        lines = f.readlines()
        lines = [x.replace('\n','') for x in lines]

    def get_value(lines, k):
        line = [x for x in lines if k in x][0]
        val_string = line.split(' = ')[1]
        if ' ' in val_string:
            return np.array(val_string.split(' ')).astype(float)
        else:
            return float(val_string)
        
    # fx,fy,cx,cy
    fx = get_value(lines, 'fu =')
    fy = get_value(lines, 'fv =')
    cx = get_value(lines, 'cu =')
    cy = get_value(lines, 'cv =')

    # rotation matrix
    R = get_value(lines, 'R =')

    # camera center
    C = get_value(lines, 'C =')

    return (fx, fy, cx, cy), R, C


# Function to scale data to 0-255 range and convert to uint8
def save_image_as_8bit(image_file, out_image_file=None):
    # Create output directory
    if out_image_file:
        out_dir = os.path.dirname(out_image_file)
        os.makedirs(out_dir, exist_ok=True)

    # Open image
    image = rxr.open_rasterio(image_file).squeeze()

    # Take the mean of all bands if multiband
    if 'band' in list(image.dims):
        image = image.mean(dim='band')

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
    
    print('Configuring rig(s).')
    # Build dataframe with parsed image metadata
    specs_df_list = []
    for i, image_file in enumerate(image_list):
        # parse components from file name
        base = os.path.basename(image_file)
        pieces = base.split('_')

        # build metadata dataframe
        specs_df_list.append(pd.DataFrame({
            'image_path': [image_file],
            'image_file': [base],
            'datetime': ['_'.join(pieces[0:2])],
            'sat': [pieces[2].split('d')[0]],
            'cam': ['d' + pieces[2].split('d')[1]],
            'sensor': [pieces[2]],
            'frame': [pieces[3]],
        }, index=[i]))

    specs_df = pd.concat(specs_df_list)
    specs_df = specs_df.sort_values(by=['datetime', 'sensor', 'frame']).reset_index(drop=True)

    # Extract image sizes per sensor
    sensors = specs_df['sensor'].unique()
    sensor_sizes = {}
    for sensor in sensors:
        # take first image for that sensor
        example_path = specs_df.loc[specs_df.sensor == sensor, "image_path"].iloc[0]
        h,w = rxr.open_rasterio(example_path).shape[1:]
        sensor_sizes[sensor] = (w, h)

    print(f"Detected {len(sensors)} sensors.")

    # Write to file
    if out_file:
        specs_df.to_csv(out_file, index=False)
        print(f"Image specs file saved to:\n{out_file}")

    return specs_df


def write_rig_config(
        specs_df, 
        tsai_dir, 
        out_file
        ):
    rigs = []
    # Group by rig (assuming 'sat' identifies a rig)
    for rig_name, rig_df in specs_df.groupby("sat"):
        print('Processing rig:', rig_name)
        rig_cameras = []

        # Identify a reference triplet pair for relative poses
        # rig # = same, camera # = different, frame # = same
        print("Identifying a reference image group for relative poses")
        required_sensors = set(specs_df['sensor'].unique().tolist())
        grouper = ["datetime", "sat", "frame"]
        # filter groups where sensor set matches the required set
        valid_groups = (
            specs_df.groupby(grouper)
                    .filter(lambda g: set(g["sensor"]) == required_sensors)
                    .groupby(grouper)
        )
        # take the first matching group
        example_key, example_group = next(iter(valid_groups))
        example_group = example_group.sort_values("sensor").reset_index(drop=True)
        print("Matched group:", example_key)
        print(example_group)

        # Use the sensor with the most images as the reference
        ref_sensor_name = rig_df['sensor'].value_counts().idxmax()
        print(f'Using sensor {ref_sensor_name} as reference for rig {rig_name}.')

        # Group by sensor
        for sensor_name, sensor_df in rig_df.groupby("sensor"):
            # Take first image to get intrinsics
            first_image = sensor_df.iloc[0]
            tsai_path = os.path.join(tsai_dir, os.path.splitext(first_image['image_file'])[0] + '.tsai')
            if not os.path.exists(tsai_path):
                raise FileNotFoundError(f"TSai file not found for {first_image}")
            # Parse TSAI for intrinsics
            (fx, fy, cx, cy), R, C = parse_tsai(tsai_path)

            # Build image prefix relative to image_dir_8bit
            prefix_path = os.path.join(first_image['sat'], first_image['sensor'])

            # For simplicity, make first sensor the reference
            ref_sensor = True if sensor_name == ref_sensor_name else False

            cam_dict = {
                "image_prefix": prefix_path,
                "ref_sensor": ref_sensor,
                "camera_model_name": "PINHOLE",
                "camera_params": [fx, fy, cx, cy],
            }

            rig_cameras.append(cam_dict)

        rigs.append({"cameras": rig_cameras})

    # Write JSON
    with open(out_file, 'w') as f:
        json.dump(rigs, f, indent=4)
    print(f"Rig configuration saved to {out_file}")



def get_image_dimensions(image_path):
    """Opens an image and returns its width and height."""
    with rxr.open_rasterio(image_path) as img:
        return img.shape[2], img.shape[1]

def setup_rig(database, specs_df, tsai_dir):
    print("==============================================================================")
    print("Setting up rigs in the pycolmap database")
    print("==============================================================================")
    
    # This will store the final rig objects to be added to the DB
    all_rigs = []

    for rig_name, rig_df in specs_df.groupby("sat"):
        print(f'Processing rig: {rig_name}')
        
        # Lists to hold cameras and their relative poses for this rig
        cameras_for_rig = []
        relative_poses = []
        
        # --- Determine the Reference Sensor for this Rig ---
        # We need its absolute pose to calculate relative poses for others
        ref_sensor_name = rig_df['sensor'].value_counts().idxmax()
        ref_row = rig_df[rig_df['sensor'] == ref_sensor_name].iloc[0]
        ref_tsai_path = os.path.join(tsai_dir, os.path.splitext(ref_row['image_file'])[0] + '.tsai')
        (ref_fx, ref_fy, ref_cx, ref_cy), ref_R, ref_C = parse_tsai(ref_tsai_path) # Assumes parse_tsai returns intrinsics too
        
        # The world-to-ref_camera pose
        # ref_pose_abs = pycolmap.Pose(R=ref_R, t=-ref_R @ ref_C)

        # --- Iterate over each sensor to create cameras and relative poses ---
        for sensor_name in sorted(rig_df['sensor'].unique()):
            sensor_df = rig_df[rig_df['sensor'] == sensor_name]
            first_image_row = sensor_df.iloc[0]

            # Get image dimensions (width, height) - THIS IS A REQUIRED STEP
            image_path_8bit = first_image_row['image_path'] # Assumes specs_df has the path to the 8-bit png
            width, height = get_image_dimensions(image_path_8bit)
            
            # Get intrinsics and extrinsics from its .tsai file
            tsai_path = os.path.join(tsai_dir, os.path.splitext(first_image_row['image_file'])[0] + '.tsai')
            (fx, fy, cx, cy), R, C = parse_tsai(tsai_path)
            
            # 1. Create the pycolmap.Camera object
            cam = pycolmap.Camera(
                model='PINHOLE',
                width=width,
                height=height,
                params=[fx, fy, cx, cy]
            )
            cameras_for_rig.append(cam)

            # 2. Calculate the relative pose from the reference sensor
            # The world-to-current_camera pose
            # current_pose_abs = pycolmap.Pose(R=R, t=-R @ C)
            
            # # Relative Pose (from ref to current) = T_ref_current = T_world_current * T_world_ref^-1
            # relative_pose = current_pose_abs @ ref_pose_abs.inverse()
            # relative_poses.append(relative_pose)
            
            print(f"  - Processed sensor: {sensor_name}")
            if sensor_name == ref_sensor_name:
                print(f"    (Reference sensor)")


        # 3. Instantiate the Rig with the collected cameras and relative poses
        # The reference camera is implicitly the first one in the list.
    #     rig = pycolmap.Rig(cameras=cameras_for_rig)
    #     all_rigs.append(rig)

    # # 4. Add all created rigs to the database
    # for rig in all_rigs:
    #     database.add_rig(rig)
        
    # print(f"\nSuccessfully added {len(all_rigs)} rig(s) to the database.")



def main(args):
    image_dir = args.image_dir
    tsai_dir = args.tsai_dir
    db_path = args.database
    output_dir = args.output_dir

    # Get the images
    image_list = sorted(glob(os.path.join(image_dir, '*.tif')))

    # Extract rigs + cameras from image file names
    specs_file = os.path.join(output_dir, 'image_specs.csv')
    specs_df = get_image_specs(image_list, out_file=specs_file)
    
    # Convert images to 8-bit PNGs, saving in appropriate rig/sensor folders
    print('Saving images as 8-bit PNGs for bundle adjustment.')
    image_dir_8bit = os.path.join(output_dir, 'images_8bit')
    for image_file in tqdm(image_list):
        rig, sensor = specs_df.loc[specs_df['image_path']==image_file, ['sat', 'sensor']].values[0]
        image_out_file = os.path.join(
            image_dir_8bit, 
            rig,
            sensor,
            os.path.splitext(os.path.basename(image_file))[0] + '.png'
            )
        if os.path.exists(image_out_file):
            continue
        save_image_as_8bit(image_file, image_out_file)
    
    # Remove auxilliary files
    aux_files = glob(os.path.join(image_dir_8bit, '*', '*', '*.xml'))
    _ = [os.remove(x) for x in aux_files]

   # Get new image list
    image_list = sorted(glob(os.path.join(image_dir_8bit, '*', '*', '*.png')))

    # Update the specs_df with new file names
    specs_df['image_path'] = specs_df.apply(
        lambda row: os.path.join(
            image_dir_8bit, 
            row['sat'],
            row['sensor'],
            os.path.splitext(row['image_file'])[0] + '.png'
            ), axis=1
            )
    specs_df['image_file'] = specs_df['image_file'].apply(lambda x: os.path.splitext(x)[0]+'.png')
    specs_updated_file = os.path.splitext(specs_file)[0] + '_8bit.csv'
    specs_df.to_csv(specs_updated_file, index=False)
    print(f'Updated image specs saved to file:\n{specs_updated_file}')

    # Create / open database
    if not os.path.exists(db_path):
        print(f"Created empty database file at {db_path}")
    db = pycolmap.Database().open(db_path)

    # Configure the rig
    print("==============================================================================")
    print("Rig configuration")
    print("==============================================================================")

    # Create rig
    setup_rig(db, specs_df, tsai_dir)

    # Create rig configuration file
    # rig_config_file = os.path.join(output_dir, "rig_config.json")
    # write_rig_config(specs_df, tsai_dir, rig_config_file)

    # Add to database

    
    
    
    # args = [
    #     "colmap", "rig_configurator",
    #     "--database_path", db_path,
    #     "--rig_config_path", rig_config_file,
    #     # f"[ --output_path {os.path.join(db_path, "sparse-model-with-rigs-and-frames")} ]"
    # ]
    # print(args)
    # subprocess.run(args)

    # # Match features
    # # Exhaustive works for smaller sets (up to hundreds)
    # args = [
    #     "colmap", "exhaustive_matcher",
    #     "--database_path", db_path,
    #     "--FeatureMatching.use_gpu", "0"
    # ]
    # subprocess.run(args)

    # # Sparse mapper
    # sparse_dir = os.path.join(output_dir, "sparse")
    # os.makedirs(sparse_dir, exist_ok=True)
    # args = [
    #     "colmap", "mapper",
    #     "--database_path", db_path, 
    #     "--image_path", image_dir_8bit,
    #     "--output_path", sparse_dir,
    #     "--Mapper.ba_use_gpu", "0",
    #     "--Mapper.ba_refine_focal_length", "0",
    #     "--Mapper.ba_refine_principal_point", "0",
    #     "--Mapper.ba_refine_extra_params", "0"
    # ]

    # # Dense mapper
    # dense_dir = os.path.join(output_dir, "dense")
    # os.makedirs(dense_dir, exist_ok=True)
    # args = [
    #     "colmap" "patch_match_stereo",
    #     "--workspace_path", dense_dir,
    #     "--workspace_format", "COLMAP",
    #     "--PatchMatchStereo.geom_consistency", "true"
    # ]

   
if __name__ == "__main__":
    base_path = "/Users/rdcrlrka/Research/SkySat-Stereo/study-sites/MCS/20240420/colmap"
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "--image-dir", 
        default=os.path.join(base_path, "images"),
        help="Directory containing .tif images"
        )
    p.add_argument(
        "--tsai-dir", 
        default=os.path.join(base_path, "cams"),
        help="Directory containing .tsai files (named as image + .tsai)"
        )
    p.add_argument(
        "--database", 
        default=os.path.join(base_path, "database.db"), 
        help="Path to COLMAP database file"
        )
    p.add_argument(
        "--output-dir", 
        default=base_path, 
        help="Base output directory"
        )
    args = p.parse_args()
    main(args)

    