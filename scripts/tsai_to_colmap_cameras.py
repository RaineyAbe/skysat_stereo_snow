import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="True"
from glob import glob
import re
import pycolmap
from pathlib import Path
import pyproj

def parse_tsai_model_for_center(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip().startswith('C ='):
                _, value = [x.strip() for x in line.split('=')]
                return np.fromstring(value, dtype=float, sep=' ')
    return None

def update_database_with_position_prior(db, image_name, position, covariance_matrix):
    if not db.exists_image(image_name):
        print(f"Warning: Image '{image_name}' not found in the database. Skipping prior.")
        return

    image = db.read_image(image_name)
    
    prior = pycolmap.PosePrior(
        position=position,
        position_covariance=covariance_matrix,
        coordinate_system=pycolmap.PosePriorCoordinateSystem.WGS84,
    )
    
    # Check if a prior already exists and update it, otherwise write a new one.
    if db.exists_pose_prior(image.image_id):
        db.update_pose_prior(image.image_id, prior)
    else:
        db.write_pose_prior(image.image_id, prior)

def main():
    """
    Main function to parse TSAI files and write position priors
    into the COLMAP database.
    """
    dataset_path = "/Users/rdcrlrka/Research/SkySat-Stereo/study-sites/MCS/20240420/colmap/"
    database_path = Path(os.path.join(dataset_path, "database.db"))
    init_cams_path = os.path.join(dataset_path, "init_cams")

    if not database_path.exists():
        print(f"Error: Database not found at '{database_path}'")
        return

    # --- Define the positional uncertainty for the priors ---
    prior_position_std_x = 5
    prior_position_std_y = 5
    prior_position_std_z = 5
    
    position_covariance = np.diag([
        prior_position_std_x**2,
        prior_position_std_y**2,
        prior_position_std_z**2,
    ])

    # --- Define coordinate systems and the transformer ---
    # Source: ECEF (Earth-Centered, Earth-Fixed) - EPSG:4978
    # Target: WGS84 Geodetic (lat, lon, height) - EPSG:4979
    ecef_crs = pyproj.CRS("EPSG:4978")
    wgs84_crs = pyproj.CRS("EPSG:4979")
    transformer = pyproj.Transformer.from_crs(ecef_crs, wgs84_crs, always_xy=False)

    # Open the database to perform all operations
    with pycolmap.Database().open(database_path) as db:
        
        # 1. Create a mapping from image name to image ID
        print("Reading images from database to create name-to-ID map...")
        image_name_to_id = {db.read_image(i).name: i for i in range(1, db.num_images()+1)}
        print(f"Found {len(image_name_to_id)} images in the database.")

        print(f"Image 1 ID: {image_name_to_id["ssc16/d1/20240420_165725_0006.png" ]}")
        print(f"Image 2 ID: {image_name_to_id["ssc16/d1/20240420_165753_0006.png"]}")

        # 2. Process the TSAI files and update priors using the map
        init_cams_files = sorted(glob(os.path.join(init_cams_path, "*.tsai")))
        print(f"Found {len(init_cams_files)} TSAI camera models. Updating database priors...")
        
        for tsai_path in init_cams_files:
            # Parse image info from camera file name
            tsai_basename = os.path.basename(tsai_path)
            match = re.match(r'(\d{8}_\d{6})_(ssc\d+)(d\d)_(\d{4})_.*\.tsai', tsai_basename)
            if not match:
                print(f"Warning: Could not parse filename format: {tsai_basename}")
                continue

            # Parse camera center
            camera_center_ecef = parse_tsai_model_for_center(tsai_path)
            if camera_center_ecef is None:
                print(f"Warning: Could not read camera center from {tsai_basename}")
                continue

            # Reproject to WGS84
            lat, lon, alt = transformer.transform(
                camera_center_ecef[0], camera_center_ecef[1], camera_center_ecef[2]
            )
            camera_center_wgs84 = np.array([lat, lon, alt])

            # Get the appropriate image ID
            timestamp, rig, camera, sequence_id = match.groups()
            image_filename = f"{timestamp}_{sequence_id}.png"
            colmap_relative_path = os.path.join(rig, camera, image_filename)
            image_id = image_name_to_id.get(colmap_relative_path)
            if image_id is None:
                print(f"Warning: Image '{colmap_relative_path}' not found in the database. Skipping prior.")
                continue
            
            # Update the database using the correct image_id
            update_database_with_position_prior(db, image_id, camera_center_wgs84, position_covariance)

    print(f"Successfully updated priors in '{database_path}'")

if __name__ == '__main__':
    main()
