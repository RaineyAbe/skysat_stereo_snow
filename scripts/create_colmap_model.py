import numpy as np
from scipy.spatial.transform import Rotation
import os
from glob import glob
import re
from pathlib import Path

def parse_tsai_model(file_path):
    """Parses a TSAI file to get intrinsics (fu, fv, cu, cv), center (C), and rotation (R)."""
    params = {}
    with open(file_path, 'r') as f:
        for line in f:
            if '=' in line:
                key, value = [x.strip() for x in line.split('=')]
                params[key] = np.fromstring(value, dtype=float, sep=' ')
    
    intrinsics = {k: params[k][0] for k in ['fu', 'fv', 'cu', 'cv']}
    C = params.get('C')
    R = params.get('R')
    if C is None or R is None:
        raise ValueError(f"Missing C or R in {file_path}")
        
    return intrinsics, C, R.reshape(3, 3)

def main():
    """
    Generates cameras.txt and images.txt from TSAI files to create an
    initial COLMAP model.
    """
    dataset_path = "/Users/rdcrlrka/Research/SkySat-Stereo/study-sites/MCS/20240420/colmap/"
    init_cams_path = os.path.join(dataset_path, "init_cams")
    output_model_path = Path(os.path.join(dataset_path, "sparse_initial"))

    # Ensure the output directory exists
    output_model_path.mkdir(exist_ok=True)
    
    cameras_txt_path = output_model_path / "cameras.txt"
    images_txt_path = output_model_path / "images.txt"
    
    # Create an empty points3D.txt, which is required by COLMAP
    (output_model_path / "points3D.txt").touch()

    init_cams_files = sorted(glob(os.path.join(init_cams_path, "*.tsai")))
    
    print(f"Found {len(init_cams_files)} TSAI models. Generating initial sparse model files...")

    camera_models = {} # To store unique camera intrinsics
    
    with open(images_txt_path, 'w') as f_images:
        # Write the required header for the images.txt file
        f_images.write("# Image list with two lines of data per image:\n")
        f_images.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f_images.write("#   POINTS2D[]\n")
        f_images.write("# Number of images: {}, mean observation length: 0\n".format(len(init_cams_files)))

        for image_id, tsai_path in enumerate(init_cams_files, 1):
            try:
                intrinsics, C, R_matrix = parse_tsai_model(tsai_path)
            except ValueError as e:
                print(f"Error: {e}")
                continue

            # --- Convert pose to COLMAP format (T = -R * C) ---
            T = -R_matrix @ C
            r = Rotation.from_matrix(R_matrix)
            quat_xyzw = r.as_quat()
            quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

            # --- Handle Camera Intrinsics ---
            # Group images with the same intrinsics under one camera model
            intrinsics_key = tuple(sorted(intrinsics.items()))
            if intrinsics_key not in camera_models:
                # Assign a new camera ID if these intrinsics are new
                camera_models[intrinsics_key] = len(camera_models) + 1
            camera_id = camera_models[intrinsics_key]
            
            # --- Get Image Name ---
            tsai_basename = os.path.basename(tsai_path)
            match = re.match(r'(\d{8}_\d{6})_(ssc\d+)(d\d)_(\d{4})_.*\.tsai', tsai_basename)
            if not match: continue
            
            timestamp, rig, camera, sequence_id = match.groups()
            image_filename = f"{timestamp}_{sequence_id}.png"
            colmap_relative_path = os.path.join(rig, camera, image_filename)

            # --- Write to images.txt ---
            f_images.write(
                f"{image_id} {' '.join(map(str, quat_wxyz))} {' '.join(map(str, T))} {camera_id} {colmap_relative_path}\n"
            )
            # Add a second, empty line for points, as required by the format
            f_images.write("\n")

    # --- Write cameras.txt ---
    with open(cameras_txt_path, 'w') as f_cameras:
        f_cameras.write("# Camera list with one line of data per camera:\n")
        f_cameras.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        # You'll need to know the width/height of your images. Let's use your example.
        WIDTH, HEIGHT = 2560, 1080 
        
        for intrinsics_key, camera_id in sorted(camera_models.items(), key=lambda item: item[1]):
            intrinsics = dict(intrinsics_key)
            f_cameras.write(
                f"{camera_id} PINHOLE {WIDTH} {HEIGHT} {intrinsics['fu']} {intrinsics['fv']} {intrinsics['cu']} {intrinsics['cv']}\n"
            )

    print(f"Successfully generated model files in {output_model_path}")

if __name__ == '__main__':
    main()
