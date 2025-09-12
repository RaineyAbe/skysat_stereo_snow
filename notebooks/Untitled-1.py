#!/usr/bin/env python3
"""
Full SfM using COLMAP (CLI) driven from Python, then export cameras to ASP .tsai.
Requirements:
  - COLMAP (installed and on PATH)
  - Optional: pycolmap (pip install pycolmap) for easier model reading
Usage: edit paths below and run.

Caveats:
 - For large datasets, prefer vocabulary/Sequential matcher settings; here we use
   exhaustive matcher for simplicity (works well for small-to-moderate sets).
 - COLMAP's mapper already performs bundle adjustment; you can tune options via CLI.
"""

import os
import sys
import subprocess
from glob import glob
import numpy as np

# Try to import pycolmap for easy parsing; if not available, fallback to text parsing
try:
    import pycolmap
    HAVE_PYCOLMAP = True
except Exception:
    HAVE_PYCOLMAP = False

# -------------------------
# User-editable paths
# -------------------------
COLMAP_BIN = "colmap"  # must be on PATH or provide full path to colmap executable
IMAGE_DIR = "/Users/raineyaberle/"    
PROJECT_WORK_DIR = "/path/to/colmap_work"
DATABASE = os.path.join(PROJECT_WORK_DIR, "database.db")
SPARSE_DIR = os.path.join(PROJECT_WORK_DIR, "sparse")
# optional: if you want model in text output (images.txt / cameras.txt):
SPARSE_TEXT_DIR = os.path.join(SPARSE_DIR, "0")  # mapper usually writes into .../sparse/0
OUT_TSAI_DIR = os.path.join(PROJECT_WORK_DIR, "tsai_out")
os.makedirs(PROJECT_WORK_DIR, exist_ok=True)
os.makedirs(SPARSE_DIR, exist_ok=True)
os.makedirs(OUT_TSAI_DIR, exist_ok=True)

# -------------------------
# Helper to run COLMAP CLI
# -------------------------
def run(cmd_args):
    print("RUN:", " ".join(cmd_args))
    p = subprocess.run(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        print("COLMAP command failed. Output:")
        print(p.stdout)
        raise RuntimeError("COLMAP command failed")
    return p.stdout

# -------------------------
# 1) Feature extraction
# -------------------------
def colmap_feature_extractor(image_dir, database_path, colmap_bin=COLMAP_BIN):
    cmd = [
        colmap_bin, "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_dir,
        # optional tuning:
        "--ImageReader.single_camera", "1",     # if all images share same intrinsics
        "--SiftExtraction.estimate_affine_shape", "0",
        "--SiftExtraction.domain_size_pooling", "0"
    ]
    return run(cmd)

# -------------------------
# 2) Exhaustive matching (or use vocab/Sequential if many images)
# -------------------------
def colmap_exhaustive_matcher(database_path, colmap_bin=COLMAP_BIN):
    cmd = [
        colmap_bin, "exhaustive_matcher",
        "--database_path", database_path,
        # optional tuning:
        "--SiftMatching.max_error", "4.0",
        "--SiftMatching.max_num_matches", "32768"
    ]
    return run(cmd)

# -------------------------
# 3) Mapper (reconstruction)
# -------------------------
def colmap_mapper(image_dir, database_path, sparse_dir, colmap_bin=COLMAP_BIN):
    # creates subfolder in sparse_dir like sparse/0
    cmd = [
        colmap_bin, "mapper",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--output_path", sparse_dir,
        "--Mapper.num_threads", "8",
        # tune mapper options as needed:
        "--Mapper.init_min_tri_angle", "4",
        "--Mapper.multiple_models", "0"  # disable multiple disconnected reconstructions unless you want them
    ]
    return run(cmd)

# -------------------------
# 4) (optional) Bundle adjuster CLI (mapper performs BA internally)
# -------------------------
def colmap_bundle_adjuster(sparse_model_dir, colmap_bin=COLMAP_BIN):
    # This calls the separate bundle_adjuster CLI if you want post-processing
    cmd = [
        colmap_bin, "bundle_adjuster",
        "--input_path", sparse_model_dir,
        "--output_path", sparse_model_dir,
        # tune LM options here if desired
    ]
    return run(cmd)

# -------------------------
# 5) Read COLMAP model and convert to .tsai
# -------------------------
def colmap_model_to_tsai(sparse_text_dir, out_tsai_dir, prefer_pycolmap=HAVE_PYCOLMAP):
    os.makedirs(out_tsai_dir, exist_ok=True)

    if HAVE_PYCOLMAP:
        print("Reading model with pycolmap...")
        recon = pycolmap.Reconstruction(sparse_text_dir)  # pycolmap can read a text model dir
        # iterate images
        for image in recon.images.values():
            # image.qvec (qw,qx,qy,qz), image.tvec (tx,ty,tz)
            q = image.qvec  # (4,)
            t = np.asarray(image.tvec).reshape(3,1)
            R = pycolmap.qvec2rotmat(q)  # rotation matrix (3x3) mapping world -> camera
            # camera center in world coords:
            C = (-R.T @ t).ravel()
            cam = recon.cameras[image.camera_id]
            # camera params differ by camera model; handle PINHOLE or SIMPLE_PINHOLE etc.
            if cam.model in ("PINHOLE", "SIMPLE_PINHOLE", "SIMPLE_RADIAL", "PINHOLE_RADIAL"):
                # Build K for common cases:
                if cam.model == "PINHOLE":
                    fx, fy, cx, cy = cam.params[0], cam.params[1], cam.params[2], cam.params[3]
                elif cam.model == "SIMPLE_PINHOLE":
                    fx, cx, cy = cam.params[0], cam.params[1], cam.params[2]
                    fy = fx
                elif cam.model == "SIMPLE_RADIAL":
                    fx, cx, cy = cam.params[0], cam.params[1], cam.params[2]
                    fy = fx
                else:
                    # fallback: try to extract first two params as fx,fy and next two as cx,cy
                    fx = cam.params[0]
                    fy = cam.params[1] if len(cam.params)>1 else fx
                    cx = cam.params[2] if len(cam.params)>2 else 0.0
                    cy = cam.params[3] if len(cam.params)>3 else 0.0
                K = np.array([[fx, 0, cx],[0, fy, cy],[0,0,1]], dtype=np.float64)
                # R is world->camera as documented
                out_name = os.path.splitext(image.name)[0]
                out_path = os.path.join(out_tsai_dir, f"{out_name}.tsai")
                write_tsai(out_path, K, R, C)
                print("Wrote:", out_path)
            else:
                print(f"Skipping camera model {cam.model} for image {image.name}")
        return

    # Fallback: parse COLMAP text files manually
    # images.txt + cameras.txt should be in sparse_text_dir
    images_txt = os.path.join(sparse_text_dir, "images.txt")
    cameras_txt = os.path.join(sparse_text_dir, "cameras.txt")
    if not (os.path.exists(images_txt) and os.path.exists(cameras_txt)):
        raise RuntimeError("No text model found and pycolmap not installed. Make sure COLMAP produced sparse text model in folder: " + sparse_text_dir)

    # Parse cameras.txt (simplified)
    cam_intrinsics = {}  # camera_id -> (model, params)
    with open(cameras_txt, 'r') as f:
        for line in f:
            if line.strip()=="" or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]
            cam_intrinsics[cam_id] = (model, params)

    # Parse images.txt
    with open(images_txt, 'r') as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()!=""]
    i=0
    while i < len(lines):
        if lines[i].startswith("#"):
            i+=1
            continue
        # line with image info
        parts = lines[i].split()
        image_id = int(parts[0])
        qvec = [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])]  # qw qx qy qz
        tvec = [float(parts[5]), float(parts[6]), float(parts[7])]
        camera_id = int(parts[8])
        image_name = parts[9]
        # compute rotation matrix from qvec:
        q = np.asarray(qvec, dtype=np.float64)
        R = qvec_to_rotmat(q)  # we'll implement this helper below
        t = np.asarray(tvec).reshape(3,1)
        C = (-R.T @ t).ravel()
        # get intrinsics for this camera
        model, params = cam_intrinsics[camera_id]
        if model in ("PINHOLE", "SIMPLE_PINHOLE", "SIMPLE_RADIAL", "PINHOLE_RADIAL"):
            if model == "PINHOLE":
                fx, fy, cx, cy = params[0], params[1], params[2], params[3]
            elif model == "SIMPLE_PINHOLE":
                fx, cx, cy = params[0], params[1], params[2]; fy = fx
            else:
                fx, cx, cy = params[0], params[1], params[2]; fy = fx
            K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
            out_name = os.path.splitext(image_name)[0]
            out_path = os.path.join(out_tsai_dir, f"{out_name}.tsai")
            write_tsai(out_path, K, R, C)
            print("Wrote:", out_path)
        else:
            print("Skipping camera model", model, "for image", image_name)
        # skip next line (points2D) as format uses alternating lines
        i += 2

# -------------------------
# Helpers: quaternion -> rot matrix, tsai writer
# -------------------------
def qvec_to_rotmat(qvec):
    # qvec is [qw, qx, qy, qz] quaternion (COLMAP convention)
    qw, qx, qy, qz = qvec
    # Build rotation matrix (Hamilton convention)
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,       2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,         1 - 2*qx*qx - 2*qz*qz,   2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,         2*qy*qz + 2*qx*qw,       1 - 2*qx*qx - 2*qy*qy]
    ], dtype=np.float64)
    return R

def write_tsai(path, K, Rwc, C, pitch=0.8):
    # Rwc is WORLD->CAMERA rotation
    fu, fv = float(K[0,0]), float(K[1,1])
    cu, cv = float(K[0,2]), float(K[1,2])
    with open(path, 'w') as f:
        f.write("VERSION_4\n")
        f.write("PINHOLE\n")
        f.write(f"fu = {fu}\n")
        f.write(f"fv = {fv}\n")
        f.write(f"cu = {cu}\n")
        f.write(f"cv = {cv}\n")
        f.write("u_direction = 1 0 0\n")
        f.write("v_direction = 0 1 0\n")
        f.write("w_direction = 0 0 1\n")
        f.write(f"C = {C[0]} {C[1]} {C[2]}\n")
        for row in Rwc:
            f.write("R = " + " ".join(f"{v:.16g}" for v in row) + "\n")
        f.write(f"pitch = {pitch}\n")

# -------------------------
# Main sequence
# -------------------------
if __name__ == "__main__":
    # 1) extract features
    print("1) Extracting features to database:", DATABASE)
    colmap_feature_extractor(IMAGE_DIR, DATABASE)

    # 2) match features
    print("2) Running exhaustive matcher")
    colmap_exhaustive_matcher(DATABASE)

    # 3) run mapper (reconstruction + BA)
    print("3) Running mapper (this performs triangulation and BA)")
    colmap_mapper(IMAGE_DIR, DATABASE, SPARSE_DIR)

    # if you want, explicitly run bundle_adjuster on the resulting sparse model dir:
    # print("4) Running bundle_adjuster")
    # colmap_bundle_adjuster(os.path.join(SPARSE_DIR, "0"))

    # 4) convert model to tsai files
    print("5) Converting COLMAP model to .tsai camera files")
    # The mapper usually writes model(s) into sparse/0
    model0_dir = os.path.join(SPARSE_DIR, "0")
    if not os.path.isdir(model0_dir):
        # sometimes mapper writes directly to SPARSE_DIR
        model0_dir = SPARSE_DIR
    colmap_model_to_tsai(model0_dir, OUT_TSAI_DIR)
    print("Done. .tsai files in:", OUT_TSAI_DIR)
