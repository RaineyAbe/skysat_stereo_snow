# OpenCV multi-view SfM with bundle adjustment (extrinsics-only) using stitching.detail
# - Reads separate intrinsics per image from .tsai files
# - Keeps intrinsics fixed and refines extrinsics (R, t) for ALL images
# - Uses OpenCV "detail" pipeline: features -> pairwise matches -> initial cameras -> BA
# - Supports multispectral images by selecting a band or combining into intensity
# - Writes adjusted .tsai files per image

import cv2
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------------------
# Utilities: .tsai I/O
# ------------------------------
def read_tsai_intrinsics(path):
    fu = fv = cu = cv = None
    with open(path) as f:
        for line in f:
            ls = line.strip()
            if ls.startswith('fu'):
                fu = float(ls.split('=')[1])
            elif ls.startswith('fv'):
                fv = float(ls.split('=')[1])
            elif ls.startswith('cu'):
                cu = float(ls.split('=')[1])
            elif ls.startswith('cv'):
                cv = float(ls.split('=')[1])
    if None in (fu, fv, cu, cv):
        raise ValueError(f"Missing intrinsics in {path}")
    K = np.array([[fu, 0,  cu],
                  [0,  fv, cv],
                  [0,   0,  1]], dtype=np.float64)
    return K

def write_tsai(path, K, Rwc, C, pitch=0.8):
    """
    Write an ASP .tsai file.
    - K: 3x3 intrinsics
    - Rwc: rotation from WORLD -> CAMERA
    - C: camera center in WORLD coords (x y z)
    """
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
        Rwc_ravel = np.array(Rwc).ravel()
        f.write("R = " + " ".join(f"{v:.16g}" for v in Rwc_ravel) + "\n")
        f.write(f"pitch = {pitch}\n")

# ------------------------------
# Imaging helpers
# ------------------------------
def load_image_uint8(path, band_index=None):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Failed to read image: {path}")

    # Force uint8 for SIFT
    if img.dtype != np.uint8:
        # If input is >8 bit, scale robustly into 8-bit
        mn, mx = float(np.nanmin(img)), float(np.nanmax(img))
        if mx > mn:
            img = np.clip((img - mn) * (255.0 / (mx - mn)), 0, 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    if img.ndim == 3 and img.shape[2] > 1:
        if band_index is not None:
            img = img[:, :, band_index]
        else:
            img = img.mean(axis=2).astype(np.uint8)

    return img

# ------------------------------
# OpenCV "detail" stitching pipeline wrappers
# ------------------------------
def compute_features(images, sift=None):
    if sift is None:
        sift = cv2.SIFT_create()
    feats = []
    for im in tqdm(images):
        # detail.computeImageFeatures2 expects a feature2d and image
        feat = cv2.detail.computeImageFeatures2(sift, im)
        feats.append(feat)
    return feats

def pairwise_match(features, match_conf=0.3, try_use_gpu=False):
    matcher = cv2.detail_BestOf2NearestMatcher(try_use_gpu, match_conf)
    pairwise_matches = matcher.apply2(features)
    matcher.collectGarbage()
    return pairwise_matches

def estimate_initial_cameras(features, pairwise_matches, intrinsics_list):
    """
    Use HomographyBasedEstimator to get an initial rotation graph, then
    inject fixed intrinsics from .tsai for each camera.
    """
    # Estimate initial rotations (this does not commit to our intrinsics)
    estimator = cv2.detail_HomographyBasedEstimator()
    success, cameras = estimator.apply(features, pairwise_matches, None)
    if not success:
        raise RuntimeError("Initial camera estimation failed.")

    # Overwrite intrinsics with provided K; set aspect, ppx, ppy, focal.
    for i, cam in enumerate(cameras):
        K = intrinsics_list[i]
        fu, fv = K[0,0], K[1,1]
        cu, cv = K[0,2], K[1,2]
        cam.focal = float((fu + fv) * 0.5)
        cam.aspect = float(fv / fu if fu != 0 else 1.0)
        cam.ppx = float(cu)
        cam.ppy = float(cv)
        # Keep the estimated rotation; translation will be solved in BA
        # Ensure R is a proper 3x3 double
        cam.R = np.asarray(cam.R, dtype=np.float64)
        cam.t = np.zeros((3,1), dtype=np.float64)

    return cameras

def run_bundle_adjustment(features, pairwise_matches, cameras, refine_intrinsics=False, conf_thresh=0.05):
    """
    Runs reprojection bundle adjustment. If refine_intrinsics=False (default),
    intrinsics are held fixed and only extrinsics (R, t) are refined.
    """
    ba = cv2.detail_BundleAdjusterReproj()
    ba.setConfThresh(conf_thresh)

    # Refinement mask (3x3) controls which intrinsic params are optimized:
    # According to OpenCV stitching docs, the mask elements toggle:
    # [ [focal, aspect, ppx],
    #   [  0  ,  fy  , ppy],
    #   [  0  ,   0  ,  0 ] ]
    # In practice for stitching, they set [0,0]=[0,1]=[0,2]=[1,1]=[1,2]=1 to refine (f,aspect,ppx,ppy).
    # We want to keep intrinsics fixed, so all zeros:
    refine_mask = np.zeros((3,3), np.uint8)
    if refine_intrinsics:
        refine_mask = np.ones((3,3), np.uint8)
    ba.setRefinementMask(refine_mask)

    # make sure cameras data type == float32
    for cam in cameras:
        cam.R = cam.R.astype(np.float32)
        cam.t = cam.t.astype(np.float32)

    # remove garbage matches
    # conf_thresh = 0.3  # same as your matcher threshold
    # ikeep = cv2.detail.leaveBiggestComponent(
    #     features, pairwise_matches, conf_thresh
    # )
    # features = [features[i] for i in ikeep]
    # pairwise_matches = []

    for pm in pairwise_matches:
        print(pm.src_img_idx, pm.dst_img_idx, pm.confidence)

    # for i, m in enumerate(pairwise_matches):
    #     print(f"Image {m.src_img_idx} -> {m.dst_img_idx}, matches: {len(m.matches)}, inliers: {np.sum(m.inliers_mask)}")
    # edges = [(m.src_img_idx, m.dst_img_idx) for m in pairwise_matches if np.sum(m.inliers_mask) > 0]
    # print("Edges:", edges)

    ok, cameras = ba.apply(features, pairwise_matches, cameras)
    if not ok:
        raise RuntimeError("Bundle adjustment failed.")
    
    for i, cam in enumerate(cameras):
        print(f'\nCamera {i}:')
        print("R = ", cam.R)
        print("t = ", cam.t)

    return cameras

# ------------------------------
# Conversions between OpenCV detail CameraParams and ASP .tsai
# ------------------------------
def camera_matrix_from_detail(cam):
    """Build K from detail::CameraParams (focal, aspect, ppx, ppy)."""
    f = cam.focal
    a = cam.aspect if cam.aspect != 0 else 1.0
    ppx, ppy = cam.ppx, cam.ppy
    K = np.array([[f, 0, ppx],
                  [0, f * a, ppy],
                  [0, 0, 1]], dtype=np.float64)
    return K

def extrinsics_world_camera_from_detail(cam):
    """
    OpenCV stitching stores cam.R and cam.t. In practice, cam.R behaves like
    CAMERA->WORLD rotation (R_cw) in that the absolute camera center C is near:
      C = -R_cw^T * t
    The projection uses WORLD->CAMERA rotation R_wc = R_cw^T and translation t_wc.
    We convert to:
      R_wc = R_cw^T
      C = -R_wc^T * t_wc = -R_cw * t_wc
    With the detail module's BA, t is estimated in a consistent coordinate frame.
    We'll compute:
      R_wc = cam.R.T
      C    = -cam.R.dot(cam.t).ravel()
    """
    R_cw = np.asarray(cam.R, dtype=np.float64)
    t = np.asarray(cam.t, dtype=np.float64).reshape(3,1)
    R_wc = R_cw.T
    C = (-R_cw @ t).ravel()
    return R_wc, C

def tsai_from_detail_camera(cam, prefer_original_K=None):
    """
    Build (K, R_wc, C) for .tsai writing.
    If prefer_original_K is provided, we return that K (keeping intrinsics fixed).
    Otherwise we reconstruct K from cam's stored params.
    """
    if prefer_original_K is not None:
        K = prefer_original_K
    else:
        K = camera_matrix_from_detail(cam)
    R_wc, C = extrinsics_world_camera_from_detail(cam)
    return K, R_wc, C

# ------------------------------
# Visualization (optional)
# ------------------------------
def plot_matches_for_pair(img1, img2, match_info, feats1, feats2, max_draw=200):
    # Get matches as DMatch list
    matches = []
    for m in tqdm(match_info.matches):
        dm = cv2.DMatch(_queryIdx=m.queryIdx, _trainIdx=m.trainIdx, _imgIdx=0,
                        _distance=m.distance)
        matches.append(dm)
    matches = sorted(matches, key=lambda d: d.distance)
    matches = matches[:max_draw]

    # Convert keypoints
    kp1 = [cv2.KeyPoint(x=kp.pt[0], y=kp.pt[1], size=kp.size,
                        angle=kp.angle, response=kp.response,
                        octave=kp.octave, class_id=kp.class_id)
           for kp in feats1.keypoints]
    kp2 = [cv2.KeyPoint(x=kp.pt[0], y=kp.pt[1], size=kp.size,
                        angle=kp.angle, response=kp.response,
                        octave=kp.octave, class_id=kp.class_id)
           for kp in feats2.keypoints]

    vis = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)
    plt.figure(figsize=(12,6))
    plt.imshow(vis)
    plt.title("Feature Matches (subset)")
    plt.axis('off')
    plt.show()

# ------------------------------
# Main pipeline
# ------------------------------
def main(
    img_list,
    tsai_list,
    out_cam_folder=None,
    band_index=None,
    match_conf=0.05,
    refine_intrinsics=False,
    conf_thresh=0.05,
    visualize_first_pair=False
):

    if out_cam_folder is None:
        out_cam_folder = os.path.join(cam_folder, "ba_out_tsai")
    os.makedirs(out_cam_folder, exist_ok=True)

    # Load images + intrinsics
    images_u8 = [load_image_uint8(p, band_index=band_index) for p in img_list]
    intrinsics_list = [read_tsai_intrinsics(p) for p in tsai_list]
    print(f"Loaded {len(images_u8)} images and {len(intrinsics_list)} .tsai intrinsics.")

    print("Computing features...")
    features = compute_features(images_u8)

    print("Computing pairwise matches...")
    pairwise_matches = pairwise_match(features, match_conf=match_conf, try_use_gpu=False)

    # Optional visualization for the first *connected* pair
    if visualize_first_pair:
        # Find strongest connected pair and visualize
        best_pair = None
        best_score = -1.0
        for i in range(len(images_u8)):
            for j in range(i+1, len(images_u8)):
                info = pairwise_matches[i * len(images_u8) + j]  # flat index layout
                if info.confidence > best_score and len(info.matches) > 0:
                    best_score = info.confidence
                    best_pair = (i, j, info)
        if best_pair is not None:
            i, j, info = best_pair
            print(f"Visualizing matches for pair ({i}, {j}) with confidence {best_score:.3f} and {len(info.matches)} matches.")
            plot_matches_for_pair(images_u8[i], images_u8[j], info, features[i], features[j])

    print("Estimating initial cameras (rotations)...")
    cameras = estimate_initial_cameras(features, pairwise_matches, intrinsics_list)

    print("Running bundle adjustment (extrinsics only)...")
    cameras = run_bundle_adjustment(
        features, pairwise_matches, cameras,
        refine_intrinsics=refine_intrinsics,
        conf_thresh=conf_thresh
    )

    # Write adjusted tsai per image
    print("Writing adjusted .tsai files...")
    for idx, cam in enumerate(cameras):
        # Keep your original K to ensure intrinsics fixed
        K_fixed = intrinsics_list[idx]
        K, R_wc, C = tsai_from_detail_camera(cam, prefer_original_K=K_fixed)

        # File names
        base = os.path.splitext(os.path.basename(tsai_list[idx]))[0]
        out_tsai = os.path.join(out_cam_folder, f"{base}_ba.tsai")

        write_tsai(out_tsai, K, R_wc, C)
        # Small console report
        # print(f"[{idx:02d}] {os.path.basename(img_list[idx])}")
        # print(f"     Center C: {C}")
        # Convert R_wc to Euler just for a compact glance (ZYX)
        try:
            yaw, pitch, roll = rotation_to_euler_zyx(R_wc)
            # print(f"     YPR(deg): {(yaw, pitch, roll)}")
        except Exception:
            pass

    print(f"Done. Wrote {len(cameras)} adjusted .tsai files to: {out_cam_folder}")

def rotation_to_euler_zyx(R):
    """
    Convert rotation matrix (WORLD->CAMERA here) to yaw(Z)-pitch(Y)-roll(X) in degrees.
    Mainly for quick reporting.
    """
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        yaw = np.arctan2(R[1,0], R[0,0])
        pitch = np.arctan2(-R[2,0], sy)
        roll = np.arctan2(R[2,1], R[2,2])
    else:
        yaw = np.arctan2(-R[0,1], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        roll = 0.0
    return np.degrees([yaw, pitch, roll])

# ------------------------------
# Script entry
# ------------------------------
if __name__ == "__main__":
    # Example inputs (edit these to your paths)
    img_folder = '/Volumes/LaCie/raineyaberle/Research/PhD/SkySat-Stereo/study-sites/ID-MCS/20240420/SkySatScene_TOAR'
    cam_folder = os.path.join(img_folder, '..', 'proc_out', 'camgen_cam_gcp')

    img_list = sorted(glob(os.path.join(img_folder, "*analytic.tif")))
    cam_list = [glob(os.path.join(cam_folder, '*' + os.path.splitext(os.path.basename(img))[0] + '*.tsai'))[0] 
                for img in img_list]

    # Run
    main(
        img_list=img_list[0:4],
        tsai_list=cam_list[0:4],
        out_cam_folder=None,     # defaults to <cam_folder>/ba_out_tsai
        band_index=1,            # set None to average all bands; or choose specific band (0-based)
        match_conf=0.05,          # matcher confidence threshold
        refine_intrinsics=False, # keep intrinsics fixed (set True to also refine f, aspect, ppx, ppy)
        conf_thresh=0.05,         # BA confidence threshold for edges
        visualize_first_pair=False
    )
