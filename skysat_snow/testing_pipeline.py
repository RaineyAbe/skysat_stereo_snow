#! /usr/bin/env python
"""
Script for running the full pipeline based on workflow described in ISPRS 2020 submission
Need to specify input image folder, input refrence DEM folder
"""

import os,glob
import sys

code_path = "/Users/raineyaberle/Research/PhD/SnowDEMs/skysat_stereo_snow/"
sys.path.append(code_path)
from skysat_snow import skysat_stereo_workflow_NEW as workflow
from skysat_snow import metashape_utils as ms

# ------------------------------
# ------ INPUT PARAMETERS ------
# ------------------------------

base_path = "/Volumes/LaCie/raineyaberle/Research/PhD/SkySat-Stereo/study-sites/ID-MCS"
site_name = "MCS"
date = '20240420'
img_folder = os.path.join(base_path, date, 'SkySatScene_TOAR')
coreg_dem = os.path.join(base_path, 'refdem', f'{site_name}_refdem_lidar_COPDEM_merged.tif')
ortho_dem = coreg_dem
multispec = os.path.join(base_path, date, f"{site_name}_{date}_4band_mosaic.tif")
job_name = f"{site_name}_{date}"
outfolder = os.path.join(base_path, date)
steps2run = [1, 3]
metashape = True

ndvi_threshold = 0.6
ndsi_threshold = 0.4
coreg_stable_only = 1

# ------------------------------
# ------ CHECK FOR INPUTS ------
# ------------------------------

img_list = glob.glob(os.path.join(img_folder,'*_analytic.tif'))
if len(img_list)<2:
    raise Exception(f"Only {len(img_list)} images detected, exiting")

if not os.path.exists(coreg_dem):
    raise Exception(f"Coreg dem {coreg_dem} could not be located, exiting")

if not os.path.exists(ortho_dem):
    raise Exception(f"Ortho dem {ortho_dem} could not be located, exiting")

# structure for output folder
out_fol = os.path.join(outfolder, 'proc_out')


# ------------------------------
# ---- DEFINE OUTPUT FOLDERS ---
# ------------------------------

# step 1: clip reference DEM(s)
refdem_dir = os.path.join(out_fol, 'refdem')
# step 2: refine cameras
cam_gcp_directory = os.path.join(out_fol, 'camgen_cam_gcp')
# step 3: initial ortho and (if no metashape) stereo for dense matches
init_ortho_dir = os.path.join(out_fol,'init_ortho')
init_stereo_dir = os.path.join(out_fol, 'init_stereo')
# step 4: align cameras
init_alignment_dir = os.path.join(out_fol, 'align_photos')
ba_prefix = os.path.join(init_alignment_dir,'run')
# step 5: intermediate ortho and final stereo
intermediate_ortho_dir = os.path.join(out_fol, 'intermediate_ortho')
final_stereo_dir = os.path.join(out_fol, 'final_stereo')
# step 6: dem mosaicing
mos_dem_dir = os.path.join(out_fol, 'composite_dems')
# step 7: dem alignment
final_alignment_dir = os.path.join(out_fol,'georegistered_dem_mos')
# step 8: camera alignment
final_aligned_cam_dir = os.path.join(out_fol,'georegistered_cameras')
# step 9: final orthorectification
final_ortho_dir = os.path.join(out_fol,'georegistered_orthomosaics')
# step 10: plot figures
final_figure = os.path.join(out_fol, f"{job_name}_result.jpg")

# Create output directory
if not os.path.exists(out_fol):
    os.makedirs(out_fol)


# ------------------------------
# ------- RUN THE STEPS --------
# ------------------------------

# NOTE: Step 1 must be run each time - files and variables are needed for later steps. 
# Most steps are skipped if the files already exist in directory. 
print('\nSTEP 1: Clip reference DEM(s), apply optional masks')
print('----------------------------------------\n')
# Calculate image bounds for trimming DEM(s)
bound_buffer_fn, utm_epsg = workflow.calculate_image_bounds(img_list, refdem_dir)
# Trim reference DEM(s) to image bounds to speed up computations
coreg_dem_fn = workflow.clip_raster(coreg_dem, 
                                    bound_buffer_fn, 
                                    t_crs = utm_epsg, 
                                    out_dir = refdem_dir)
if coreg_dem != ortho_dem:
    ortho_dem_fn = workflow.clip_raster(ortho_dem, 
                                        bound_buffer_fn, 
                                        t_crs = utm_epsg, 
                                        out_dir = refdem_dir)
else:
    ortho_dem_fn = coreg_dem_fn

# Mask images if selected


# Not needed with Metashape alignment? 
# Inputting cams and GCPs to alignPhotos currently leads to null results...
# if 2 in steps2run:
#     print('\nSTEP 2: Refine frame cameras')
#     print('----------------------------------------\n')
#     workflow.generate_frame_cameras(img_list, 
#                                     dem_fn = ortho_dem_fn, 
#                                     product_level = 'l1b', 
#                                     out_folder = cam_gcp_directory)   
    

if 2 in steps2run:
    print('\n--------------------')
    print('STEP 2: Camera alignment')
    print('--------------------\n')
    if metashape:
          # align photos
          aligned_cameras_fn = ms.align_photos(img_list, 
                                               cam_folder = cam_gcp_directory,
                                               crs = utm_epsg, 
                                               out_folder = init_alignment_dir)
          # restructure aligned cameras into separate RPC files for ASP use
          ms.xml_cameras_to_rpc_txt(xml_fn = aligned_cameras_fn, 
                                    out_folder = init_alignment_dir)

    # else:
        # orthorectify
        # workflow.orthorectify(img_list, 
        #                     cam_folder = cam_gcp_directory, 
        #                     out_folder = init_ortho_dir, 
        #                     dem = ortho_dem_fn, 
        #                     t_res = 0.7, 
        #                     t_crs = utm_epsg, 
        #                     orthomosaic = 0)
        # identify overlapping pairs
        # run initial stereo for dense match creation
        # run stereo
        # run bundle adjustment

if 5 in steps2run:
    print('\n--------------------')
    print('STEP 5: Orthorectification and stereo')
    print('--------------------\n')
    print("Running intermediate orthorectification with bundle adjusted pinhole cameras")
    workflow.orthorectify(img_list, 
                          cam_folder = init_alignment_dir, 
                          out_folder = intermediate_ortho_dir, 
                          dem = ortho_dem_fn, 
                          t_res = 0.7, 
                          t_crs = utm_epsg, 
                          orthomosaic = 0)
    # identify overlapping image pairs for stereo
    final_overlap_stereo_pkl, final_overlap_stereo_txt = workflow.identify_overlapping_image_pairs(intermediate_ortho_dir, utm_epsg, intermediate_ortho_dir, overlap_perc, cross_track=False)
    # print("Running final stereo reconstruction")
    # workflow.execute_skysat_stereo(overlap_pkl=final_overlap_stereo_pkl, cam_folder=init_ba, out_folder=final_stereo_dir, session=final_stereo_session, 
    #                                dem=ortho_dem, texture='normal',  cross_track=False, correlator_mode=False)
    # pc_list = sorted(glob.glob(os.path.join(final_stereo_dir, '20*', '20*', '*-PC.tif')))
    # print(f"Gridding {len(pc_list)} point clouds")
    # workflow.gridding_wrapper(pc_list, tsrs=utm_epsg, tr=2)


# if 6 in steps2run:
#     print('\n--------------------')
#     print('STEP 6: Mosaic DEMs')
#     print('--------------------\n')
#     dem_list = sorted(glob.glob(os.path.join(final_stereo_dir, '20*', '20*', '*-DEM.tif')))
#     print(f"Mosaicing {len(dem_list)} DEMs")
#     workflow.dem_mosaic_wrapper(dem_list, mos_dem_dir, tr=2, tsrs=utm_epsg)


# if 7 in steps2run:
#     print('\n--------------------')
#     print('STEP 7: Align DEM median mosaic to reference DEM')
#     print('--------------------\n')
#     median_mos_dem = os.path.join(mos_dem_dir, 'dem_median_mos.tif')
#     workflow.align_dem(coreg_dem, median_mos_dem, final_alignment_dir, max_displacement=40, tr=2)


# if 8 in steps2run:
#     print('\n--------------------')
#     print('STEP 8: Align frame camera models')
#     print('--------------------\n')
#     # get the cameras from bundle adjust
#     camera_list = sorted(glob.glob(os.path.join(init_ba, 'run-*.tsai')))
#     print(f"Detected {len(camera_list)} cameras to be registered to DEM")
#     # make output directory for cameras 
#     if not os.path.exists(final_aligned_cam_dir):
#         os.makedirs(final_aligned_cam_dir)
#     # align using the ICP transform vector
#     alignment_vector_icp = glob.glob(os.path.join(final_alignment_dir, 'run-icp-transform.txt'))[0]
#     print("Aligning cameras")
#     print('First, using the ICP transform...')
#     workflow.align_cameras_wrapper(input_camera_list=camera_list, transform_txt=alignment_vector_icp, outfolder=final_aligned_cam_dir)
#     # align using the Nuth and Kaab transform vector
#     camera_list = sorted(glob.glob(os.path.join(final_aligned_cam_dir, 'run-*.tsai')))
#     alignment_vector_nk = glob.glob(os.path.join(final_alignment_dir, 'run-nk-transform.txt'))[0]
#     print('Then, using the Nuth and Kaab transform...')
#     workflow.align_cameras_wrapper(input_camera_list=camera_list, transform_txt=alignment_vector_nk, outfolder=final_aligned_cam_dir)


# if 9 in steps2run:
#     print('\n--------------------')
#     print('STEP 9: Construct final georegistered orthomosaics')
#     print('--------------------\n')
#     georegistered_median_mos_dem = glob.glob(os.path.join(final_alignment_dir, '*nk*DEM.tif'))[0]
#     print("Running final orthomsaic creation")
#     workflow.execute_skysat_orthorectification(img_list, out_folder=final_ortho_dir, dem=georegistered_median_mos_dem, session=final_ortho_session, 
#                                                tr='2', tsrs=utm_epsg, ba_prefix=os.path.join(final_aligned_cam_dir, 'run'), overlap_txt=init_overlap_stereo_txt, 
#                                                copy_rpc=0, orthomosaic=1)


# if 10 in steps2run:
#     print('\n--------------------')
#     print('STEP 10: Produce final plot of orthoimage, DEM, NMAD, and countmaps')
#     print('--------------------\n')
#     ortho = glob.glob(os.path.join(final_ortho_dir, '*finest_orthomosaic.tif'))[0]
#     count = glob.glob(os.path.join(mos_dem_dir, '*count*.tif'))[0]
#     nmad = glob.glob(os.path.join(mos_dem_dir, '*nmad*.tif'))[0]
#     georegistered_median_mos_dem = glob.glob(os.path.join(final_alignment_dir, '*nk*DEM.tif'))[0]
#     print("plotting final figure")
#     workflow.plot_composite_fig(ortho, georegistered_median_mos_dem, count, nmad, outfn=final_figure)


print('DONE!')