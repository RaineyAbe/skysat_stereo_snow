#! /usr/bin/env python
"""
Script for running the full pipeline based on workflow described in ISPRS 2020 submission
Need to specify input image folder, input refrence DEM folder
"""

from datetime import datetime
import os,glob
from skysat_stereo import skysat_stereo_workflow as workflow

# -----Input parameters
site_name = "MCS"
date = "20240420"
base_path = "/bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites"
img_folder = os.path.join(base_path, site_name, date, 'SkySatScene')
coreg_dem = os.path.join(base_path, site_name, 'refdem', f'{site_name}_refdem_lidar_COPDEM_merged.tif')
ortho_dem = coreg_dem
multispec = os.path.join(base_path, site_name, date, f"{site_name}_{date}_4band_mosaic.tif")
job_name = f"{site_name}_{date}"
outfolder = os.path.join(base_path, site_name, date)
ortho_workflow = 1
steps2run = [4, 5, 6, 7, 8, 9, 10]
ndvi_threshold=0.6
ndsi_threshold=0.4
ba_dem = None
ba_dem_uncertainty = 5
ba_cam_weight = 0
coreg_stable_only = 1
block_matching = 0
aoi_bbox = None
overlap_perc = 0.1

# Check for input files
img_list = glob.glob(os.path.join(img_folder,'*.tif'))+glob.glob(os.path.join(img_folder,'*.tiff'))
if len(img_list)<2:
    raise Exception(f"Only {len(img_list)} images detected, exiting")


if not os.path.exists(coreg_dem):
    raise Exception(f"Coreg dem {coreg_dem} could not be located, exiting")


if not os.path.exists(ortho_dem):
    raise Exception(f"Ortho dem {ortho_dem} could not be located, exiting")


# structure for output folder
out_fol = os.path.join(outfolder,'proc_out')

#Universal Args
if ortho_workflow == 1:
    map = True
else:
    map = False


if ortho_workflow == 1:
    map = True
    init_stereo_session = 'rpcmaprpc'
    init_ortho_session = 'rpc'
    final_stereo_session = 'pinholemappinhole'
    final_ortho_session = 'pinhole'
else:
    map = False
    init_stereo_session = 'rpc'
    init_ortho_session = 'rpc'
    final_stereo_session, final_ortho_session = ['nadirpinhole','pinhole']


# For consistency, lets hardcode some folder names here :)
# step 1 outputs
land_cover_dir = os.path.join(out_fol, 'land_cover_masks')
refdem_dir = os.path.join(out_fol, 'refdem')
# step 2 outputs
cam_gcp_directory = os.path.join(out_fol,'camgen_cam_gcp')
# step 3 outputs
init_ortho_dir = os.path.join(out_fol,'init_rpc_ortho')
init_stereo_dir = os.path.join(out_fol,'init_rpc_stereo')
# step 4 bundle_adjust dense matches
init_ba = os.path.join(out_fol,'ba_pinhole')
ba_prefix = os.path.join(init_ba,'run')
# step 5 stereo_args
intermediate_ortho_dir = os.path.join(out_fol,'intermediate_pinhole_ortho')
final_stereo_dir = os.path.join(out_fol, 'final_pinhole_stereo')
# step 6, dem mosaicing
mos_dem_dir = os.path.join(out_fol, 'composite_dems')
# step 7. dem alignment
final_alignment_dir = os.path.join(out_fol,'georegistered_dem_mos')
# step 8, camera alignment
final_aligned_cam_dir = os.path.join(out_fol,'georegistered_cameras')
# step 9, final orthorectification
final_ortho_dir = os.path.join(out_fol,'georegistered_orthomosaics')
# step 10, plot figure
final_figure = os.path.join(out_fol,f"{job_name}_result.jpg")

# create output directory
if not os.path.exists(out_fol):
    os.makedirs(out_fol)


# Run the steps
# NOTE: Step 1 must be run each time - files and variables are needed for later steps. 
# However, several steps are skipped if the files already exist in directory. 
print('\n--------------------')
print('Step 1: Prepare land cover masks, reference surfaces, and overlapping image pairs')
print('--------------------\n')
# Construct land cover masks
trees_mask_fn, snow_mask_fn, ss_mask_fn = workflow.construct_land_cover_masks(multispec, land_cover_dir, ndvi_threshold, ndsi_threshold, plot_results=True)
# Calculate image bounds for trimming DEM(s)
bound_fn, bound_buffer_fn, utm_epsg = workflow.calculate_image_bounds(img_folder, out_fol)
# Trim reference DEM(s) to image bounds to speed up computation and (optional) create stable reference elevations
coreg_dem_fn, ortho_dem_fn = workflow.prepare_reference_elevations(coreg_dem, ortho_dem, bound_buffer_fn, ss_mask_fn, coreg_stable_only=coreg_stable_only, crs=utm_epsg, out_dir=refdem_dir)
# Identify overlapping image pairs for stereo
overlap_stereo_pkl, overlap_stereo_txt = workflow.identify_overlapping_image_pairs(img_folder, utm_epsg, out_fol, overlap_perc, cross_track=False)


if 2 in steps2run:
    print('\n--------------------')
    print('Step 2: Generate frame cameras')
    print('--------------------\n')
    cam_gen_log = workflow.skysat_preprocess(img_folder, product_level='l1b', overlap_pkl=overlap_stereo_pkl, 
                                             dem_fn=ortho_dem, out_folder=cam_gcp_directory)
    now = datetime.now()
    log_fn = os.path.join(cam_gcp_directory, 'camgen_{}.log'.format(now))
    print("saving subprocess camgen log at {}".format(log_fn))
    with open(log_fn, 'w') as f:
        for log in cam_gen_log:
            f.write(log)


if 3 in steps2run:
    print('\n--------------------')
    print('Step 3: Initial orthorectification and stereo in correlator mode')
    print('--------------------\n')    
    if map:
        # orthorectify all the images first
        print("Orthorectifying images using RPC camera")
        workflow.execute_skysat_orthorectification(img_list=img_list, out_folder=init_ortho_dir, dem=ortho_dem, tr='0.7',
                                                   tsrs=utm_epsg, session=init_ortho_session, overlap_txt=overlap_stereo_txt, 
                                                   copy_rpc=1, orthomosaic=0)
        init_stereo_input_img_folder = init_ortho_dir
    else:
        init_stereo_input_img_folder = img_folder
    # now run stereo in correlator mode    
    workflow.execute_skysat_stereo(init_stereo_input_img_folder, init_stereo_dir, session=init_stereo_session, dem=ortho_dem, 
                                   texture='normal', block=1, crop_map=0, overlap_pkl=overlap_stereo_pkl, cross_track=False, correlator_mode=True)

print('Copying dense matches to bundle adjust folder')
workflow.dense_match_wrapper(init_stereo_dir, init_ba, modify_overlap=False)


if 4 in steps2run:
    print('\n--------------------')
    print('Step 4: Bundle adjustment')
    print('--------------------\n')
    # determine whether to use DEM in bundle adjustment
    if ba_dem:
        dem_for_ba = coreg_dem
    else:
        dem_for_ba = None
    workflow.bundle_adjustment(img_folder, ba_prefix, cam_folder=cam_gcp_directory, overlap_pkl=overlap_stereo_pkl, session='nadirpinhole', 
                               ba_dem=dem_for_ba, ba_dem_uncertainty=ba_dem_uncertainty, cam_weight=ba_cam_weight, num_iter=100, num_pass=2)


if 5 in steps2run:
    print('\n--------------------')
    print('Step 5: Orthorectification and stereo')
    print('--------------------\n')
    if map:
        print("Running intermediate orthorectification with bundle adjusted pinhole cameras")
        workflow.execute_skysat_orthorectification(img_list, out_folder=final_ortho_dir, dem=ortho_dem, tr='0.7', tsrs=utm_epsg, 
                                                   ba_prefix=ba_prefix, overlap_txt=overlap_stereo_txt, copy_rpc=1)
        final_stereo_input_img_folder = intermediate_ortho_dir
    else:
        final_stereo_input_img_folder = img_folder
    print("Running final stereo reconstruction")
    workflow.execute_skysat_stereo(overlap_stereo_pkl, final_stereo_dir, final_ortho_dir, correlator_mode=False)    
    # DEM gridding
    pc_list = sorted(glob.glob(os.path.join(final_stereo_dir, '20*', '20*', '*-PC.tif')))
    print(f"Gridding {len(pc_list)} point clouds")
    workflow.gridding_wrapper(pc_list, tr=2)


if 6 in steps2run:
    print('\n--------------------')
    print('Step 6: Mosaic DEMs')
    print('--------------------\n')
    dem_list = sorted(glob.glob(os.path.join(final_stereo_dir, '*-DEM.tif')))
    print(f"Mosaicing {len(dem_list)} DEMs")
    workflow.dem_mosaic_wrapper(dem_list, mos_dem_dir, tr=2, tsrs=utm_epsg)


if 7 in steps2run:
    print('\n--------------------')
    print('Step 7: Align DEM median mosaic to reference DEM')
    print('--------------------\n')
    # now perform alignment
    median_mos_dem = os.path.join(mos_dem_dir, 'dem_median_mos.tif')
    workflow.align_dem(coreg_dem, median_mos_dem, final_alignment_dir, max_displacement=40, tr=2)


if 8 in steps2run:
    print('\n--------------------')
    print('Step 8: Align frame camera models')
    print('--------------------\n')
    # get the cameras from bundle adjust
    camera_list = sorted(glob.glob(os.path.join(init_ba, 'run-*.tsai')))
    print(f"Detected {len(camera_list)} cameras to be registered to DEM")
    # get the alignment vector from step 7
    alignment_vector = glob.glob(os.path.join(final_alignment_dir, 'alignment_vector.txt'))[0]
    # make output directory for cameras 
    if not os.path.exists(final_aligned_cam_dir):
        os.makedirs(final_aligned_cam_dir)
    print("Aligning cameras")
    workflow.align_cameras_wrapper(input_camera_list=camera_list, transform_txt=alignment_vector, outfolder=final_aligned_cam_dir)


if 9 in steps2run:
    print('\n--------------------')
    print('Step 9: Construct final georegistered orthomosaics')
    print('--------------------\n')
    georegistered_median_mos_dem = glob.glob(os.path.join(final_alignment_dir, '*nk*DEM.tif'))[0]
    print("Running final orthomsaic creation")
    workflow.execute_skysat_orthorectification(images=img_list, data='triplet', session=final_ortho_session,
                                                outdir=final_ortho_dir, tsrs=utm_epsg, dem=georegistered_median_mos_dem,
                                                ba_prefix=os.path.join(final_aligned_cam_dir, 'run'), mode='science',
                                                overlap_list=overlap_stereo_txt, copy_rpc=0, orthomosaic=1)


if 10 in steps2run:
    print('\n--------------------')
    print('Step 10: Produce final plot of orthoimage, DEM, NMAD, and countmaps')
    print('--------------------\n')
    ortho = glob.glob(os.path.join(final_ortho_dir, '*finest_orthomosaic.tif'))[0]
    count = glob.glob(os.path.join(mos_dem_dir, '*count*.tif'))[0]
    nmad = glob.glob(os.path.join(mos_dem_dir, '*nmad*.tif'))[0]
    georegistered_median_mos_dem = glob.glob(os.path.join(final_alignment_dir, '*nk*DEM.tif'))[0]
    print("plotting final figure")
    workflow.plot_composite_fig(ortho, georegistered_median_mos_dem, count, nmad, outfn=final_figure)


print('DONE!')