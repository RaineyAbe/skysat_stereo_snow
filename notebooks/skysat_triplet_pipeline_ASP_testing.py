#! /usr/bin/env python
"""
Script for running the full pipeline based on workflow slightly modified from the ISPRS 2020 submission
Need to specify input image folder and reference DEM(s) folder
"""

import os
from glob import glob
import numpy as np
import sys

# add path to workflow functions
utils_dir = '/Users/raineyaberle/Research/PhD/SnowDEMs/skysat_stereo_snow/skysat_stereo_snow'
sys.path.append(utils_dir)
import skysat_stereo_workflow as workflow

# add path to ASP
asp_dir = '/Users/raineyaberle/Research/PhD/SnowDEMs/StereoPipeline-3.6.0-alpha-2025-08-05-x86_64-OSX/bin'
sys.path.append(asp_dir)

# Define input parameters
img_folder = '/Volumes/LaCie/raineyaberle/Research/PhD/SkySat-Stereo/study-sites/ID-MCS/20240420/SkySatScene_TOAR'
ortho_dem = '/Volumes/LaCie/raineyaberle/Research/PhD/SkySat-Stereo/study-sites/ID-MCS/refdem/MCS_refdem_lidar_COPDEM_merged.tif'
coreg_dem = ortho_dem
coreg_stable_only = False
job_name = 'MCS_20240420'
out_folder = '/Volumes/LaCie/raineyaberle/Research/PhD/SkySat-Stereo/study-sites/ID-MCS/20240420'
ndvi_threshold = 0.1
ndsi_threshold = 0.1
texture = 'normal'
full_workflow = 1
partial_workflow_steps = []

# Check for input files
img_list = glob(os.path.join(img_folder,'*.tif'))+glob(os.path.join(img_folder,'*.tiff'))
# make sure it doesn't include mask files
img_list = [x for x in img_list if 'udm' not in x]
print('Number of images located =', len(img_list))
if len(img_list)<2:
    print(f"Less than 2 images located, exiting")
    sys.exit()
if not os.path.exists(coreg_dem):
    print(f"coregdem {coreg_dem} could not be located, exiting")
    sys.exit()
if not os.path.exists(ortho_dem):
    print(f"orthodem {ortho_dem} could not be located, exiting")
    sys.exit()

if coreg_dem != ortho_dem:
    diff_dem = True
else:
    diff_dem = False

# structure for output folder
out_dir = os.path.join(out_folder, 'proc_out')

# Universal Args
final_stereo_session = 'pinholemappinhole'
final_ortho_session = 'pinhole'

# For consistency, lets hardcode expected file & folder names here :)
refdem_dir = os.path.join(out_dir, 'refdem')
cam_gcp_dir = os.path.join(out_dir, 'camgen_cam_gcp')
ortho_dir = os.path.join(out_dir, 'ortho')


if full_workflow == 1:
    steps2run = np.arange(0,11) # run the entire 10 steps
else:
    steps2run = np.array(partial_workflow_steps).astype(int)

# create output directory
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Calculate image bounds + 1 km
bound_buffer_fn, utm_epsg = workflow.calculate_images_footprint(img_list, out_dir)

# Clip DEM to image bounds + 1 km, make sure they're in the ideal UTM zone
ortho_dem = workflow.clip_raster(
    raster_fn = ortho_dem, 
    crop_shp_fn = bound_buffer_fn,
    t_crs = utm_epsg, 
    out_dir = refdem_dir
    )
if diff_dem:
    coreg_dem = workflow.clip_raster(
        raster_fn = coreg_dem, 
        crop_shp_fn = bound_buffer_fn, 
        t_crs = utm_epsg, 
        out_dir = refdem_dir
        )
else:
    coreg_dem = ortho_dem


workflow.generate_frame_cameras(img_list, 
                                ortho_dem, 
                                product_level = 'l1b',
                                out_folder = cam_gcp_dir)


#######################################################
### TESTING USING STEREO MATCHES + GCP FROM CAM_GEN ###
# step 2
init_ortho_dir = os.path.join(out_dir, 'baStereoGCP_init_ortho')
init_stereo_dir = os.path.join(out_dir, 'baStereoGCP_init_stereo')
init_stereo_pairs_fn = os.path.join(init_stereo_dir, 'overlapping_image_pairs.txt')
# step 3
ba_dir = os.path.join(out_dir, 'baStereoGCP_bundle_adjust')
ba_prefix = os.path.join(ba_dir, 'run')
# step 4
final_ortho_dir = os.path.join(out_dir, 'baStereoGCP_final_ortho')
final_stereo_dir = os.path.join(out_dir, 'baStereoGCP_final_stereo')
final_stereo_pairs_fn = os.path.join(final_stereo_dir, 'overlapping_image_pairs.txt')
# step 5
dem_mosaics_dir = os.path.join(out_dir, 'baStereoGCP_dem_mosaics')
# step 6 outputs
coreg_dir = os.path.join(out_dir, 'baStereoGCP_dem_coreg')
# final figures
fig_ddem_fn = os.path.join(out_dir, 'baStereoGCP_' + job_name + '_dDEM.png')
fig_final_fn = os.path.join(out_dir, 'baStereoGCP_' + job_name + '.png')

# intial ortho
workflow.run_mapproject(
        img_list = img_list,
        cam_folder = cam_gcp_dir, 
        ba_prefix = None,
        out_folder = init_ortho_dir, 
        dem = ortho_dem, 
        t_res = 1, 
        t_crs = utm_epsg, 
        orthomosaic = False
        )

# initial stereo
workflow.identify_overlapping_image_pairs(
    img_list = glob(os.path.join(init_ortho_dir, '*_map.tif')), 
    overlap_perc = 25, 
    utm_epsg = utm_epsg, 
    out_folder = init_stereo_dir
    )
workflow.run_stereo(
    stereo_pairs_fn = init_stereo_pairs_fn,
    cam_folder = ortho_dir,
    dem_fn = ortho_dem,
    out_folder = init_stereo_dir, 
    texture = texture,
    correlator_mode = True
    )

# copy clean matches from initial stereo to bundle adjust folder
import shutil
from tqdm import tqdm
os.makedirs(ba_dir, exist_ok=True)    
match_fns = glob(os.path.join(init_stereo_dir, '20*', '20*', 'run-*disp*.match'))
print(f'Copying {len(match_fns)} dense match files from stereo to bundle adjust directory...')
for fn in tqdm(match_fns):
    out_fn = os.path.join(ba_dir, os.path.basename(fn).split('run-disp-')[1])
    _ = shutil.copyfile(fn, out_fn)

# bundle adjustment
workflow.run_bundle_adjust(
    img_folder = img_folder, 
    cam_folder = cam_gcp_dir, 
    gcp_list = [os.path.join(cam_gcp_dir, 'clean_gcp.gcp')],
    overlap_fn = os.path.join(init_stereo_dir, 'overlapping_image_pairs_simple.txt'),
    ba_prefix = ba_prefix
    )

# final ortho
workflow.run_mapproject(
        img_list = img_list,
        cam_folder = ba_dir, 
        ba_prefix = None,
        out_folder = final_ortho_dir, 
        dem = ortho_dem, 
        t_res = 1, 
        t_crs = utm_epsg, 
        orthomosaic = True
        )

# final stereo
workflow.identify_overlapping_image_pairs(
    img_list = glob(os.path.join(final_ortho_dir, '*_map.tif')), 
    overlap_perc = 25, 
    utm_epsg = utm_epsg, 
    out_folder = final_stereo_dir
    )
workflow.run_stereo(
    stereo_pairs_fn = final_stereo_pairs_fn,
    cam_folder = final_ortho_dir,
    dem_fn = ortho_dem,
    out_folder = final_stereo_dir, 
    texture = texture,
    correlator_mode = False
    )

# rasterize point clouds
pc_list = glob(os.path.join(final_stereo_dir, '20*', '20*', '*-PC.tif'))
workflow.run_point2dem(
    pc_list = pc_list,
    t_res = 2,
    t_crs = utm_epsg
    )
    
# mosaic DEMs
dem_list = glob(os.path.join(final_stereo_dir, '20*', '20*', '*-DEM.tif'))
workflow.mosaic_dems(
    dem_list = dem_list, 
    out_folder = dem_mosaics_dir, 
    t_res = 2, 
    t_crs = utm_epsg, 
    )

# get un-coregistered file names
dem_fn = os.path.join(dem_mosaics_dir, 'dem_median_mosaic.tif')
ortho_fn = glob(os.path.join(ortho_dir, '*orthomosaic.tif'))[0]
nmad_fn = os.path.join(dem_mosaics_dir, 'dem_nmad_mosaic.tif')
count_fn = os.path.join(dem_mosaics_dir, 'dem_count_mosaic.tif')

# coregister DEMs
workflow.coregister_dems_xdem(
    dem_fn = dem_fn, 
    refdem_fn = coreg_dem, 
    raster_list = [ortho_fn, nmad_fn, count_fn],
    utm_crs = utm_epsg,
    out_dir = coreg_dir
    )

# get coregistered file names
dem_coreg_fn = os.path.join(coreg_dir, 'dem_median_mosaic_coregistered.tif')
ortho_coreg_fn = glob(os.path.join(coreg_dir, '*orthomosaic_coregistered.tif'))[0]
nmad_coreg_fn = os.path.join(coreg_dir, 'dem_nmad_mosaic_coregistered.tif')
count_coreg_fn = os.path.join(coreg_dir, 'dem_count_mosaic_coregistered.tif')

# plot & save dDEM
print('\nPlotting dDEM figure...')
import xdem
import matplotlib.pyplot as plt
dem = xdem.DEM(dem_coreg_fn)
refdem = xdem.DEM(coreg_dem).reproject(dem)
ddem = dem - refdem
fig, ax = plt.subplots()
ddem.plot(cmap='coolwarm_r', vmin=-10, vmax=10, ax=ax)
fig.savefig(fig_ddem_fn, dpi=250, bbox_inches='tight')
print('dDEM figure saved to file:', fig_ddem_fn)


print("\nPlotting final figure...")
workflow.plot_composite_fig(
    ortho_coreg_fn, 
    dem_coreg_fn, 
    count_coreg_fn, 
    nmad_coreg_fn, 
    outfn=fig_final_fn
    )

#######################################################
### TESTING USING ROADS AS GCP DURING BUNDLE ADJUST ###
# step 2 outputs
ba_dir = os.path.join(out_dir, 'baRoads_bundle_adjust')
ba_prefix = os.path.join(ba_dir, 'run')
# step 3 outputs
stereo_dir = os.path.join(out_dir, 'baRoads_stereo')
stereo_pairs_fn = os.path.join(stereo_dir, 'overlapping_image_pairs.txt')
# step 4 outputs
dem_mosaics_dir = os.path.join(out_dir, 'baRoads_dem_mosaics')
# step 5 outputs
coreg_dir = os.path.join(out_dir, 'baRoads_dem_coreg')
# final figures
fig_ddem_fn = os.path.join(out_dir, 'baRoads_' + job_name + '_dDEM.png')
fig_final_fn = os.path.join(out_dir, 'baRoads_' + job_name + '.png')

workflow.run_bundle_adjust(
    img_folder = img_folder, 
    cam_folder = cam_gcp_dir, 
    dem_fn = '/Volumes/LaCie/raineyaberle/Research/PhD/SkySat-Stereo/study-sites/ID-MCS/refdem/MCS_REFDEM_WGS84_ROADS.tif',
    dem_uncertainty = 3,
    ba_prefix = ba_prefix,
    cam_weight = 0, 
    num_iter = 1000, 
    num_pass = 2
    )

workflow.run_mapproject(
        img_list = img_list,
        cam_folder = cam_gcp_dir, 
        ba_prefix = ba_prefix + '_round2',
        out_folder = ortho_dir, 
        dem = ortho_dem, 
        t_res = 1, 
        t_crs = utm_epsg, 
        orthomosaic = True
        )

workflow.identify_overlapping_image_pairs(
    img_list = glob(os.path.join(ortho_dir, '*_map.tif')), 
    overlap_perc = 25, 
    # bh_ratio_range = [0.2,3],
    utm_epsg = utm_epsg, 
    out_folder = stereo_dir
    )

workflow.run_stereo(
    stereo_pairs_fn = stereo_pairs_fn,
    cam_folder = ortho_dir,
    dem_fn = ortho_dem,
    out_folder = stereo_dir, 
    texture = texture
    )

#####################################
### TESTING WITH NO BUNDLE ADJUST ###

# step 2 outputs
ba_dir = os.path.join(out_dir, 'noBA_bundle_adjust')
ba_prefix = os.path.join(ba_dir, 'run')
# step 3 outputs
stereo_dir = os.path.join(out_dir, 'noBA_stereo')
stereo_pairs_fn = os.path.join(stereo_dir, 'overlapping_image_pairs.txt')
# step 4 outputs
dem_mosaics_dir = os.path.join(out_dir, 'noBA_dem_mosaics')
# step 5 outputs
coreg_dir = os.path.join(out_dir, 'noBA_dem_coreg')
# final figures
fig_ddem_fn = os.path.join(out_dir, 'noBA_' + job_name + '_dDEM.png')
fig_final_fn = os.path.join(out_dir, 'noBA_' + job_name + '.png')

workflow.run_mapproject(
            img_list = img_list,
            cam_folder = cam_gcp_dir, 
            ba_prefix = None,
            out_folder = ortho_dir, 
            dem = ortho_dem, 
            t_res = 1, 
            t_crs = utm_epsg, 
            orthomosaic = True
            )

workflow.identify_overlapping_image_pairs(
        img_list = glob(os.path.join(ortho_dir, '*_map.tif')), 
        overlap_perc = 25, 
        # bh_ratio_range = [0.1,3],
        utm_epsg = utm_epsg, 
        out_folder = stereo_dir
        )

stereo_pairs_fn = os.path.join(stereo_dir, 'overlapping_image_pairs.txt')
workflow.run_stereo(
    stereo_pairs_fn = stereo_pairs_fn,
    cam_folder = ortho_dir,
    dem_fn = ortho_dem,
    out_folder = stereo_dir, 
    texture = texture
    )


#####################################
##### TESTING PAIR-WISE BUNDLE ADJUSTMENT #####

ba_dir = os.path.join(out_dir, 'baPairwise_bundle_adjust')
ba_prefix = os.path.join(ba_dir, 'run')
ortho_dir = os.path.join(out_dir, 'baPairwise_ortho')
stereo_dir = os.path.join(out_dir, 'baPairwise_stereo')
dem_mosaics_dir = os.path.join(out_dir, 'baPairwise_dem_mosaics')
coreg_dir = os.path.join(out_dir, 'baPairwise_dem_coreg')
fig_ddem_fn = os.path.join(out_dir, 'baPairwise_' + job_name + '_dDEM.png')
fig_final_fn = os.path.join(out_dir, 'baPairwise_' + job_name + '.png')

# determine overlapping pairs
workflow.identify_overlapping_image_pairs(
        img_list = glob(os.path.join(img_folder, '*analytic.tif')), 
        overlap_perc = 10, 
        # bh_ratio_range = [0.1,3],
        utm_epsg = utm_epsg, 
        out_folder = ba_dir
        )
overlap_fn = os.path.join(ba_dir, 'overlapping_image_pairs.txt')

workflow.iterative_pairwise_bundle_adjust(
        overlap_fn = overlap_fn, 
        cam_folder = cam_gcp_dir, 
        gcp_folder = cam_gcp_dir,
        ba_prefix = ba_prefix
        )

# update img_list to those with successful bundle_adjust
cam_list = sorted(glob(os.path.join(ba_dir, '*.tsai')))
img_list = [img for img in img_list if 
            len(glob(os.path.join(
                ba_dir, 
                '*' + os.path.splitext(os.path.basename(img))[0] + '*.tsai'))) > 0]
workflow.run_mapproject(
            img_list = img_list,
            cam_folder = ba_dir, 
            ba_prefix = None,
            out_folder = ortho_dir, 
            dem = ortho_dem, 
            t_res = 1, 
            t_crs = utm_epsg, 
            orthomosaic = True
            )

workflow.identify_overlapping_image_pairs(
        img_list = glob(os.path.join(ortho_dir, '*_map.tif')), 
        overlap_perc = 25, 
        # bh_ratio_range = [0.1,3],
        utm_epsg = utm_epsg, 
        out_folder = stereo_dir
        )
stereo_pairs_fn = os.path.join(stereo_dir, 'overlapping_image_pairs.txt')

workflow.run_stereo(
    stereo_pairs_fn = stereo_pairs_fn,
    cam_folder = ortho_dir,
    dem_fn = ortho_dem,
    out_folder = stereo_dir, 
    texture = texture
    )

#####################################

pc_list = glob(os.path.join(stereo_dir, '20*', '20*', '*-PC.tif'))
workflow.run_point2dem(
    pc_list = pc_list,
    t_res = 2,
    t_crs = utm_epsg
    )
    
dem_list = glob(os.path.join(stereo_dir, '20*', '20*', '*-DEM.tif'))
workflow.mosaic_dems(
    dem_list = dem_list, 
    out_folder = dem_mosaics_dir, 
    t_res = 2, 
    t_crs = utm_epsg, 
    )

# get un-coregistered file names
dem_fn = os.path.join(dem_mosaics_dir, 'dem_median_mosaic.tif')
ortho_fn = glob(os.path.join(ortho_dir, '*orthomosaic.tif'))[0]
nmad_fn = os.path.join(dem_mosaics_dir, 'dem_nmad_mosaic.tif')
count_fn = os.path.join(dem_mosaics_dir, 'dem_count_mosaic.tif')

workflow.coregister_dems_xdem(
    dem_fn = dem_fn, 
    refdem_fn = coreg_dem, 
    raster_list = [ortho_fn, nmad_fn, count_fn],
    utm_crs = utm_epsg,
    out_dir = coreg_dir
    )

# get coregistered file names
dem_coreg_fn = os.path.join(coreg_dir, 'dem_median_mosaic_coregistered.tif')
ortho_coreg_fn = glob(os.path.join(coreg_dir, '*orthomosaic_coregistered.tif'))[0]
nmad_coreg_fn = os.path.join(coreg_dir, 'dem_nmad_mosaic_coregistered.tif')
count_coreg_fn = os.path.join(coreg_dir, 'dem_count_mosaic_coregistered.tif')

# plot & save dDEM
print('\nPlotting dDEM figure...')
import xdem
import matplotlib.pyplot as plt
dem = xdem.DEM(dem_coreg_fn)
refdem = xdem.DEM(coreg_dem).reproject(dem)
ddem = dem - refdem
fig, ax = plt.subplots()
ddem.plot(cmap='coolwarm_r', vmin=-10, vmax=10, ax=ax)
fig.savefig(fig_ddem_fn, dpi=250, bbox_inches='tight')
print('dDEM figure saved to file:', fig_ddem_fn)


print("\nPlotting final figure...")
workflow.plot_composite_fig(
    ortho_coreg_fn, 
    dem_coreg_fn, 
    count_coreg_fn, 
    nmad_coreg_fn, 
    outfn=fig_final_fn
    )
