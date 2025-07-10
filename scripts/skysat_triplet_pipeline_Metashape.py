#! /usr/bin/env python

import argparse
import os
from glob import glob
from skysat_snow import skysat_stereo_workflow_NEW as workflow
from skysat_snow import metashape_utils as ms_utils
from skysat_snow import query_refdem_utils as refdem_utils

# ------------------------------
# ------ INPUT PARAMETERS ------
# ------------------------------

# optional variables for easier path referencing
# base_path = "/Volumes/LaCie/raineyaberle/Research/PhD/SkySat-Stereo/study-sites/Greenland-Saqqerleq"
base_path = "/bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/Saqqerleq"
site_name = "Saqqerleq"
date = '20230822'

job_name = f"{site_name}_{date}"
img_folder = os.path.join(base_path, 'SkySatScene_TOAR')
outfolder = os.path.join(base_path)
steps2run = [4, 5]
metashape = True
masks = ['water_check']
refdem = 'ArcticDEM'


# ------------------------------
# ------ CHECK INPUT FILES -----
# ------------------------------

img_list = sorted(glob(os.path.join(img_folder,f'{date}*_basic_analytic.tif')))
if len(img_list)<2:
    raise Exception(f"Only {len(img_list)} images detected, exiting")
else:
    print('Number of images found:', len(img_list))

udm2_map = {img: img.replace('_basic_analytic.tif', '_udm2.tif') for img in img_list}
if "cloud" in [m.lower() for m in masks]:
    existing_udm2 = {img: udm for img, udm in udm2_map.items() if os.path.exists(udm)}
    print(f'Number of UDM2 files found:', len(existing_udm2))
    if len(existing_udm2) < len(img_list):
        print('Not all images will be masked for clouds.')

# if selected coregister, refdem must be specified
if 4 in steps2run:
    # make sure refdem is specified
    if not refdem:
        raise Exception('To coregister output DEM, reference DEM must be specified. Exiting.')
    # make sure refdem is acceptable
    if (refdem!='ArcticDEM') & (refdem!='REMA') & (refdem!='COPDEM') & (not os.path.exists(refdem)):
            raise Exception(f"Reference DEM {refdem} could not be located. Exiting.")
    

# structure for output folder
out_fol = os.path.join(outfolder, 'proc_out')

# ------------------------------
# ---- DEFINE OUTPUT FOLDERS ---
# ------------------------------
# step 1: apply any masking to photos
masked_dir = os.path.join(out_fol, 'masked_photos')
# step 2: align cameras
project_fn = os.path.join(out_fol, job_name + '.psx') # Metashape project
alignment_dir = os.path.join(out_fol, 'aligned_photos')
# step 4: generate point cloud, DEM, and orthomosaic
sfm_dir = os.path.join(out_fol, 'sfm')
# step 5: coregistration with reference DEM
coreg_dir = os.path.join(out_fol, 'coregistration')
# step 6: plot figure of results
final_figure_fn = os.path.join(out_fol, f"{job_name}_result.jpg")

# Create output directory
os.makedirs(out_fol, exist_ok=True)


# ------------------------------
# ------- RUN THE STEPS --------
# ------------------------------

if 1 in steps2run:
    print('\n----------------------------------------')
    print('APPLY MASKS TO IMAGES:', masks)
    print('----------------------------------------\n')
    img_list = workflow.apply_masks_to_images(img_list, 
                                              mask_classes = masks, 
                                              out_dir = masked_dir)
elif masks:
    img_list = sorted(glob(os.path.join(masked_dir, '*.tif')))


if 2 in steps2run:
    print('\n----------------------------------------')
    print('ALIGN PHOTOS')
    print('----------------------------------------\n')
    aligned_cameras_fn = ms_utils.align_photos(img_list, 
                                               cam_folder = img_folder,
                                               project_fn = project_fn,
                                               out_folder = alignment_dir)


if 3 in steps2run:
    print('\n----------------------------------------')
    print('BUILD POINT CLOUD, DEM, AND ORTHOMOSAIC')
    print('----------------------------------------\n')
    pc_fn, dem_fn, ortho_fn = ms_utils.build_dem(project_fn, 
                                                 dem_resolution = 2, 
                                                 out_dir = sfm_dir)
else:
    dem_fn = glob(os.path.join(sfm_dir, '*_DEM.tif'))[0]
    ortho_fn = glob(os.path.join(sfm_dir, '*_orthomosaic.tif'))[0]


if 4 in steps2run:
    print('\n----------------------------------------')
    print('COREGISTER TO REFERENCE DEM')
    print('----------------------------------------\n')
    # Query GEE for reference DEM if needed
    if (refdem=='ArcticDEM') | (refdem=='REMA') | (refdem=='COPDEM'):        
        refdem_fn = refdem_utils.query_gee_for_refdem(dem_fn, 
                                                      refdem, 
                                                      out_dir=coreg_dir, 
                                                      crs="EPSG:4326", 
                                                      scale=30)
    else:
        refdem_fn = refdem
    # Coregister
    dem_fn, ortho_fn = workflow.coregister_dems_xdem(dem_fn = dem_fn, 
                                                     refdem_fn = refdem_fn, 
                                                     ortho_fn = ortho_fn,
                                                     out_dir = coreg_dir)


if 5 in steps2run:
    print('\n----------------------------------------')
    print('PLOT RESULTS FIGURE')
    print('----------------------------------------\n')
    ms_utils.plot_results_fig(dem_fn, 
                              ortho_fn, 
                              final_figure_fn)


print('\nDONE!')