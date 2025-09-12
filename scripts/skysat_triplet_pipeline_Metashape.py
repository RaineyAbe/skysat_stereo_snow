#! /usr/bin/env python

"""
Script for running the DEM construction pipeline using Agisoft Metashape.
"""

import argparse
import os
from glob import glob
from skysat_stereo_snow import skysat_stereo_workflow as workflow
from skysat_stereo_snow import metashape_utils as ms_utils
from skysat_stereo_snow import query_refdem_utils as refdem_utils

def getparser():
    parser = argparse.ArgumentParser(description='Wrapper script to run full triplet stereo to DEM workflow.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-img_folder', default=None, type=str, help='Path to folder containing L1B TOAR imagery.')
    parser.add_argument('-masks', default=[], nargs='+', help='Masks to apply to images before constructing DEM. Options:'
                        '\n\t- "cloud", "cloud_shadow", "light_haze", "snow": UDM2-based mask options (files must be in img_folder)'
                        '\n\t- "water_check": NDWI thresholding is used to remove images with > 99 percent water from analysis.')
    parser.add_argument('-coregister', default=0, type=int, choices=[1,0], help='Whether to coregister the resulting DEM to a reference DEM. "refdem" must be specified.')
    parser.add_argument('-refdem', default=None, type=str, help='Reference DEM to use in coregistration. Several options available:'
                        '\n\t- Path to file in directory'
                        '\n\t- Name of DEM to query, clip, and download from Google Earth Engine: "ArcticDEM", "REMA", "COPDEM"')
    parser.add_argument('-job_name', default=None, type=str, help='Identifier for output folder and final composite products.')
    parser.add_argument('-out_folder',default=None,type=str,help='path to output folder to save results in')
    parser.add_argument('-full_workflow', choices=[1,0], type=int, default=1, help='Specify 1 to run full workflow')
    parser.add_argument('-partial_workflow_steps', nargs='+', default=None, help='Specify steps of workflow to run.' 
                        '\n\t1 = Mask images'
                        '\n\t2 = Align photos'
                        '\n\t3 = Build DEM and orthomosaic'
                        '\n\t4 = Coregister to reference DEM'
                        '\n\t5 = Plot results figure')
    return parser


def main():
        
    # ------------------------------
    # ---- GET INPUT PARAMETERS ----
    # ------------------------------

    parser = getparser()
    args = parser.parse_args()
    img_folder = args.img_folder
    masks = args.masks
    coregister = args.coregister
    refdem = args.refdem
    job_name = args.job_name
    out_folder = args.out_folder
    full_workflow = args.full_workflow
    steps2run = args.partial_workflow_steps
    if full_workflow:
        steps2run = [1, 2, 3, 4, 5]

    # ------------------------------
    # ------ CHECK INPUT FILES -----
    # ------------------------------

    img_list = sorted(glob(os.path.join(img_folder,f'*analytic.tif')))
    if len(img_list)<2:
        raise Exception(f"Only {len(img_list)} TOAR images detected, exiting")
    else:
        print('Number of images found:', len(img_list))

    udm2_map = {img: img.replace('_basic_analytic.tif', '_udm2.tif') for img in img_list}
    if "cloud" in [m.lower() for m in masks]:
        existing_udm2 = {img: udm for img, udm in udm2_map.items() if os.path.exists(udm)}
        print(f'Number of UDM2 files found:', len(existing_udm2))
        if len(existing_udm2) < len(img_list):
            print('Not all images will be masked for clouds.')

    # if selected coregister, refdem must be specified
    if coregister & (4 in steps2run):
        # make sure refdem is specified
        if not refdem:
            raise Exception('To coregister output DEM, reference DEM must be specified. Exiting.')
        # make sure refdem is acceptable
        if (refdem!='ArcticDEM') & (refdem!='REMA') & (refdem!='COPDEM') & (not os.path.exists(refdem)):
                raise Exception(f"Reference DEM {refdem} could not be located. Exiting.")

    # ------------------------------
    # ---- DEFINE OUTPUT FOLDERS ---
    # ------------------------------

    # root output folder
    out_fol = os.path.join(out_folder, job_name)
    # step 1: apply any masking to photos
    masked_dir = os.path.join(out_fol, 'masked_photos')
    # step 2: align cameras
    project_fn = os.path.join(out_fol, job_name + '.psx') # Metashape project
    alignment_dir = os.path.join(out_fol, 'aligned_photos')
    # step 3: generate point cloud, DEM, and orthomosaic
    sfm_dir = os.path.join(out_fol, 'sfm')
    # step 4: coregistration with reference DEM
    coreg_dir = os.path.join(out_fol, 'coregistration')
    # step 5: plot results figure
    final_figure_fn = os.path.join(out_fol, job_name + '.png')

    # Create output directory
    os.makedirs(out_fol, exist_ok=True)


    # ------------------------------
    # ------- RUN THE STEPS --------
    # ------------------------------

    if (1 in steps2run) & (len(masks) > 0):
        print('\n----------------------------------------')
        print('APPLY MASKS TO IMAGES:', masks)
        print('----------------------------------------\n')
        img_list = workflow.apply_masks_to_images(
            img_list, 
            mask_classes = masks, 
            out_dir = masked_dir,
            copy_cams=True
        )
    elif masks:
        img_list = sorted(glob(os.path.join(masked_dir, '*.tif')))


    if 2 in steps2run:
        print('\n----------------------------------------')
        print('ALIGN PHOTOS')
        print('----------------------------------------\n')
        aligned_cameras_fn = ms_utils.align_photos(
            img_list, 
            project_fn = project_fn,
            out_folder = alignment_dir
        )


    if 3 in steps2run:
        print('\n----------------------------------------')
        print('BUILD POINT CLOUD, DEM, AND ORTHOMOSAIC')
        print('----------------------------------------\n')
        dem_fn, ortho_fn = ms_utils.build_dem(
            project_fn, 
            dem_resolution = 2,
            out_dir = sfm_dir
        )
    else:
        dem_fn = glob(os.path.join(sfm_dir, '*_DEM.tif'))[0]
        ortho_fn = glob(os.path.join(sfm_dir, '*_orthomosaic.tif'))[0]


    if 4 in steps2run:
        print('\n----------------------------------------')
        print('COREGISTER TO REFERENCE DEM')
        print('----------------------------------------\n')
        # Query GEE for reference DEM if needed
        if (refdem=='ArcticDEM') | (refdem=='REMA') | (refdem=='COPDEM'):        
            refdem_fn = refdem_utils.query_gee_for_refdem(
                dem_fn, 
                refdem, 
                out_dir=coreg_dir, 
                crs="EPSG:4326", 
                scale=30
            )
        else:
            refdem_fn = refdem
        # Coregister
        workflow.coregister_dems_xdem(
            dem_fn = dem_fn, 
            refdem_fn = refdem_fn,
            ortho_fn = ortho_fn,
            out_dir = coreg_dir
        )
    if coregister:
        dem_fn = glob(os.path.join(coreg_dir, '*DEM*coregistered.tif'))[0]
        ortho_fn = glob(os.path.join(coreg_dir, '*ortho*coregistered.tif'))[0]


    if 5 in steps2run:
        print('\n----------------------------------------')
        print('PLOT RESULTS FIGURE')
        print('----------------------------------------\n')
        ms_utils.plot_results_fig(
            dem_fn, 
            ortho_fn, 
            final_figure_fn
        )


    print('\nDONE!')

if __name__ == '__main__':
    main()