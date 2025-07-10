#! /usr/bin/env python
import argparse
from datetime import datetime
import os,sys,glob
import numpy as np

# add path to workflow functions
import sys
utils_path = "/Users/raineyaberle/Research/PhD/SnowDEMs/skysat_stereo_snow/skysat_stereo_snow"
sys.path.append(utils_path)
import skysat_stereo_workflow as workflow

"""
Script for running the full pipeline based on workflow slightly modified from the ISPRS 2020 submission
Need to specify input image folder, multispectral image folder/filename, reference DEM(s) folder
"""

def getparser():
    parser = argparse.ArgumentParser(description='Wrapper script to run full triplet stereo workflow')
    parser.add_argument('-in_img',default=None,type=str,help='path to Folder containing L1B imagery')
    parser.add_argument('-multispec',default=None,type=str,help='path to 4-band image mosaic or folder containing images, used to construct stable surfaces mask.')
    parser.add_argument('-ndvi_threshold',default=0.5,type=float,help='threshold for the Normalized Difference Vegetation Index used to mask trees')
    parser.add_argument('-ndsi_threshold',default=0.4,type=float,help='threshold for the Normalized Difference Snow Index used to mask snow')
    parser.add_argument('-orthodem',default=None,type=str,help='path to reference DEM to use in orthorectification and camera resection, if not provided, will use coregdem')
    parser.add_argument('-coregdem',default=None,type=str,help='path to reference DEM to use in coregisteration')
    parser.add_argument('-coreg_stable_only',default=0,type=int,choices=[1,0],help='whether to coregister using stable reference elevations only')
    parser.add_argument('-ba_dem',default=False,type=int,choices=[1,0],help='whether to include coregdem in bundle adjust')
    parser.add_argument('-ba_dem_uncertainty',default=10,type=float,help='uncertainty of the coreg DEM, used for weighting in bundle adjustment')
    parser.add_argument('-ba_cam_weight',default=0,type=float,help='weight of the initial camera position in bundle adjustment')
    parser.add_argument('-aoi_bbox',default=None,type=str,help='path to bounding box shapefile if limiting processing to a smaller aoi')
    parser.add_argument('-ortho_workflow',default=1,type=int,choices=[1,0],help='option to orthorectify before stereo or not')
    parser.add_argument('-block_matching',default=0,type=int,choices=[1,0],help='whether to use block matching in final stereo matching, default is 0 (not)')
    parser.add_argument('-job_name',default=None,type=str,help='identifier for output folder and final composite products')
    parser.add_argument('-outfolder',default=None,type=str,help='path to output folder to save results in')
    parser.add_argument('-full_workflow',choices=[1,0],type=int,default=1,help='Specify 1 to run full workflow (default: %(default)s)')
    parser.add_argument('-partial_workflow_steps',nargs='+',help='specify steps of workflow to run')
    return parser


def main():
    parser = getparser()
    args = parser.parse_args()

    img_folder = args.in_img
    multispec = args.multispec
    ndvi_threshold = args.ndvi_threshold
    ndsi_threshold = args.ndsi_threshold
    coreg_dem = args.coregdem
    ba_dem = args.ba_dem
    ba_dem_uncertainty = args.ba_dem_uncertainty
    ba_cam_weight = args.ba_cam_weight
    coreg_stable_only = args.coreg_stable_only
    block_matching = args.block_matching
    if args.orthodem is not None:
        ortho_dem = args.orthodem
    else:
        ortho_dem = coreg_dem
    # Check for input files
    img_list = glob.glob(os.path.join(img_folder,'*.tif'))+glob.glob(os.path.join(img_folder,'*.tiff'))
    if len(img_list)<2:
        print(f"Only {len(img_list)} images detected, exiting")
        sys.exit()
    if not os.path.exists(coreg_dem):
        print(f"Coreg dem {coreg_dem} could not be located, exiting")
        sys.exit()
    if not os.path.exists(ortho_dem):
        print(f"Ortho dem {ortho_dem} could not be located, exiting")
        sys.exit()

    # structure for output folder
    out_fol = os.path.join(args.outfolder,'proc_out')
    job_name = args.job_name
    
    # Universal Args
    if args.ortho_workflow == 1:
        map = True
        final_stereo_session = 'pinholemappinhole'
        final_ortho_session = 'pinhole'
    else:
        map = False
        final_stereo_session, final_ortho_session = ['nadirpinhole','pinhole']

    # For consistency, lets hardcode expected file names,folder names here :)
    # step 1 outputs
    overlap_full_txt = os.path.join(out_fol,'overlap.txt')
    overlap_full_pkl = os.path.splitext(overlap_full_txt)[0]+'_with_overlap_perc.pkl'
    overlap_stereo_pkl = os.path.splitext(overlap_full_pkl)[0]+'_stereo_only.pkl'
    overlap_stereo_txt = os.path.splitext(overlap_full_pkl)[0]+'_stereo_only.txt'
    bound_fn = os.path.splitext(overlap_full_txt)[0]+'_bound.gpkg'
    bound_buffer_fn = os.path.splitext(bound_fn)[0]+'_1km_buffer.gpkg'
    land_cover_dir = os.path.join(out_fol, 'land_cover_masks')
    refdem_dir = os.path.join(out_fol, 'refdem')
    # step 2 outputs
    cam_gcp_directory = os.path.join(out_fol,'camgen_cam_gcp')
    # step 3 bundle_adjust dense matches
    init_ba = os.path.join(out_fol,'ba_pinhole')
    ba_prefix = os.path.join(init_ba,'run')
    # step 4 stereo_args
    intermediate_ortho_dir = os.path.join(out_fol,'intermediate_pinhole_ortho')
    final_stereo_dir = os.path.join(out_fol, 'final_pinhole_stereo')
    # step 5, individual dem aligning
    init_alignment_dir = os.path.join(out_fol,'aligned_dems')
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
    # step 11, experimental rpc production
    if args.full_workflow == 1:
        steps2run = np.arange(0,11) # run the entire 10 steps
    else:
        steps2run = np.array(args.partial_workflow_steps).astype(int)

    # create output directory
    if not os.path.exists(out_fol):
        os.makedirs(out_fol)


    # Run the steps
    if 1 in steps2run:
        print('\n--------------------')
        print('Step 1: Compute overlapping pairs')
        print('--------------------\n')
        overlap_perc = 0.01
        workflow.prepare_stereopair_list(img_folder, overlap_perc, overlap_full_txt, aoi_bbox=None)

    # Construct stable surface masks
    trees_mask_fn, snow_mask_fn, ss_mask_fn = workflow.construct_land_cover_masks(multispec, land_cover_dir, ndvi_threshold, ndsi_threshold, plot_results=True)

    # Trim reference DEM(s) to speed up computation and (optional) create stable reference elevations
    coreg_dem, ortho_dem, epsg_code = workflow.prepare_reference_elevations(coreg_dem, ortho_dem, bound_fn, bound_buffer_fn, ss_mask_fn, coreg_stable_only=coreg_stable_only, out_dir=refdem_dir)

    if 2 in steps2run:
        print('\n--------------------')
        print('Step 2: Generate frame cameras')
        print('--------------------\n')
        cam_gen_log = workflow.skysat_preprocess(img_folder, mode='triplet',
                                                 product_level='l1b', overlap_pkl=overlap_stereo_pkl, dem=ortho_dem,
                                                 outdir=cam_gcp_directory)
        now = datetime.now()
        log_fn = os.path.join(cam_gcp_directory, 'camgen_{}.log'.format(now))
        print("saving subprocess camgen log at {}".format(log_fn))
        with open(log_fn, 'w') as f:
            for log in cam_gen_log:
                f.write(log)


    if 3 in steps2run:
        print('\n--------------------')
        print('Step 3: Bundle adjustment')
        print('--------------------\n')
        if ba_dem:
            dem_for_ba = coreg_dem
        else:
            dem_for_ba = None
        workflow.bundle_adjustment(img_folder, ba_prefix, cam_gcp_directory, overlap_list=overlap_stereo_txt, 
                                   refdem=ortho_dem, ba_dem=dem_for_ba, ba_dem_uncertainty=ba_dem_uncertainty, 
                                   cam_weight=ba_cam_weight, num_iter=700, num_pass=2)
    
    
    if 4 in steps2run:
        print('\n--------------------')
        print('Step 4: Orthorectification and stereo')
        print('--------------------\n')
        if map:
            print("Running intermediate orthorectification with bundle adjusted pinhole cameras")
            workflow.execute_skysat_orthorectification(images=img_list, data='triplet', session=final_ortho_session,
                                                        outdir=intermediate_ortho_dir, tr='0.5', tsrs=epsg_code, dem=ortho_dem,
                                                        ba_prefix=ba_prefix + '-run-run', mode='science',
                                                        overlap_list=overlap_stereo_txt,
                                                        copy_rpc=1, orthomosaic=0)
            final_stereo_input_img_folder = intermediate_ortho_dir
        else:
            final_stereo_input_img_folder = img_folder
        # now run stereo
        print("Running final stereo reconstruction")
        workflow.execute_skysat_stereo(final_stereo_input_img_folder,
                                        final_stereo_dir, ba_prefix=ba_prefix + '-run-run',
                                        mode='triplet', session=final_stereo_session,
                                        dem=ortho_dem, texture='normal', writeout_only=False,
                                        block=block_matching, crop_map=1, threads=2, overlap_pkl=overlap_stereo_pkl,
                                        cross_track=False)    
        # DEM gridding
        pc_list = sorted(glob.glob(os.path.join(final_stereo_dir, '20*', '20*', '*-PC.tif')))
        print(f"Gridding {len(pc_list)} point clouds")
        workflow.gridding_wrapper(pc_list, tr=2)


    if 5 in steps2run:
        print('\n--------------------')
        print('Step 5: Align individual DEMs')
        print('--------------------\n')
        dem_list = sorted(glob.glob(os.path.join(final_stereo_dir, '20*', '20*', '*-DEM.tif')))
        workflow.align_individual_dems(dem_list, coreg_dem, init_alignment_dir, max_displacement=40, tr=0.5)
        
        
    if 6 in steps2run:
        print('\n--------------------')
        print('Step 6: Mosaic DEMs')
        print('--------------------\n')
        dem_list = sorted(glob.glob(os.path.join(init_alignment_dir, '*-DEM.tif')))
        print(f"Mosaicing {len(dem_list)} DEMs")
        workflow.dem_mosaic_wrapper(dem_list, mos_dem_dir, tr=2, tsrs=epsg_code)


    if 7 in steps2run:
        print('\n--------------------')
        print('Step 7: Align DEM median mosaic to reference DEM')
        print('--------------------\n')
        # now perform alignment
        median_mos_dem = os.path.join(mos_dem_dir, 'dem_median_mos.tif')
        workflow.align_dem(coreg_dem, median_mos_dem, final_alignment_dir, max_displacement=40, tr=2)


    if 8 in steps2run:
        print('\n--------------------')
        print('Step 8: Construct final georegistered orthomosaics')
        print('--------------------\n')
        georegistered_median_mos_dem = glob.glob(os.path.join(final_alignment_dir, '*nk*DEM.tif'))[0]
        print("Running final orthomsaic creation")
        workflow.execute_skysat_orthorectification(images=img_list, data='triplet', session=final_ortho_session,
                                                    outdir=final_ortho_dir, tsrs=epsg_code, dem=georegistered_median_mos_dem,
                                                    ba_prefix=ba_prefix + '-run-run', mode='science',
                                                    overlap_list=overlap_stereo_txt, copy_rpc=0, orthomosaic=1)


    if 9 in steps2run:
        print('\n--------------------')
        print('Step 9: Produce final plot of orthoimage, DEM, NMAD, and countmaps')
        print('--------------------\n')
        ortho = glob.glob(os.path.join(final_ortho_dir, '*finest_orthomosaic.tif'))[0]
        count = glob.glob(os.path.join(mos_dem_dir, '*count*.tif'))[0]
        nmad = glob.glob(os.path.join(mos_dem_dir, '*nmad*.tif'))[0]
        georegistered_median_mos_dem = glob.glob(os.path.join(final_alignment_dir, '*nk*DEM.tif'))[0]
        print("plotting final figure")
        workflow.plot_composite_fig(ortho, georegistered_median_mos_dem, count, nmad, outfn=final_figure)


if __name__ == '__main__':
    main()
