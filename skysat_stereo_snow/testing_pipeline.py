#! /usr/bin/env python
"""
Script for running the full pipeline based on workflow described in ISPRS 2020 submission
Need to specify input image folder, input refrence DEM folder
"""

from datetime import datetime
import os,sys,glob
from skysat_stereo import skysat_stereo_workflow as workflow

# -----Input parameters
site_name = "Banner"
date = "20240419-1"
base_path = "/bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites"
img_folder = os.path.join(base_path, site_name, date, 'SkySatScene')
coreg_dem = glob.glob(os.path.join(base_path, site_name, 'refdem', '*_merged.tif'))[0]
ortho_dem = coreg_dem
multispec_dir = os.path.join(base_path, site_name, date, f"{site_name}_{date}_4band_mosaic.tif")
job_name = f"{site_name}_{date}"
outfolder = os.path.join(base_path, site_name, date)
ortho_workflow = 1
steps2run = [1, 2, 4, 5, 6, 7, 8, 9, 10]
ndvi_threshold=0.6
ndsi_threshold=0.4
block_matching = 0
aoi_bbox = None
mask_dem = 0
mask_dem_opt = 'none'

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
out_fol = os.path.join(outfolder,'proc_out')

#Universal Args
if ortho_workflow == 1:
    map = True
else:
    map = False


if map:
    init_stereo_session = 'rpcmaprpc'
    init_ortho_session = 'rpc'
    final_stereo_session = 'pinholemappinhole'
    final_ortho_session = 'pinhole'
else:
    init_stereo_session = 'rpc'
    init_ortho_session = 'rpc'
    final_stereo_session, final_ortho_session = ['nadirpinhole','pinhole']


# For consistency, lets hardcode expected file names,folder names here :)
# step 1 outputs
overlap_full_txt = os.path.join(out_fol,'overlap.txt')
overlap_full_pkl = os.path.splitext(overlap_full_txt)[0]+'_with_overlap_perc.pkl'
overlap_stereo_pkl = os.path.splitext(overlap_full_pkl)[0]+'_stereo_only.pkl'
overlap_stereo_txt = os.path.splitext(overlap_full_pkl)[0]+'_stereo_only.txt'
bound_fn = os.path.splitext(overlap_full_txt)[0]+'_bound.gpkg'
bound_buffer_fn = os.path.splitext(bound_fn)[0]+'_1km_buffer.gpkg'
stable_surface_dir = os.path.join(out_fol,'stable_surfaces')
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
final_stereo_dir = os.path.join(out_fol,'final_pinhole_stereo')
# step 6, dem gridding and mosaicing
mos_dem_dir = os.path.join(final_stereo_dir,'composite_dems')
# step 7. dem_alignment
alignment_dir = os.path.join(out_fol,'georegistered_dem_mos')
# step 8, camera alignment
aligned_cam_dir = os.path.join(out_fol,'georegistered_cameras')
# step 9, final orthorectification
final_ortho_dir = os.path.join(out_fol,'georegistered_orthomosaics')
# step 10, plot figure
final_figure = os.path.join(out_fol,f"{job_name}_result.jpg")
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


# Trim ref DEMs to speed up computation, create stable surface masks and stable reference elevations for bundle adjustment
coreg_dem, ortho_dem, epsg_code = workflow.prepare_reference_elevations(coreg_dem, ortho_dem, multispec_dir, bound_fn, bound_buffer_fn, stable_surface_dir, 
                                                                        ndvi_threshold=0.5, ndsi_threshold=0.4, coreg_stable_only=False)


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
    # specify whether to run using maprojected sessions or not
    print('\n--------------------')
    print('Step 3: Orthorectify and run stereo using RPC cameras')
    print('--------------------\n')
    if map:
        # orthorectify all the images first
        print("Orthorectifying images using RPC camera")
        workflow.execute_skysat_orthorectification(images=img_list, data='triplet', session=init_ortho_session,
                                                outdir=init_ortho_dir, tsrs=epsg_code, dem=ortho_dem, mode='science',
                                                overlap_list=overlap_stereo_txt, copy_rpc=1, orthomosaic=0)
        init_stereo_input_img_folder = init_ortho_dir
    else:
        init_stereo_input_img_folder = img_folder
    print("Running stereo using RPC cameras")
    # Note crop_map = 0 option, this does not do warping to common extent and resolution for orthoimages before stereo, because we want to
    # presrve this crucail information for correctly unwarped dense match points
    workflow.execute_skysat_stereo(init_stereo_input_img_folder, init_stereo_dir,
                                   mode='triplet', session=init_stereo_session,
                                   dem=ortho_dem, texture='normal', writeout_only=False,
                                   block=1, crop_map=0, threads=2, overlap_pkl=overlap_stereo_pkl,
                                   cross_track=False)
    # copy dense match file to ba directory
    workflow.dense_match_wrapper(stereo_master_dir=os.path.abspath(init_stereo_dir),
                                 ba_dir=os.path.abspath(init_ba), modify_overlap=0)


if 4 in steps2run:
    print('\n--------------------')
    print('Step 4: Bundle adjustment')
    print('--------------------\n')
    workflow.bundle_adjustment(img_folder, ba_prefix, cam_gcp_directory, overlap_list=overlap_stereo_txt, 
                               dem=coreg_dem, num_iter=2000, num_pass=2)
    

if 5 in steps2run:
    print('\n--------------------')
    print('Step 5: Final stereo')
    print('--------------------\n')
    if map:
        print("Running intermediate orthorectification with bundle adjusted pinhole cameras")
        workflow.execute_skysat_orthorectification(images=img_list, data='triplet', session=final_ortho_session,
                                                    outdir=intermediate_ortho_dir, tsrs=epsg_code, dem='WGS84',
                                                    ba_prefix=ba_prefix + '-run', mode='science',
                                                    overlap_list=overlap_stereo_txt,
                                                    copy_rpc=1, orthomosaic=0)
        final_stereo_input_img_folder = intermediate_ortho_dir
    else:
        final_stereo_input_img_folder = img_folder
    # now run stereo
    print("Running final stereo reconstruction")
    workflow.execute_skysat_stereo(final_stereo_input_img_folder,
                                    final_stereo_dir, ba_prefix=ba_prefix + '-run',
                                    mode='triplet', session=final_stereo_session,
                                    dem=ortho_dem, texture='normal', writeout_only=False,
                                    block=block_matching, crop_map=1, threads=2, overlap_pkl=overlap_stereo_pkl,
                                    cross_track=False)


if 6 in steps2run:
    print('\n--------------------')
    print('Step 6: Mosaic DEMs')
    print('--------------------\n')
    pc_list = sorted(glob.glob(os.path.join(final_stereo_dir, '20*/2*/run-PC.tif')))
    print(f"Identified {len(pc_list)} clouds")
    # this is dem gridding followed by mosaicing
    workflow.gridding_wrapper(pc_list, tr=2)
    print("Mosaicing DEMs")
    workflow.dem_mosaic_wrapper(dir=os.path.abspath(final_stereo_dir), mode='triplet',
                                out_folder=os.path.abspath(mos_dem_dir))


if 7 in steps2run:
    print('\n--------------------')
    print('Step 7: Align DEMs')
    print('--------------------\n')
    # now perform alignment
    median_mos_dem = glob.glob(os.path.join(mos_dem_dir, '*_median_mos.tif'))[0]
    print("Aligning DEMs")
    workflow.alignment_wrapper_single(coreg_dem, source_dem=median_mos_dem, max_displacement=100, outprefix=os.path.join(alignment_dir, 'run-run'))


if 8 in steps2run:
    print('\n--------------------')
    print('Step 8: Align frame camera models')
    print('--------------------\n')
    camera_list = sorted(glob.glob(os.path.join(init_ba, 'run-run-*.tsai')))
    print(f"Detected {len(camera_list)} cameras to be registered to DEM")
    alignment_vector = glob.glob(os.path.join(alignment_dir, 'alignment_vector.txt'))[0]
    if not os.path.exists(aligned_cam_dir):
        os.mkdir(aligned_cam_dir)
    print("Aligning cameras")
    workflow.align_cameras_wrapper(input_camera_list=camera_list, transform_txt=alignment_vector, outfolder=aligned_cam_dir)


if 9 in steps2run:
    print('\n--------------------')
    print('Step 9: Construct final georegistered orthomosaic')
    print('--------------------\n')
    georegistered_median_dem = os.path.join(alignment_dir, 'run-run-trans_source-DEM.tif')
    workflow.execute_skysat_orthorectification(images=img_list, data='triplet', session=final_ortho_session,
                                                outdir=final_ortho_dir, tsrs=epsg_code, dem=georegistered_median_dem,
                                                ba_prefix=os.path.join(aligned_cam_dir, 'run-run'), mode='science',
                                                overlap_list=overlap_stereo_txt, copy_rpc=0, orthomosaic=1)


if 10 in steps2run:
    print('\n--------------------')
    print('Step 10: Produce final plot of orthoimage, DEM, NMAD, and countmaps')
    print('--------------------\n')
    ortho_fn = glob.glob(os.path.join(final_ortho_dir, '*finest_orthomosaic.tif'))[0]
    count_fn = glob.glob(os.path.join(mos_dem_dir, '*count*.tif'))[0]
    nmad_fn = glob.glob(os.path.join(mos_dem_dir, '*nmad*.tif'))[0]
    georegistered_median_dem = glob.glob(os.path.join(alignment_dir, 'run-run-trans_*DEM.tif'))[0]
    workflow.plot_composite_fig(ortho_fn, georegistered_median_dem, count_fn, nmad_fn, outfn=final_figure)


print('DONE!')