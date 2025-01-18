import os,sys,glob,re,shutil
import numpy as np
import geopandas as gpd
import pandas as pd
from pygeotools.lib import iolib,malib
from tqdm import tqdm
from p_tqdm import p_umap
from skysat_stereo import skysat
from skysat_stereo import asp_utils as asp
from rpcm import geo
from skysat_stereo import misc_geospatial as misc
from shapely.geometry import Polygon
import itertools
from osgeo import osr
from pyproj import Transformer
from p_tqdm import p_map


# Step 4 - checking bundle adjust arguments
# bundle_adjust_stable
code_edit_dir = "/bsuhome/raineyaberle/scratch/SkySat-Stereo/scripts"
sys.path.append(code_edit_dir)
import bundle_adjustment_lib_edited as ba

img='/bsuhome/raineyaberle/scratch/SkySat-Stereo/study_sites/MCS/20241003/SkySatScene'
ba_prefix='/bsuhome/raineyaberle/scratch/SkySat-Stereo/study_sites/MCS/20241003/proc_out/ba_pinhole/run' 
cam='/bsuhome/raineyaberle/scratch/SkySat-Stereo/study_sites/MCS/20241003/proc_out/camgen_cam_gcp'
session='nadirpinhole'
overlap_list='/bsuhome/raineyaberle/scratch/SkySat-Stereo/study_sites/MCS/20241003/proc_out/overlap_with_overlap_perc_stereo_only.txt'
num_iter=700
num_pass=2
mode='full_triplet'
bound=None
camera_param2float='trans+rot'
dem=None
num_iter=2000
num_pass=2

img_list = sorted(glob.glob(os.path.join(img,'*.tif')))
if len(img_list) < 2:
    img_list = sorted(glob.glob(os.path.join(img, '*.tiff')))
    #img_list = [os.path.basename(x) for x in img_list]
    if os.path.islink(img_list[0]):
        img_list = [os.readlink(x) for x in img_list] 


if overlap_list is not None:
    # need to remove images and cameras which are not optimised during bundle adjustment
    # read pairs from input overlap list
    initial_count = len(img_list)
    with open(overlap_list) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    l_img = [x.split(' ')[0] for x in content]
    r_img = [x.split(' ')[1] for x in content]
    total_img = l_img + r_img
    uniq_idx = np.unique(total_img, return_index=True)[1]
    img_list = [total_img[idx] for idx in sorted(uniq_idx)]
    # Filter to images with cameras
    img_list_new = []
    for img_fn in img_list:
        cam_fn = os.path.join(cam, os.path.basename(img_fn).replace('.tif', '_rpc.tsai')) # step 2 camera model
        if os.path.exists(cam_fn):
            img_list_new.append(img_fn)
    img_list = img_list_new
    print(f"Out of the initial {initial_count} images, {len(img_list)} will be orthorectified using adjusted cameras")


if camera_param2float == 'trans+rot':
    cam_wt = 0
else:
    # this will invoke adjustment with rotation weight of 0 and translation weight of 0.4
    cam_wt = None


print(f"Camera weight is {cam_wt}")

if cam is not None:
    #cam = os.path.abspath(cam)
    if 'run' in os.path.basename(cam):
        cam_list = [glob.glob(cam+'-'+os.path.splitext(os.path.basename(x))[0]+'*.tsai')[0] for x in img_list]
        print("No of cameras is {}".format(len(cam_list)))
    else:
        cam_list = [glob.glob(os.path.join(cam,os.path.splitext(os.path.basename(x))[0]+'*.tsai'))[0] for x in img_list]


if dem:
    dem = iolib.fn_getma(dem)
    dem_stats = malib.get_stats_dict(dem)
    min_elev,max_elev = [dem_stats['min']-500,dem_stats['max']+500] 
    dem = None


# the concept is simple
#first 3 cameras, and then corresponding first three cameras from next collection are fixed in the first go
# these serve as a kind of #GCP, preventing a large drift in the triangulated points/camera extrinsics during optimization
img_time_identifier_list = np.array([os.path.basename(img).split('_')[1] for img in img_list])
img_time_unique_list = np.unique(img_time_identifier_list)
second_collection_list = np.where(img_time_identifier_list == img_time_unique_list[1])[0][[0,1,2]]
fix_cam_idx = np.array([0,1,2]+list(second_collection_list))
print(type(fix_cam_idx)) 
round1_opts = ba.get_ba_opts(
    ba_prefix, session=session,num_iterations=num_iter,num_pass=num_pass,
    fixed_cam_idx=fix_cam_idx,overlap_list=overlap_list,camera_weight=cam_wt)
# enter round2_opts here only ?
if session == 'nadirpinhole':
    ba_args = img_list+ cam_list
else:
    ba_args = img_list


# print("Running round 1 bundle adjustment for given triplet stereo combination")
print('\nROUND 1:')
print('bundle_adjust', round1_opts, ba_args)
# run_cmd('bundle_adjust', round1_opts+ba_args)
    
# Save the first and foremost bundle adjustment reprojection error file
# init_residual_fn_def = sorted(glob.glob(ba_prefix+'*initial*residuals*pointmap*.csv'))[0]
# init_residual_fn = os.path.splitext(init_residual_fn_def)[0]+'_initial_reproj_error.csv' 
# init_per_cam_reproj_err = sorted(glob.glob(ba_prefix+'-*initial_residuals_raw_pixels.txt'))[0]
# init_per_cam_reproj_err_disk = os.path.splitext(init_per_cam_reproj_err)[0]+'_initial_per_cam_reproj_error.txt'
# init_cam_stats = sorted(glob.glob(ba_prefix+'-*initial_residuals_stats.txt'))[0]
# init_cam_stats_disk = os.path.splitext(init_cam_stats)[0]+'_initial_camera_stats.txt'
# shutil.copy2(init_residual_fn_def,init_residual_fn)
# shutil.copy2(init_per_cam_reproj_err,init_per_cam_reproj_err_disk)
# shutil.copy2(init_cam_stats,init_cam_stats_disk)
    
identifier = os.path.basename(cam_list[0]).split('_',14)[0][:2]
print(ba_prefix+'-{}*.tsai'.format(identifier))
cam_list = sorted(glob.glob(os.path.join(ba_prefix+ '-{}*.tsai'.format(identifier))))
ba_args = img_list+cam_list
fixed_cam_idx2 = np.delete(np.arange(len(img_list),dtype=int),fix_cam_idx)
round2_opts = ba.get_ba_opts(ba_prefix, overlap_list=overlap_list,session=session,
                            fixed_cam_idx=fixed_cam_idx2,camera_weight=cam_wt)

print('\nROUND 2:')
print('bundle_adjust', round2_opts, ba_args)
    # print("running round 2 bundle adjustment for given triplet stereo combination")
    # run_cmd('bundle_adjust', round2_opts+ba_args)
        

# Step 2 - adding distortion to camera models
# workflow.skysat_preprocess
img_folder = '/bsuhome/raineyaberle/scratch/SkySat-Stereo/study_sites/MCS/20241003/SkySatScene'
mode='triplet'
product_level='l1b'
overlap_pkl='/bsuhome/raineyaberle/scratch/SkySat-Stereo/study_sites/MCS/20241003/proc_out/overlap_with_overlap_perc_stereo_only.pkl'
dem='/bsuhome/raineyaberle/scratch/SkySat-Stereo/study_sites/MCS/20241003/proc_out/refdem/MCS_REFDEM_WGS84.tif'
outdir='/bsuhome/raineyaberle/scratch/SkySat-Stereo/study_sites/MCS/20241003/proc_out/camgen_cam_gcp'
sampling=None
frame_index=None
sampler=5

df = pd.read_pickle(overlap_pkl)
img_list = list(np.unique(np.array(list(df.img1.values)+list(df.img2.values))))
img_list = [os.path.splitext(os.path.basename(img))[0] for img in img_list]
cam_list = [glob.glob(os.path.join(img_folder,'{}*.tif'.format(img)))[0] for img in img_list]
n = len(img_list)
if product_level == 'l1b':
    pitch = [0.8]*n
else:
    pitch = [1.0]*n


out_fn = [os.path.join(outdir,'{}_rpc.tsai'.format(frame)) for frame in img_list]
out_gcp = [os.path.join(outdir,'{}_rpc.gcp'.format(frame)) for frame in img_list]
camera = cam_list
frame_index = [None]*n
img_list = cam_list
gcp_factor = 8

fl = [553846.153846]*n
cx = [1280]*n
cy = [540]*n
ht_datum = [malib.get_stats_dict(iolib.fn_getma(dem))['median']]*n # use this value for height where DEM has no-data
gcp_std = [1]*n
datum = ['WGS84']*n
refdem = [dem]*n
n_proc = 30

#n_proc = cpu_count()
# print("Starting camera resection procedure")
# cam_gen_log = p_map(asp.cam_gen,img_list,fl,cx,cy,pitch,ht_datum,gcp_std,out_fn,out_gcp,datum,refdem,camera,frame_index,num_cpus = n_proc)
# print("writing gcp with basename removed")
# # count expexted gcp 
# print(f"Total expected GCP {gcp_factor*n}")    
# asp.clean_gcp(out_gcp,outdir) 



# Step 5 - errors with finding camera files
# execute_skysat_orhtorectification
img_folder = '/bsuhome/raineyaberle/scratch/SkySat-Stereo/study_sites/MCS/20241003/SkySatScene'
images = glob.glob(os.path.join(img_folder, '*.tif'))
data='triplet'
session='pinhole'
outdir='/bsuhome/raineyaberle/scratch/SkySat-Stereo/study_sites/MCS/20241003/proc_out/intermediate_pinhole_ortho'
tsrs='EPSG:32611'
dem='/bsuhome/raineyaberle/scratch/SkySat-Stereo/study_sites/MCS/20241003/proc_out/refdem/MCS_REFDEM_WGS84.tif'
ba_prefix='/bsuhome/raineyaberle/scratch/SkySat-Stereo/study_sites/MCS/20241003/proc_out/ba_pinhole/run' + '-run'
mode='science'
overlap_list='/bsuhome/raineyaberle/scratch/SkySat-Stereo/study_sites/MCS/20241003/proc_out/overlap_with_overlap_perc_stereo_only.txt'
copy_rpc=1
orthomosaic=0
tr=None
del_opt=False
cam_folder=None
mode='science'
frame_index_fn=None

img_list = images
if overlap_list is not None:
    # need to remove images and cameras which are not optimised during bundle adjustment
    # read pairs from input overlap list
    initial_count = len(img_list)
    with open(overlap_list) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    l_img = [x.split(' ')[0] for x in content]
    r_img = [x.split(' ')[1] for x in content]
    total_img = l_img + r_img
    uniq_idx = np.unique(total_img, return_index=True)[1]
    img_list = [total_img[idx] for idx in sorted(uniq_idx)]
    # Filter to images with cameras
    if ba_prefix is not None:
        img_list_new = []
        for img_fn in img_list:
            cam_fn = glob.glob(os.path.abspath(ba_prefix)+'-'+os.path.splitext(os.path.basename(img_fn))[0]+'*.tsai')
            if len(cam_fn) > 0:
                img_list_new.append(img_fn)
        img_list = img_list_new
    print(f"Out of the initial {initial_count} images, {len(img_list)} will be orthorectified using adjusted cameras")


img_prefix = [os.path.splitext(os.path.basename(img))[0] for img in img_list]
out_list = [os.path.join(outdir,img+'_map.tif') for img in img_prefix]
session_list = [session]*len(img_list)
dem_list = [dem]*len(img_list)
tr_list = [tr]*len(img_list)
srs_list = [tsrs]*len(img_list)

if session == 'pinhole':
    if ba_prefix:
        cam_list = [glob.glob(os.path.abspath(ba_prefix)+'-'+os.path.splitext(os.path.basename(x))[0]+'*.tsai')[0] for x in img_list]
        print("No of cameras is {}".format(len(cam_list)))
    else:
        print(os.path.join(os.path.abspath(cam_folder),os.path.splitext(os.path.basename(img_list[0]))[0]+'*.tsai'))
        cam_list = [glob.glob(os.path.join(os.path.abspath(cam_folder),os.path.splitext(os.path.basename(x))[0]+'*.tsai'))[0] for x in img_list]
else:
    cam_list = [None]*len(img_list)
    if ba_prefix:
        # not yet implemented
        ba_prefix_list = [ba_prefix]*len(img_list)

print("Mapping given images")
ortho_logs = p_map(asp.mapproject,img_list,out_list,session_list,dem_list,tr_list,srs_list,cam_list,
    [None]*len(img_list),[None]*len(img_list),num_cpus=int(iolib.cpu_count()/4))
ortho_log = os.path.join(outdir,'ortho.log')
print("Saving Orthorectification log at {}".format(ortho_log))
with open(ortho_log,'w') as f:
    for log in ortho_logs:
        f.write(log)

if copy_rpc == 1:
    print("Copying RPC from native image to orthoimage in parallel")
    try:
        copy_rpc_out = p_map(skysat.copy_rpc,img_list,out_list,num_cpus=iolib.cpu_count())
    except Exception as e:
        print(e)


if orthomosaic == 1:
    print("Will also produce median, weighted average and highest resolution orthomosaic")
    if data == 'triplet':
        # sort images based on timestamps and resolutions
        img_list, time_list = skysat.sort_img_list(out_list)
        res_sorted_list = skysat.res_sort(out_list)
        # define mosaic prefix containing timestamps of inputs
        mos_prefix = '_'.join(np.unique([t.split('_')[0] for t in time_list]))+'__'+'_'.join(np.unique([t.split('_')[1] for t in time_list]))
        # define output filenames
        res_sorted_mosaic = os.path.join(outdir,'{}_finest_orthomosaic.tif'.format(mos_prefix))
        median_mosaic = os.path.join(outdir,'{}_median_orthomosaic.tif'.format(mos_prefix))
        wt_avg_mosaic = os.path.join(outdir,'{}_wt_avg_orthomosaic.tif'.format(mos_prefix))
        indi_mos_list = [os.path.join(outdir,f'{time}_first_orthomosaic.tif') for time in time_list]
        print("producing finest resolution on top mosaic, per-pixel median and wt_avg mosaic")
        all_3_view_mos_logs = p_map(asp.dem_mosaic, [res_sorted_list]*3, [res_sorted_mosaic,median_mosaic,wt_avg_mosaic], 
                                    ['None']*3, [None]*3, ['first','median',None],[None]*3,num_cpus=4)
        print("producing idependent mosaic for different views in parallel")
        indi_mos_count = len(time_list)
        if indi_mos_count>3:
            tile_size = 400
        else:
            tile_size = None
        indi_mos_log = p_map(asp.dem_mosaic,img_list, indi_mos_list, ['None']*indi_mos_count, [None]*indi_mos_count, 
            ['first']*indi_mos_count,[tile_size]*indi_mos_count)
        # write out log files
        out_log = os.path.join(outdir,'science_mode_ortho_mos.log')
        total_mos_log = all_3_view_mos_logs+indi_mos_log
        print("Saving orthomosaic log at {}".format(out_log))
        with open(out_log,'w') as f:
            for log in itertools.chain.from_iterable(total_mos_log):
                f.write(log)