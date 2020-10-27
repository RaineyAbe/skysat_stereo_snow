#! /usr/bin/env python
import os,sys,glob,shutil
import subprocess
import argparse
from distutils.spawn import find_executable
from pygeotools.lib import iolib,malib
import geopandas as gpd
import numpy as np
from datetime import datetime
import pandas as pd
from multiprocessing import cpu_count

# Usage: ba_skysat.py -mode full_video,full_triplet,quick_transform_pc_align,general_ba -t pinhole,rpc -img image_folder -cam optional (rpc might not require it) -ba_prefix out_ba -overlap-list -init_transform -gcp gcp_folder or file
# TODO:
# Keep other passed arguments flexible for extending as general purpose, like gcp_list. Others which go into ba_opt can be checked with None construct when variables are initailized in main command
# maybe put all arguments and check if os.path.abspath can be done during runtime from the get_ba_opts function


def run_cmd(bin, args, **kw):
    # Note, need to add full executable
    # from dshean/vmap.py
    #binpath = os.path.join('/home/sbhushan/src/StereoPipeline/bin',bin)
    binpath = find_executable(bin)
    if binpath is None:
        msg = ("Unable to find executable %s\n"
        "Install ASP and ensure it is in your PATH env variable\n"
       "https://ti.arc.nasa.gov/tech/asr/intelligent-robotics/ngt/stereo/")
        sys.exit(msg)
    # binpath = os.path.join('/opt/StereoPipeline/bin/',bin)
    call = [binpath, ]
    print(call)
    call.extend(args)
    print(call)
	# print(type(call))
    # print(' '.join(call))
    try:
        code = subprocess.call(call, shell=False)
    except OSError as e:
        raise Exception('%s: %s' % (binpath, e))
    if code != 0:
        raise Exception('ASP step ' + kw['msg'] + ' failed')


def get_ba_opts(ba_prefix, camera_weight=None,translation_weight=0.4,rotation_weight=0,fixed_cam_idx=None,overlap_list=None, overlap_limit=None, initial_transform=None, input_adjustments=None, flavor='general_ba', session='nadirpinhole', gcp_transform=False,num_iterations=2000,num_pass=2,lon_lat_limit=None,elevation_limit=None):
    ba_opt = []
    ba_opt.extend(['--threads', str(cpu_count())])
    ba_opt.extend(['-o', ba_prefix])
    ba_opt.extend(['--min-matches', '4'])
    ba_opt.extend(['--disable-tri-ip-filter'])
    ba_opt.extend(['--force-reuse-match-files'])
    ba_opt.extend(['--ip-per-tile', '4000'])
    ba_opt.extend(['--ip-inlier-factor', '0.2'])
    ba_opt.extend(['--ip-num-ransac-iterations', '1000'])
    ba_opt.extend(['--skip-rough-homography'])
    ba_opt.extend(['--min-triangulation-angle', '0.0001'])
    ba_opt.extend(['--save-cnet-as-csv'])
    ba_opt.extend(['--individually-normalize'])
    #ba_opt.extend(['--robust-threshold', '10'])
    if camera_weight is not None:
        ba_opt.extend(['--camera-weight', str(camera_weight)])
    else:
        ba_opt.extend(['--translation-weight',str(translation_weight)])
        ba_opt.extend(['--rotation-weight',str(rotation_weight)])
    if fixed_cam_idx is not None:
        ba_opt.extend(['--fixed-camera-indices',' '.join(fixed_cam_idx.astype(str))])
    ba_opt.extend(['-t', session])
    ba_opt.extend(['--remove-outliers-params', '75 3 5 6'])
    # How about adding num random passes here ? Think about it, it might help if we are getting stuck in local minima :)
    if session == 'nadirpinhole':
        ba_opt.extend(['--inline-adjustments'])
        ba_opt.extend(['--num-iterations', str(num_iterations)])
        ba_opt.extend(['--num-passes', str(num_pass)])
    #ba_opt.extend(['--parameter-tolerance','1e-14'])
    # gcp_transform=True
    if gcp_transform:
        ba_opt.extend(['--transform-cameras-using-gcp'])
        # maybe add gcp arg here, can be added when function is called as well
    if initial_transform:
        ba_opt.extend(['--initial-transform', initial_transform])
    if input_adjustments:
        ba_opt.extend(['--input-adjustments', input_adjustments])
    if overlap_limit:
        ba_opt.extend(['--overlap-limit',str(overlap_limit)])
    if overlap_list:
        ba_opt.extend(['--overlap-list', overlap_list])
    if lon_lat_limit:
        ba_opt.extend(['--lon-lat-limit',str(lon_lat_limit[0]),str(lon_lat_limit[1]),str(lon_lat_limit[2]),str(lon_lat_limit[3])])
    if elevation_limit:
        ba_opt.extend(['--elevation-limit',str(elevation_limit[0]),str(elevation_limit[1])])
    return ba_opt


def getparser():
    parser = argparse.ArgumentParser(
        description='Script for performing bundle adjustment, with several custom flavors built-in based on recent use-cases')
    ba_choices = ['full_video', 'full_triplet',
        'transform_pc_align', 'general_ba']
    parser.add_argument('-mode', default='full_video', choices=ba_choices,
                        help='bundle adjust workflow to implement (default: %(default)s)')
    session_choices = ['nadirpinhole', 'rpc']
    parser.add_argument('-t', default='nadirpinhole', choices=session_choices,
                        help='choose between pinhole and rpc mode (default: %(default)s)')
    parser.add_argument('-ba_prefix', default=None,
                        help='output prefix for ba output', required=True)
    parser.add_argument('-img', default=None,
                        help='directory containing images', required=True)
    parser.add_argument(
        '-cam', default=None, help='directory containing cameras, if using pinhole. RPC model expects information in GDAL header')
    # parser.add_argument('-gcp',default=None,help='list of gcps',nargs='+',required=False)
    parser.add_argument('-gcp', default=None,
                        help='folder containing list of gcps', required=False)
    parser.add_argument('-initial_transform', default=None,
                        help='.txt file produced by pc_align, which can be used to translate cameras to that position')
    parser.add_argument('-input_adjustments', default=None,
                        help='ba_prefix from previous ba_run if using RPC or not using inline adjustments with pinhole')
    parser.add_argument('-overlap_list', default=None,
                        help='list containing pairs for which feature matching will be restricted to')
    parser.add_argument('-overlap_limit', default=20,
                        help='default overlap limit for video sequence over which feature would be matched  (default: %(default)s)')
    parser.add_argument('-frame_index',default=None,help='subsampled frame_index.csv produced by preprocessing script (default: %(default)s)')
    parser.add_argument('-num_iter',default=2000,help='defualt number of iterations (default: %(default)s)')
    parser.add_argument('-num_pass',default=2,help='defualt number of solver passes, eliminating points with high reprojection error at each pass (default: %(default)s)')
    camera_param_float_ch = ['trans+rot','rot_only']
    parser.add_argument('-camera_param2float',type=str,default='trans+rot',choices=camera_param_float_ch,help='either float translation and rotation parameters freely, or enforce a higher tranlsation weight and allow free float of rotation parameters, incase the satellite positions are known accurately.')
    parser.add_argument('-dem',default=None,help='DEM to filter match points after optimization')
    parser.add_argument('-bound',default=None,help='Bound shapefile to limit extent of match points after optimization')
    return parser


def main():
    parser = getparser()
    args = parser.parse_args()
    img = args.img
    img_list = sorted(glob.glob(os.path.join(img, '*.tif')))
    if len(img_list) < 2:
        img_list = sorted(glob.glob(os.path.join(img, '*.tiff')))
        #img_list = [os.path.basename(x) for x in img_list]
        if os.path.islink(img_list[0]):
            img_list = [os.readlink(x) for x in img_list]  
    if args.cam:
        cam = os.path.abspath(args.cam)
        if 'run' in os.path.basename(cam):
            cam_list = sorted(glob.glob(cam+'-*.tsai'))
        else:
            cam_list = sorted(glob.glob(os.path.join(cam, '*.tsai')))
        cam_list = cam_list[:len(img_list)]
    session = args.t
    if args.ba_prefix:
        ba_prefix = args.ba_prefix
    if args.initial_transform:
        initial_transform = os.path.abspath(initial_transform)
    if args.input_adjustments:
        input_adjustments = os.path.abspath(input_adjustments)
    if args.overlap_list:
        overlap_list = os.path.abspath(args.overlap_list)
    if args.gcp:
        gcp_list = sorted(glob.glob(os.path.join(args.gcp, '*.gcp')))
    ba_prefix = os.path.abspath(args.ba_prefix)
    mode = args.mode
    if args.bound:
        bound = gpd.read_file(args.bound)
        geo_crs = {'init':'epsg:4326'}
        if bound.crs is not geo_crs:
           bound = bound.to_crs(geo_crs)
        lon_min,lat_min,lon_max,lat_max = bound.total_bounds
    if args.camera_param2float == 'trans+rot':
        cam_wt = 0
    else:
        # this will invoke adjustment with rotation weight of 0 and translation weight of 4
        cam_wt = None
    print(f"Camera weight is {cam_wt}")

    if args.dem:
        dem = iolib.fn_getma(args.dem)
        dem_stats = malib.get_stats_dict(dem)
        min_elev,max_elev = [dem_stats['min']-500,dem_stats['max']+500] 
    if mode == 'full_video':
        frame_index = args.frame_index
        df = pd.read_csv(frame_index)
        gcp = os.path.abspath(args.gcp)
        df['dt'] = [datetime.strptime(date.split('+00:00')[0],'%Y-%m-%dT%H:%M:%S.%f') for date in df.datetime.values]
        delta = (df.dt.values[1]-df.dt.values[0])/np.timedelta64(1, 's')
        # i hardocde overlap limit to have 40 seconds coverage
        overlap_limit = np.int(np.ceil(40/delta))
        print("Calculated overlap limit as {}".format(overlap_limit))
        img_list = [glob.glob(os.path.join(img,'*{}*.tiff'.format(x)))[0] for x in df.name.values]
        cam_list = [glob.glob(os.path.join(cam,'*{}*.tsai'.format(x)))[0] for x in df.name.values]
        gcp_list = [glob.glob(os.path.join(gcp,'*{}*.gcp'.format(x)))[0] for x in df.name.values]
        #also append the clean gcp here
        print(os.path.join(gcp,'*clean*_gcp.gcp'))
        gcp_list.append(glob.glob(os.path.join(gcp,'*clean*_gcp.gcp'))[0])
        # this attempt did not work here
        # but given videos small footprint, the median (scale)+trans+rotation is good enough for all terrain
        # so reverting back to them
        #stereo_baseline = 10
        #fix_cam_idx = np.array([0]+[0+stereo_baseline])
        round1_opts = get_ba_opts(
            ba_prefix, overlap_limit=overlap_limit, flavor='2round_gcp_1', session=session,num_iterations=args.num_iter,camera_weight=cam_wt,fixed_cam_idx=None)
        print("Running round 1 bundle adjustment for input video sequence")
        if session == 'nadirpinhole':
            ba_args = img_list+cam_list
        else:
            ba_args = img_list
        # Check if this command executed till last
        print('Running bundle adjustment round1')
        run_cmd('bundle_adjust', round1_opts+ba_args)
       
        # Make files used to evaluate solution quality
        init_residual_fn_def = sorted(glob.glob(ba_prefix+'*initial*no_loss_*pointmap*.csv'))[0]
        init_per_cam_reproj_err = sorted(glob.glob(ba_prefix+'-*initial_residuals_no_loss_function_raw_pixels.txt'))[0]
        init_per_cam_reproj_err_disk = os.path.splitext(init_per_cam_reproj_err)[0]+'_initial_per_cam_reproj_error.txt'
        init_residual_fn = os.path.splitext(init_residual_fn_def)[0]+'_initial_reproj_error.csv' 
        shutil.copy2(init_residual_fn_def,init_residual_fn)
        shutil.copy2(init_per_cam_reproj_err,init_per_cam_reproj_err_disk)
        # Copy final reprojection error files before transforming cameras
        final_residual_fn_def = sorted(glob.glob(ba_prefix+'*final*no_loss_*pointmap*.csv'))[0]
        final_residual_fn = os.path.splitext(final_residual_fn_def)[0]+'_final_reproj_error.csv'
        final_per_cam_reproj_err = sorted(glob.glob(ba_prefix+'-*final_residuals_no_loss_function_raw_pixels.txt'))[0]
        final_per_cam_reproj_err_disk = os.path.splitext(final_per_cam_reproj_err)[0]+'_final_per_cam_reproj_error.txt'
        shutil.copy2(final_residual_fn_def,final_residual_fn)
        shutil.copy2(final_per_cam_reproj_err,final_per_cam_reproj_err_disk)

        if session == 'nadirpinhole':
            identifier = os.path.basename(cam_list[0]).split(df.name.values[0])[0]
            print(ba_prefix+identifier+'-{}*.tsai'.format(df.name.values[0]))
            cam_list = [glob.glob(ba_prefix+identifier+'-{}*.tsai'.format(img))[0] for img in df.name.values]
            print(len(cam_list))
            ba_args = img_list+cam_list+gcp_list
            
            #fixed_cam_idx2 = np.delete(np.arange(len(img_list),dtype=int),fix_cam_idx)
            round2_opts = get_ba_opts(
                ba_prefix, overlap_limit = overlap_limit, flavor='2round_gcp_2', session=session, gcp_transform=True,camera_weight=0,
                num_iterations=0,num_pass=1)
        else:
            # round 1 is adjust file
            input_adjustments = ba_prefix
            round2_opts = get_ba_opts(
                ba_prefix, overlap_limit = overlap_limit, input_adjustments=ba_prefix, flavor='2round_gcp_2', session=session)
            ba_args = img_list+gcp_list
        print("running round 2 bundle adjustment for input video sequence")
        run_cmd('bundle_adjust', round2_opts+ba_args)
        
    elif mode == 'full_triplet':
        if args.overlap_list is None:
            print(
                "Attempted bundle adjust will be expensive, will try to find matches in each and every pair")
        # the concept is simple
        #first 3 cameras, and then corresponding first three cameras from next collection are fixed in the first go
        # these serve as a kind of #GCP, preventing a large drift in the triangulated points/camera extrinsics during optimization
        img_time_identifier_list = np.array([os.path.basename(img).split('_')[1] for img in img_list])
        img_time_unique_list = np.unique(img_time_identifier_list)
        second_collection_list = np.where(img_time_identifier_list == img_time_unique_list[1])[0][[0,1,2]]
        fix_cam_idx = np.array([0,1,2]+list(second_collection_list))
        print(type(fix_cam_idx)) 
      
        round1_opts = get_ba_opts(
            ba_prefix, session=session,num_iterations=args.num_iter,num_pass=args.num_pass,fixed_cam_idx=fix_cam_idx,overlap_list=args.overlap_list,camera_weight=cam_wt)
            # enter round2_opts here only ?
        if session == 'nadirpinhole':
            ba_args = img_list+ cam_list
        else:
            ba_args = img_list
        print("Running round 1 bundle adjustment for given triplet stereo combination")
        run_cmd('bundle_adjust', round1_opts+ba_args)
        # Save the first and foremost bundle adjustment reprojection error file
        init_residual_fn_def = sorted(glob.glob(ba_prefix+'*initial*no_loss_*pointmap*.csv'))[0]
        init_residual_fn = os.path.splitext(init_residual_fn_def)[0]+'_initial_reproj_error.csv' 
        init_per_cam_reproj_err = sorted(glob.glob(ba_prefix+'-*initial_residuals_no_loss_function_raw_pixels.txt'))[0]
        init_per_cam_reproj_err_disk = os.path.splitext(init_per_cam_reproj_err)[0]+'_initial_per_cam_reproj_error.txt'
        shutil.copy2(init_residual_fn_def,init_residual_fn)
        shutil.copy2(init_per_cam_reproj_err,init_per_cam_reproj_err_disk)
        if session == 'nadirpinhole':
            identifier = os.path.basename(cam_list[0]).split('_',14)[0][:2]
            print(ba_prefix+'-{}*.tsai'.format(identifier))
            cam_list = sorted(glob.glob(os.path.join(ba_prefix+ '-{}*.tsai'.format(identifier))))
            ba_args = img_list+cam_list
            fixed_cam_idx2 = np.delete(np.arange(len(img_list),dtype=int),fix_cam_idx)
            round2_opts = get_ba_opts(ba_prefix, overlap_list=overlap_list,session=session, fixed_cam_idx=fixed_cam_idx2,camera_weight=cam_wt)
        else:
            # round 1 is adjust file
            input_adjustments = ba_prefix
            round2_opts = get_ba_opts(
                ba_prefix, overlap_limit, input_adjustments=ba_prefix, flavor='2round_gcp_2', session=session,elevation_limit=[min_elev,max_elev],lon_lat_limit=[lon_min,lat_min,lon_max,lat_max])
            ba_args = img_list+gcp_list
        print("running round 2 bundle adjustment for given triplet stereo combination")
        run_cmd('bundle_adjust', round2_opts+ba_args)
        final_residual_fn_def = sorted(glob.glob(ba_prefix+'*final*no_loss_*pointmap*.csv'))[0]
        final_residual_fn = os.path.splitext(final_residual_fn_def)[0]+'_final_reproj_error.csv'
        shutil.copy2(final_residual_fn_def,final_residual_fn)
        final_per_cam_reproj_err = sorted(glob.glob(ba_prefix+'-*final_residuals_no_loss_function_raw_pixels.txt'))[0]
        final_per_cam_reproj_err_disk = os.path.splitext(final_per_cam_reproj_err)[0]+'_final_per_cam_reproj_error.txt'
        shutil.copy2(final_per_cam_reproj_err,final_per_cam_reproj_err_disk)

        # input is just a transform from pc_align or something similar with no optimization
        if mode == 'transform_pc_align':
            if session == 'nadirpinhole':
                if args.gcp:
                    ba_args = img_list+cam_list+gcp_list
                    ba_opt = get_ba_opts(ba_prefix,overlap_list,flavor='2round_gcp_2',session=session,gcp_transform=True)
                else:
                    ba_args = img_list+cam_list+gcp_list
                    ba_opt = get_ba_opts(ba_prefix,overlap_list,flavor='2round_gcp_2',session=session,gcp_transform=True)
            else:
                if args.gcp:
                    ba_args = img_list+gcp_list
                    ba_opt = get_ba_opts(ba_prefix,overlap_list,initial_transform=initial_transform,flavor='2round_gcp_2',session=session,gcp_transform=True)
                else:
                    ba_args = img_list+gcp_list
                    ba_opt = get_ba_opts(ba_prefix,overlap_list,initial_transform=initial_transform,flavor='2round_gcp_2',session=session,gcp_transform=True)
            print("Simply transforming the cameras without optimization")
            run_cmd('bundle_adjust',ba_opt+ba_args,'Running bundle adjust')
            
            # general usecase bundle adjust
            if mode == 'general_ba':
                round1_opts = get_ba_opts(ba_prefix,overlap_limit=args.overlap_limit,flavor='2round_gcp_1',session=session)
                print ("Running general purpose bundle adjustment")
                if session == 'nadirpinhole':
                    ba_args = img_list+cam_list
                else:
                    ba_args = img_list
                # Check if this command executed till last
                run_cmd('bundle_adjust',round1_opts+ba_args,'Running bundle adjust')
        print("Script is complete !")
if  __name__=="__main__":
    main()
