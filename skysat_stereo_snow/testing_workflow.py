import os
import glob
import sys
import shutil
import subprocess
import pandas as pd
from tqdm import tqdm
import numpy as np
import multiprocessing
from skysat_stereo import asp_utils as asp

def run_cmd(bin, args, **kw):
    """
    wrapper around subprocess function to excute bash commands
    Parameters
    ----------
    bin: str
        command to be excuted (e.g., stereo or gdalwarp)
    args: list
        arguments to the command as a list
    Retuns
    ----------
    out: str
        log (stdout) as str if the command executed, error message if the command failed
    """
    #Note, need to add full executable
    # binpath = '/Users/raineyaberle/Research/PhD/SnowDEMs/StereoPipeline-3.5.0-alpha-2024-10-05-x86_64-OSX/bin/' + bin
    binpath = shutil.which(bin)
    # print(binpath)
    if binpath is None:
        msg = ("Unable to find executable %s\n"
        "Install ASP and ensure it is in your PATH env variable\n"
       "https://ti.arc.nasa.gov/tech/asr/intelligent-robotics/ngt/stereo/" % bin)
        sys.exit(msg)
    call = [binpath,]
    if args is not None: 
        call.extend(args)
    try:
        out = subprocess.run(call,check=True,capture_output=True,encoding='UTF-8').stdout
    except:
        out = "the command {} failed to run, see corresponding asp log".format(call)
    return out


def align_images(overlap_pkl, cam_folder, out_folder, clean_up=False, threads=None):
    # make output directory if it doesn't exist
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    # determine how many threads to use in processing
    if not threads:
        threads = multiprocessing.cpu_count()
    # define function to get associated camera model
    def get_camera_file(im_fn, cam_fol):
        cam = glob.glob(os.path.join(cam_fol, os.path.splitext(os.path.basename(im_fn))[0].replace('_map','') + '*.tsai'))
        if len(cam) > 0:
            return cam[0]
        else:
            raise FileNotFoundError(f"Camera not found for image '{im_fn}'.\nCheck the input overlap file and camera folder.")
    # load overlapping image pairs
    overlap = pd.read_pickle(overlap_pkl)
    # sort by descending overlap percentage
    overlap.sort_values(by='overlap_perc', ascending=False, inplace=True)
    overlap.reset_index(drop=True, inplace=True)
    # Initialize list of aligned and unaligned images
    im_list = set(list(overlap['img1'].values) + list(overlap['img2'].values))
    im_align_list = []
    im_unalign_list = im_list
    # Start with most overlapping image pair
    print('Starting with most overlapping image pair')
    img1 = overlap.iloc[0]['img1']
    cam1 = get_camera_file(img1, cam_folder)
    overlap_im = overlap.loc[(overlap['img1']==img1) | (overlap['img2']==img1)]
    overlap_im.reset_index(drop=True, inplace=True)
    img2 = [x for x in overlap_im.iloc[0][['img1', 'img2']].values if x!=img1][0]
    cam2 = get_camera_file(img2, cam_folder)
    overlap_perc = overlap_im.iloc[0]['overlap_perc']
    # copy img1 to aligned folder
    shutil.copy(img1, os.path.join(out_folder, os.path.splitext(os.path.basename(img1))[0] + '_align.tif'))
    shutil.copy(cam1, os.path.join(out_folder, os.path.splitext(os.path.basename(cam1))[0] + '_align.tif'))
    # update aligned lists
    im_align_list += [img1]
    im_unalign_list = [x for x in im_list if x not in im_align_list]
    # Start a progress bar
    pbar = tqdm(total=len(im_unalign_list))
    # while im_unalign_list:  
    print(f'Aligning {os.path.basename(img2)} to {os.path.basename(img1)}.')
    print(f'Overlap percent = {overlap_perc} %')
    # Make folder for img2 alignment outputs
    img2_folder = os.path.join(out_folder, os.path.basename(img2).split('_basic')[0])
    if not os.path.exists(img2_folder):
        os.mkdir(img2_folder)
    # Define output files
    output_prefix = os.path.join(img2_folder, 'run')
    f = output_prefix + '-F.tif'
    img2_align = os.path.join(img2_folder, os.path.splitext(os.path.basename(img2))[0] + '_align.tif')
    img2_transform = output_prefix + '-transform.txt'
    # Run stereo in correlator mode
    if (not os.path.exists(f)) & (not os.path.exists(img2_align)):
        print('Running image correlator...')
        cmd = ['--correlator-mode', 
                '--stereo-algorithm', 'asp_mgm', 
                '--subpixel-mode', '3', 
                '--threads', str(threads),
                img1, img2, output_prefix]
        out = run_cmd('parallel_stereo', cmd)
        print(out)
    # Run image_align using the filtered disparity map
    if not os.path.exists(img2_align):
        print('Aligning image...')
        cmd = [img1, img2, 
                '--output-image', img2_align,
                '--output-prefix', output_prefix,
                '--alignment-transform', 'rigid',
                '--ecef-transform-type', 'rigid',
                '--dem1', img1,
                '--dem2', img2,
                '--disparity-params', f + " 1000000",
                '--threads', str(threads)]
        out = run_cmd('image_align', cmd)
        print(out)
    # Apply the transform to img2 camera
    print('Aligning camera...')
    cmd = [img1, img2, cam1, cam2,
           '--initial-transform', img2_transform,
           '--apply-initial-transform-only', 
           '-o', output_prefix]
    out = run_cmd('bundle_adjust', cmd)
    print(out)
    # update list of aligned and unaligned images
    im_align_list += [img2]
    im_unalign_list = [x for x in im_list if x not in im_align_list]
    # update progress bar
    pbar.update(1)
    # Determine which image pair to align next
    overlap_remaining = overlap[
        (overlap['img1'].isin(im_align_list) & overlap['img2'].isin(im_unalign_list)) |
        (overlap['img2'].isin(im_align_list) & overlap['img1'].isin(im_unalign_list))
    ]
    if overlap_remaining.empty:
        print("No more overlapping images to align.")
        # break
    # Select the next image pair with the highest overlap
    overlap_remaining = overlap_remaining.sort_values(by='overlap_perc', ascending=False).reset_index(drop=True)
    next_row = overlap_remaining.iloc[0]
    # make sure the aligned image is set to img1, unaligned image is set to img2
    if next_row['img1'] in im_align_list:
        img1 = next_row['img1']
        img2 = next_row['img2']
    else:
        img1 = next_row['img2']
        img2 = next_row['img1']
    # get the associated cameras
    cam1 = get_camera_file(img1, cam_folder)
    cam2 = get_camera_file(img2, cam_folder)
    # get overlap percentage of next alignment image pair for printing
    overlap_perc = next_row['overlap_perc']


out_fol = '/bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/MCS/20240420/proc_out'
im_fol = os.path.join(out_fol, 'init_rpc_ortho')
cam_fol = os.path.join(out_fol, 'camgen_cam_gcp')
im_align_fol = os.path.join(out_fol, 'init_image_alignment')
overlap_stereo_pkl = os.path.join(im_align_fol, 'overlap_with_overlap_perc_stereo_only.pkl')

align_images(overlap_stereo_pkl, cam_fol, im_align_fol)

# IMG1=/bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/MCS/20240420/SkySatScene/20241003_221111_ssc9d2_0012_basic_panchromatic.tif
# IMG2=/bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/MCS/20241003/SkySatScene/20241003_221037_ssc9d2_0014_basic_panchromatic.tif
# MATCH_FILE=/bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/MCS/20240420/proc_out/init_image_alignment/20240420_165822_ssc16d3_0011/run-20240420_165753_ssc16d3_0014_basic_panchromatic_map__20240420_165822_ssc16d3_0011_basic_panchromatic_map-clean.match

# pc_align                                     \
# $IMG1 $IMG2                            \
# --max-displacement 100 \
# -o /bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/MCS/20240420/proc_out/init_image_alignment/20240420_165822_ssc16d3_0011/run

# bundle_adjust \
# /bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/MCS/20240420/proc_out/init_rpc_ortho/20240420_165753_ssc16d3_0014_basic_panchromatic_map.tif \
# /bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/MCS/20240420/proc_out/init_rpc_ortho/20240420_165822_ssc16d3_0011_basic_panchromatic_map.tif \
# /bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/MCS/20240420/proc_out/camgen_cam_gcp/20240420_165753_ssc16d3_0014_basic_panchromatic_rpc.tsai \
# /bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/MCS/20240420/proc_out/camgen_cam_gcp/20240420_165822_ssc16d3_0011_basic_panchromatic_rpc.tsai \
# --initial-transform /bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/MCS/20240420/proc_out/init_image_alignment/20240420_165822_ssc16d3_0011/run-transform.txt \
# --apply-initial-transform-only \
# -o /bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/MCS/20240420/proc_out/init_image_alignment/20240420_165822_ssc16d3_0011/run

# IMG1=/bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/MCS/20240420/SkySatScene/20241003_221111_ssc9d2_0012_basic_panchromatic.tif
# IMG2=/bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/MCS/20241003/SkySatScene/20241003_221037_ssc9d2_0014_basic_panchromatic.tif
# CAM1=

# bundle_adjust $IMG1 $IMG2 left.xml right.xml \
#   --initial-transform align/run-transform.txt       \
#   --apply-initial-transform-only -o ba_align/run

# IMG1=/bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/MCS/20240420/SkySatScene/20241003_221111_ssc9d2_0012_basic_panchromatic.tif
# IMG2=/bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/MCS/20241003/SkySatScene/20241003_221037_ssc9d2_0014_basic_panchromatic.tif
# OUTPUT_IMAGE=/bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/MCS/20241003/proc_out/image_align_test/20241003_221111_ssc9d2_0012/20241003_221037_ssc9d2_0014_basic_panchromatic_align.tif
# OUTPUT_PREFIX=/bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/MCS/20241003/proc_out/init_image_alignment/20241003_221111_ssc9d2_0012/run
# F=/bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/MCS/20241003/proc_out/init_image_alignment/20241003_221111_ssc9d2_0012/run-F.tif

# image_align $IMG1 $IMG2 \
# --output-image $OUTPUT_IMAGE \
# --output-prefix $OUTPUT_PREFIX \
# --alignment-transform rigid \
# --ecef-transform-type rigid \
# --dem1 $IMG1 \
# --dem2 $IMG2 \
# --disparity-params "$F 1000000"

# IMG=/bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/MCS/20240420/proc_out/init_rpc_ortho/20240420_165822_ssc16d3_0012_basic_panchromatic_map.tif
# CAM=/bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/MCS/20240420/proc_out/init_rpc_ortho/20240420_165822_ssc16d3_0012_basic_panchromatic_map_RPC.TXT
# PITCH=0.8
# FL=553846.153846
# CX=1280
# CY=540

# cam_gen $IMG \
# --input-camera $CAM \
# --focal-length $FL \
# --pixel-pitch $PITCH \
# -o /bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/MCS/20240420/proc_out/init_rpc_ortho/20240420_165822_ssc16d3_0012_basic_panchromatic_map.tsai
