import numpy as np
import os,glob
import subprocess

# define inputs
img_fol = '/bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/Banner/20240419-2/SkySatScene'
img_list = sorted(glob.glob(os.path.join(img_fol, '20*_panchromatic.tif')))
refdem = '/bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/Banner/refdem/Banner_refdem_lidar_COPDEM_merged.tif'

out_fol = '/bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/Banner/20240419-2/proc_out'
cam_fol = os.path.join(out_fol, 'camgen_cam_gcp')
cam_list = sorted(glob.glob(os.path.join(cam_fol, '*.tsai')))
overlap_list = os.path.join(out_fol, 'overlap_with_overlap_perc_stereo_only.txt')
ba_prefix = os.path.join(out_fol, 'ba_testing', 'run')

# identify three captures and three overlapping captures
img_time_identifier_list = np.array([os.path.basename(img).split('_')[1] for img in img_list])
img_time_unique_list = np.unique(img_time_identifier_list)
second_collection_list = np.where(img_time_identifier_list == img_time_unique_list[1])[0][[0,1,2]]
fix_cam_idx = np.array([0,1,2]+list(second_collection_list))

#####
# ROUND 1: adjust 6 frames only, serving as a kind of GCP
#####
# Save fixed image file and camera txt files
fix_img_list = [img_list[i] for i in fix_cam_idx]
fix_img_txt = 'round1_images.txt'
with open(fix_img_txt, 'w') as file:
  for img in fix_img_list:
    file.write(img + '\n')


fix_cam_list = [cam for cam in cam_list if os.path.join(img_fol, os.path.basename(cam.replace('_rpc.tsai','.tif'))) in fix_img_list]
fix_cam_txt = 'round1_cameras.txt'
with open(fix_cam_txt, 'w') as file:
  for cam in fix_cam_list:
    file.write(cam + '\n')


# run bundle adjust
cmd = ['parallel_bundle_adjust',
'--image-list', fix_img_txt,
'--camera-list', fix_cam_txt,
'--overlap-list', overlap_list,
'--heights-from-dem', refdem,
'--heights-from-dem-uncertainty', '5',
'--force-reuse-match-files',
'--threads', '48',
'--num-iterations', '700',
'--num-passes', '2',
'--remove-outliers-params', "75 3 20 20",
'--camera-position-weight', '0',
'--min-matches', '4',
'--ip-per-tile', '4000',
'--ip-inlier-factor', '0.2',
'--individually-normalize',
'--inline-adjustments',
'--save-cnet-as-csv',
'-o', ba_prefix
]

out = subprocess.run(cmd, shell=False, capture_output=True)


#####
# ROUND 2: adjust remaining frame, fixing the adjusted frames
#####
# write image list
all_img_txt = os.path.join(os.path.dirname(ba_prefix), 'round2_images.txt')
with open(all_img_txt, 'w') as file:
    for img in img_list:
        file.write(img + '\n')


# write camera list
cam_list_adj = []
for img in img_list:
    if img in fix_img_list:
        cam = ba_prefix + '-' + os.path.splitext(os.path.basename(img))[0] + '_rpc.tsai'
        print(cam)
    else:
        cam = [cam for cam in cam_list if os.path.join(img_fol, os.path.basename(cam)).replace('_rpc.tsai','.tif')==img][0]
    cam_list_adj.append(cam)


cam_txt_adj = os.path.join(os.path.dirname(ba_prefix), 'round2_cameras.txt')
with open(cam_txt_adj, 'w') as file:
    for cam in cam_list_adj:
        file.write(cam + '\n')


# run bundle adjust
cmd = ['parallel_bundle_adjust',
'--image-list', all_img_txt,
'--camera-list', cam_txt_adj,
'--overlap-list', overlap_list,
'--heights-from-dem', refdem,
'--heights-from-dem-uncertainty', '5',
'--fixed-camera-indices', ' '.join(fix_cam_idx.astype(str)),
'--force-reuse-match-files',
'--threads', '48',
'--num-iterations', '700',
'--num-passes', '2',
'--remove-outliers-params', "75 3 20 20",
'--camera-position-weight', '0',
'--min-matches', '4',
'--ip-per-tile', '4000',
'--ip-inlier-factor', '0.2',
'--individually-normalize',
'--inline-adjustments',
'--save-cnet-as-csv',
'-o', ba_prefix + 'run'
]

out = subprocess.run(cmd, shell=False, capture_output=True)

parallel_bundle_adjust \
--image-list /bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/Banner/20240419-2/proc_out/ba_testing/round2_images.txt \
--camera-list /bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/Banner/20240419-2/proc_out/ba_testing/round2_cameras.txt \
--overlap-list /bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/Banner/20240419-2/proc_out/overlap_with_overlap_perc_stereo_only.txt \
--heights-from-dem /bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/Banner/refdem/Banner_refdem_lidar_COPDEM_merged.tif \
--heights-from-dem-uncertainty 5 \
--fixed-camera-indices "0 1 2 20 21 22" \
--force-reuse-match-files \
--threads 48 \
--num-iterations 700 \
--num-passes 2 \
--remove-outliers-params "75 3 20 20" \
 --camera-position-weight 0 \
 --min-matches 4 \
 --ip-per-tile 4000 \
 --ip-inlier-factor 0.2 \
 --individually-normalize \
 --inline-adjustments \
 --save-cnet-as-csv \
 -o /bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/Banner/20240419-2/proc_out/ba_testing/run
