#!/bin/bash
#SBATCH -J MCS_20241003         # job name
#SBATCH -o output_MCS_20241003.o%j  # output and error file name (%j expands to jobID)
#SBATCH -n 1              # total number of tasks requested
#SBATCH -c 48             # CPU cores per task
#SBATCH -N 1              # number of nodes you want to run on
#SBATCH -p bsudfq         # queue (partition)
#SBATCH -t 12:00:00       # run time (hh:mm:ss) - 12.0 hours in this example.

# Load the stereo pipeline
module load stereo_pipeline/3.5.0_alpha_2025-01-05

# Activate the environment
. ~/.bashrc
micromamba activate skysat_stereo_snow

# Define some variables here for convenience
base_dir="/bsuhome/raineyaberle/scratch/SkySat-Stereo"
site_name="MCS"
date="20241003"
coregdem="${base_dir}/study-sites/${site_name}/refdem/${site_name}_refdem_lidar_COPDEM_merged.tif"
#orthodem="${base_dir}/study-sites/${site_name}/refdem/MCS_COPDEM_buffer2km_WGS84_ellipsoid_UTM11_filled_coreg.tif"

# Run the triplet stereo pipeline
python ${base_dir}/skysat_stereo/scripts/skysat_triplet_pipeline.py \
-in_img "${base_dir}/study-sites/${site_name}/${date}/SkySatScene" \
-multispec "${base_dir}/study-sites/${site_name}/${date}/${site_name}_${date}_4band_mosaic.tif" \
-outfolder "${base_dir}/study-sites/${site_name}/${date}/coregStable_ba-u1m" \
-orthodem $coregdem \
-coregdem $coregdem \
-coreg_stable_only 1 \
-ba_dem 1 \
-ba_dem_uncertainty 1 \
-ba_cam_weight 0 \
-ndvi_threshold 0.4 \
-ndsi_threshold 0.4 \
-job_name "${site_name}_${date}" \
-full_workflow 1
#-partial_workflow_steps 5 6 7 8 9 10
