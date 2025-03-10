#!/bin/bash
#SBATCH -J MCS_20240420         # job name
#SBATCH -o output_MCS_20240420.o%j  # output and error file name (%j expands to jobID)
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
date="20240420"
orthodem="${base_dir}/study-sites/${site_name}/refdem/${site_name}_refdem_lidar_COPDEM_merged.tif"
coregdem="${base_dir}/study-sites/${site_name}/refdem/${site_name}_REFDEM_WGS84_ROADS-ONLY.tif"

# Run the triplet stereo pipeline
python ${base_dir}/skysat_stereo/scripts/skysat_triplet_pipeline.py \
-in_img "${base_dir}/study-sites/${site_name}/${date}/SkySatScene" \
-multispec "${base_dir}/study-sites/${site_name}/${date}/${site_name}_${date}_4band_mosaic.tif" \
-outfolder "${base_dir}/study-sites/${site_name}/${date}/coregRoads_ba-noDEM" \
-orthodem $orthodem \
-coregdem $coregdem \
-coreg_stable_only 0 \
-ba_dem 0 \
-ba_dem_uncertainty 5 \
-ba_cam_weight 0 \
-ndvi_threshold 0.4 \
-ndsi_threshold 0.4 \
-job_name "${site_name}_${date}" \
-full_workflow 1 
#-partial_workflow_steps 4 5 6 7 8 9 10
