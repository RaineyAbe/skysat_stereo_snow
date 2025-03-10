#!/bin/bash
#SBATCH -J MCS_20240420         # job name
#SBATCH -o output_MCS_20240420_default.o%j  # output and error file name (%j expands to jobID)
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
coregdem="${base_dir}/study-sites/${site_name}/refdem/${site_name}_refdem_lidar_COPDEM_merged.tif"

# Run the triplet stereo pipeline
python ${base_dir}/skysat_stereo/scripts/skysat_triplet_pipeline_default.py \
-in_img "${base_dir}/study-sites/${site_name}/${date}/SkySatScene" \
-outfolder "${base_dir}/study-sites/${site_name}/${date}/default" \
-orthodem $coregdem \
-coregdem $coregdem \
-job_name "${site_name}_${date}" \
-full_workflow 0 \
-partial_workflow_steps 7 8 9 10
