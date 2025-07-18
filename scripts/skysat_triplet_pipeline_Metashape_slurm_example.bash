#!/bin/bash
#SBATCH -J Saqqerleq               # job name
#SBATCH -o output_Saqqerleq.o%j    # output and error file name (%j expands to jobID)
#SBATCH -n 1                       # total number of tasks requested
#SBATCH -c 48                      # CPU cores per task
#SBATCH -N 1                       # number of nodes you want to run on
#SBATCH -p bsudfq                  # queue (partition)
#SBATCH -t 12:00:00                # run time (hh:mm:ss) - 12.0 hours in this example.

# Activate the environment
. ~/.bashrc
micromamba activate skysat_stereo_snow

# Define some variables here for convenience
base_dir="/bsuhome/raineyaberle/scratch/SkySat-Stereo"
site_name="Saqqerleq"

# Run the Metashape stereo pipeline
python ${base_dir}/skysat_stereo_snow/scripts/skysat_triplet_pipeline_Metashape.py \
-img_folder "${base_dir}/study-sites/${site_name}/SkySatScene_TOAR" \
-masks "water_check" \
-coregister 1 \
-refdem "ArcticDEM" \
-job_name "${site_name}_skysat_stereo_snow" \
-out_folder "${base_dir}/study-sites/${site_name}" \
-full_workflow 1
