# colmap_testing.bash

DATASET_PATH="/Users/rdcrlrka/Research/SkySat-Stereo/study-sites/MCS/20240420/colmap"
MODEL_SCRIPT="/Users/rdcrlrka/Research/SkySat-Stereo/skysat_stereo_snow/scripts/create_colmap_model.py"

colmap feature_extractor \
    --image_path $DATASET_PATH/images \
    --database_path $DATASET_PATH/database.db \
    --ImageReader.single_camera_per_folder 1

colmap rig_configurator \
    --database_path $DATASET_PATH/database.db \
    --rig_config_path $DATASET_PATH/rig_config.json

python $MODEL_SCRIPT

colmap exhaustive_matcher \
    --database_path $DATASET_PATH/database.db

# Triangulate points using the initial model and feature matches.
mkdir -p $DATASET_PATH/sparse_initial_triangulated

colmap point_triangulator \
    --database_path $DATASET_PATH/database.db \
    --image_path $DATASET_PATH/images \
    --input_path $DATASET_PATH/sparse_initial \
    --output_path $DATASET_PATH/sparse_initial_triangulated \
    --clear_points 1 \
    --Mapper.tri_create_max_angle_error 10 \
    --Mapper.tri_merge_max_reproj_error 10 \
    --Mapper.tri_min_angle 0.1
    
mkdir -p $DATASET_PATH/sparse_final_optimized

colmap bundle_adjuster \
    --input_path $DATASET_PATH/sparse_initial_triangulated \
    --output_path $DATASET_PATH/sparse_final_optimized \
    --BundleAdjustment.refine_focal_length=0 \
    --BundleAdjustment.refine_extra_params=0 \
    --BundleAdjustment.refine_principal_point=0
