# `skysat_stereo_snow` current to-do list

- Add option to filter point cloud by confidence before constructing DEM

- Multi-date camera alignment. First try resulted in wonky camera positions, so more testing is needed. 

- Ground Control Points (GCPs) and camera optimization: 
    - For iceberg images: Identify open water pixels, create GCP with elevations set to zero. Likely need to assign to each camera and make sure coordinates are appropriate (take care of pre- vs. post-alignment shifts). 

    - For snow images: Identify roads, use reference DEM to sample and use as GCP. Preliminary analysis: images with refined cameras from cam_gen were well aligned with the reference DEM, so coregistration (which could be a headache) may not be necessary before sampling. 