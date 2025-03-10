import os
import glob
import sys
import shutil
import subprocess
import pandas as pd
from tqdm import tqdm
from p_tqdm import p_map
import xdem
import numpy as np
import multiprocessing

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


def align_individual_dems(dem_list, refdem_fn, out_dir, max_displacement=40, tr=0.5):
    """
    Coregisters a list of DEMs to a reference DEM. For DEMs without at least 10% coverage of the reference DEM,
    the function will create a median mosaic of the aligned DEMs and iteratively coregister remaining DEMs 
    until no additional DEMs have sufficient overlap.
    
    Parameters
    ----------  
    dem_list: list
        List of DEM file paths to be coregistered.
    refdem_fn: str
        File path to the reference DEM.
    out_dir: str
        Output directory for aligned DEMs.
    max_displacement: int
    tr: float
        Resolution of the output DEMs.
    
    Returns
    ----------
    None
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    # Define function to get percent coverage of the reference DEM for a given DEM
    def get_coverage(refdem, dem_fn):
        dem = xdem.DEM(dem_fn, load_data=True)
        ref = refdem.reproject(dem)
        npx = np.where((dem.data.mask == False) & (ref.data.mask == False), 1, 0).sum()
        return {"filename": dem_fn, "refdem_percent_coverage": npx / dem.data.size * 100}

    # Define function to align a DEM to the reference DEM
    def align_dem(dem_fn, refdem_fn, out_dir, max_displacement=40, threads=4, tr=0.5, output_prefix=None):
        if output_prefix is None:
            output_prefix = os.path.join(out_dir, os.path.split(os.path.dirname(dem_fn))[1] + '-run')
        align_out_fn = output_prefix + '-trans_source.tif'
        grid_out_fn = output_prefix + '-trans_source-DEM.tif'
        if not os.path.exists(align_out_fn):
            align_cmd = ['--max-displacement', str(max_displacement), '--threads', str(threads),
                         '--highest-accuracy', '--save-transformed-source-points', '-o', output_prefix,
                         refdem_fn, dem_fn]
            run_cmd('pc_align', align_cmd)
        if os.path.exists(align_out_fn) and not os.path.exists(grid_out_fn):
            grid_cmd = ['--tr', str(tr), align_out_fn]
            run_cmd('point2dem', grid_cmd)
    
    # Define function to create median mosaic of a list of DEMs
    def create_median_mosaic(dem_list, out_fn):
        mosaic_cmd = dem_list + ['--median', '--tr', str(tr), '-o', out_fn]
        run_cmd('dem_mosaic', mosaic_cmd)

    # Calculate percent coverage of the reference DEM for each DEM
    print('Calculating percent coverage of the reference DEM for each DEM')
    refdem = xdem.DEM(refdem_fn, load_data=True)
    coverage_data = p_map(lambda x: get_coverage(refdem, x), dem_list, num_cpus=int(multiprocessing.cpu_count()/2))
    coverage_df = pd.DataFrame(coverage_data)
    
    # ROUND 1: Coregister DEMs with at least 10% overlap to the reference DEM
    dem_align_list = coverage_df[coverage_df['refdem_percent_coverage'] >= 10]['filename'].tolist()
    if not dem_align_list:
        print("No DEMs with at least 10% coverage of the reference DEM.")
        return
    print(f'Number of DEMs with reference DEM coverage >= 10%: {len(dem_align_list)}')
    for dem_fn in tqdm(dem_align_list):
        align_dem(dem_fn, refdem_fn, out_dir, max_displacement=max_displacement, threads=multiprocessing.cpu_count())
    # Create median mosaic of aligned DEMs
    mosaic_fn = os.path.join(out_dir, 'aligned_dem_mosaic.tif')
    print(f'Creating median mosaic of {len(dem_align_list)} aligned DEMs')
    create_median_mosaic(dem_align_list, mosaic_fn)
    
    # ROUND 2+: Coregister the remaining DEMs to the median mosaic of the aligned DEMs
    remaining_dems = [x for x in dem_list if x not in dem_align_list]
    while remaining_dems:        
        # Check which remaining DEMs have at least 10% coverage of the new median mosaic
        mosaic = xdem.DEM(mosaic_fn, load_data=True)
        coverage_data = p_map(lambda x: get_coverage(mosaic, x), remaining_dems, num_cpus=int(multiprocessing.cpu_count()/2))
        coverage_df = pd.DataFrame(coverage_data)
        new_align_list = coverage_df[coverage_df['refdem_percent_coverage'] >= 10]['filename'].tolist()
        if not new_align_list:
            print("No additional DEMs with at least 10% coverage of the median mosaic. Stopping.")
            break
        print(f'Number of additional DEMs with median mosaic coverage >= 10%: {len(new_align_list)}')
        # Coregister the new DEMs
        for dem_fn in tqdm(new_align_list):
            align_dem(dem_fn, mosaic_fn, out_dir, max_displacement=max_displacement, threads=multiprocessing.cpu_count())
        # Update the list of aligned and remaining DEMs
        dem_align_list.extend(new_align_list)
        remaining_dems = [x for x in remaining_dems if x not in dem_align_list]

        # Create new median mosaic of all aligned DEMs
        final_mosaic_fn = os.path.join(out_dir, 'aligned_dem_mosaic.tif')
        print(f'Creating new median mosaic of {len(dem_align_list)} aligned DEMs')
        create_median_mosaic(dem_align_list, final_mosaic_fn)
    
    print("Coregistration process complete!")

    
out_dir = '/bsuhome/raineyaberle/scratch/SkySat-Stereo/study-sites/MCS/20240420/proc_out'
dem_files = sorted(glob.glob(os.path.join(out_dir, 'final_pinhole_stereo', '20*', '20*', '*-DEM.tif')))
refdem_fn = os.path.join(out_dir, 'refdem', 'MCS_refdem_lidar_COPDEM_merged_shpclip_trim_stable.tif')
align_dir = os.path.join(out_dir, 'aligned_dems')

align_individual_dems(dem_files, refdem_fn, align_dir)

# def align_individual_dems(dem_files, out_dir):
#     # Check if output directory exists
#     if not os.path.exists(out_dir):
#         os.mkdir(out_dir)
        
#     # Create a GeoDataFrame with DEM file names and bounding boxes
#     with rio.open(dem_files[0]) as src:
#         crs = f"EPSG:{src.crs.to_epsg()}"
#     def get_bbox(dem_path):
#         with rio.open(dem_path) as src:
#             return box(*src.bounds)
#     dem_gdf = gpd.GeoDataFrame({"filename": dem_files, "geometry": [get_bbox(f) for f in dem_files]}, crs=crs)
#     dem_gdf['area'] = [geom.area for geom in dem_gdf['geometry']]

#     # Initialize DEM sets
#     iteration = 1
#     remaining_dems = list(dem_files)
#     aligned_dems = []
    
#     # Identify the largest DEM
#     dem_gdf.sort_values(by='area', ascending=False, inplace=True)
#     dem_largest = dem_gdf.iloc[0]['filename']
    
#     # Set the largest DEM as the initial reference DEM
#     refdem = dem_largest
#     remaining_dems = [x for x in remaining_dems if x!=refdem]
#     aligned_dems.append(refdem)
    
#     # Initialize progress bar
#     pbar = tqdm(total=len(remaining_dems), desc="Coregistering DEMs", unit="DEM")
    
#     # Create a separate function to compute overlaps with new reference DEMs
#     def get_updated_overlap_df(refdem, dem_gdf):
#         # Get reference DEM bounding box
#         with rio.open(refdem) as src:
#             ref_bbox = box(*src.bounds)
#         # Recalculate overlap data for the current refdem
#         new_overlap_data = []
#         for i in range(len(dem_gdf)):
#             row = dem_gdf.iloc[i]
#             if ref_bbox.intersects(row.geometry):
#                 intersection = ref_bbox.intersection(row.geometry).area
#                 min_area = min(ref_bbox.area, row.geometry.area)
#                 percent_overlap = (intersection / min_area) * 100
#                 new_overlap_data.append({"dem1": refdem, "dem2": row.filename, "percent_overlap": percent_overlap})
#         return pd.DataFrame(new_overlap_data)

#     # Begin alignment in batches
#     while remaining_dems:
#         print(f"\nIteration {iteration}: Coregistering next batch of DEMs")        
        
#         # Identify DEMs overlapping the reference DEM
#         overlap_df = get_updated_overlap_df(refdem, dem_gdf)
#         if overlap_df.empty:
#             print("No overlapping DEMs found.")
#             break
        
#         # Identify DEMs to be aligned
#         overlap_df = overlap_df[overlap_df['dem2'].isin(remaining_dems)]
#         if overlap_df.empty:
#             print("No overlapping DEMs found.")
#             break
        
#         # Sort overlap df by percent overlap and select the top 3
#         overlap_df = overlap_df.sort_values(by='percent_overlap', ascending=False).head(3)
#         dem_align_list = overlap_df['dem2'].tolist()
#         if len(dem_align_list) < 1:
#             print("No more overlapping DEMs found.")
#             break
        
#         # Coregister the batch
#         output_prefix = os.path.join(out_dir, f'run{iteration}')
#         align_cmd = dem_align_list + [refdem] + ['--save-transformed-clouds', '-o', output_prefix]
#         print('n_align command:', align_cmd)
#         run_cmd('n_align', align_cmd)
        
#         # Update progress bar and DEM sets
#         pbar.update(len(dem_align_list))
#         aligned_dems.extend(dem_align_list)
#         remaining_dems = [x for x in remaining_dems if x not in dem_align_list]
        
#         # Create median mosaic of the aligned DEMs
#         mosaic_output = os.path.join(out_dir, f'run{iteration}_mosaic.tif')
#         mosaic_cmd = list(aligned_dems) + ['--median', '-o', mosaic_output]
#         print('dem_mosaic command:', mosaic_cmd)
#         run_cmd('dem_mosaic', mosaic_cmd)
        
#         # Update reference DEM to the new mosaic
#         refdem = mosaic_output
        
#         iteration += 1
    
#     pbar.close()
#     print("Coregistration complete!")
