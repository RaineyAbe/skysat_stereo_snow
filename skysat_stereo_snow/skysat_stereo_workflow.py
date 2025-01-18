#! /usr/bin/env python

import os,sys,glob,re,shutil
import numpy as np
import geopandas as gpd
import pandas as pd
from pygeotools.lib import iolib,malib
from tqdm import tqdm
from p_tqdm import p_umap, p_map
from skysat_stereo import skysat
from skysat_stereo import asp_utils as asp
from rpcm import geo
from skysat_stereo import misc_geospatial as misc
import subprocess
from shapely.geometry import Polygon
import itertools
from pyproj import Transformer
import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt
import matplotlib

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
    print(call)
    try:
        out = subprocess.run(call,check=True,capture_output=True,encoding='UTF-8').stdout
    except:
        out = "the command {} failed to run, see corresponding asp log".format(call)
    return out


def prepare_stereopair_list(img_folder,perc_overlap,out_fn,aoi_bbox=None,cross_track=False):
    """
    """ 
    geo_crs = 'EPSG:4326'
    # populate img list
    try:
        img_list = sorted(glob.glob(os.path.join(img_folder,'*.tif')))
        print("Number of images {}".format(len(img_list)))
    except:
        print ("No images found in the directory. Make sure they end with a .tif extension")
        sys.exit()
    out_shp = os.path.splitext(out_fn)[0]+'_bound.gpkg'
    n_proc = iolib.cpu_count()
    shp_list = p_map(skysat.skysat_footprint,img_list,num_cpus=2*n_proc)
    merged_shape = misc.shp_merger(shp_list)
    bbox = merged_shape.total_bounds
    merged_shape = misc.shp_merger(shp_list)
    bbox = merged_shape.total_bounds
    print (f'Bounding box lon_lat is:{bbox}')
    print (f'Bounding box lon_lat is:{bbox}')
    bound_poly = Polygon([[bbox[0],bbox[3]],[bbox[2],bbox[3]],[bbox[2],bbox[1]],[bbox[0],bbox[1]]])
    bound_shp = gpd.GeoDataFrame(index=[0],geometry=[bound_poly],crs=geo_crs)
    bound_centroid = bound_shp.centroid
    cx = bound_centroid.x.values[0]
    cy = bound_centroid.y.values[0]
    pad = np.ptp([bbox[3],bbox[1]])/6.0
    lat_1 = bbox[1]+pad
    lat_2 = bbox[3]-pad
    #local_ortho = '+proj=ortho +lat_0={} +lon_0={}'.format(cy,cx)
    local_aea = "+proj=aea +lat_1={} +lat_2={} +lat_0={} +lon_0={} +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs".format(lat_1,lat_2,cy,cx)
    print ('Local Equal Area coordinate system is : {} \n'.format(local_aea))
    print('Saving bound shapefile at {} \n'.format(out_shp))
    bound_shp.to_file(out_shp,driver='GPKG')
    
    # condition to check bbox_aoi
    if aoi_bbox is not None:
        bbox = gpd.read_file(aoi_bbox)
        mask = merged_shape.to_crs(bbox.crs).intersects(bbox)
        img_list = merged_shape[mask].img.values

    img_combinations = list(itertools.combinations(img_list,2))
    n_comb = len(img_combinations)
    perc_overlap = np.ones(n_comb,dtype=float)*perc_overlap
    proj = local_aea
    tv = p_map(skysat.frame_intsec, img_combinations, [proj]*n_comb, perc_overlap,num_cpus=4*n_proc)
    # result to this contains truth value (0 or 1, overlap percentage)
    truth_value = [tvs[0] for tvs in tv]
    overlap = [tvs[1] for tvs in tv]
    valid_list = list(itertools.compress(img_combinations,truth_value))
    overlap_perc_list = list(itertools.compress(overlap,truth_value))
    print('Number of valid combinations are {}, out of total {}  input images making total combinations {}\n'.format(len(valid_list),len(img_list),n_comb))
    with open(out_fn, 'w') as f:
        img1_list = [x[0] for x in valid_list]
        img2_list = [x[1] for x in valid_list]
        for idx,i in enumerate(valid_list):
            #f.write("%s %s\n" % i) 
            f.write(f"{os.path.abspath(img1_list[idx])} {os.path.abspath(img2_list[idx])}\n")
    out_fn_overlap = os.path.splitext(out_fn)[0]+'_with_overlap_perc.pkl'
    img1_list = [x[0] for x in valid_list]
    img2_list = [x[1] for x in valid_list]
    out_df = pd.DataFrame({'img1':img1_list,'img2':img2_list,'overlap_perc':overlap_perc_list})
    out_df.to_pickle(out_fn_overlap)
    
    out_fn_stereo = os.path.splitext(out_fn_overlap)[0]+'_stereo_only.pkl'
    stereo_only_df = skysat.prep_trip_df(out_fn_overlap,cross_track=cross_track)
    stereo_only_df.to_pickle(out_fn_stereo)
    out_fn_stereo_ba = os.path.splitext(out_fn_overlap)[0]+'_stereo_only.txt'
    stereo_only_df[['img1','img2']].to_csv(out_fn_stereo_ba,sep=' ',header=False,index=False)
    
    return stereo_only_df, out_df


def skysat_preprocess(img_folder,mode,sampling=None,frame_index=None,product_level='l1a',
        sampler=5,overlap_pkl=None,dem=None,outdir=None):
    """
    """
    if not os.path.exists(outdir):
        try:
            os.makedir(outdir)
        except:
            os.makedirs(outdir)
    if mode == 'video':
        frame_index = skysat.parse_frame_index(frame_index,True)
        product_level = 'l1a'
        num_samples = len(frame_index)
        frames = frame_index.name.values
        outdf = os.path.join(outdir,os.path.basename(frame_index))
        if sampling == 'sampling_interval':
            print("Hardcoded sampling interval results in frame exclusion at the end of the video sequence based on step size, better to chose the num_images mode and the program will equally distribute accordingly")
            idx = np.arange(0,num_samples,sampler)
            outdf = '{}_sampling_inteval_{}.csv'.format(os.path.splitext(outdf)[0],sampler)
        else:
            print("Sampling {} from {} of the input video sequence".format(sampler,num_samples))
            idx = np.linspace(0,num_samples-1,sampler,dtype=int)
            outdf = '{}_sampling_inteval_aprox{}.csv'.format(os.path.splitext(outdf)[0],idx[1]-idx[0])
        sub_sampled_frames = frames[idx]
        sub_df = frame_index[frame_index['name'].isin(list(sub_sampled_frames))]
        sub_df.to_csv(outdf,sep=',',index=False)
        #this is camera/gcp initialisation
        n = len(sub_sampled_frames)
        img_list = [glob.glob(os.path.join(img_folder,'{}*.tiff'.format(frame)))[0] for frame in sub_sampled_frames]
        pitch = [1]*n
        out_fn = [os.path.join(outdir,'{}_frame_idx.tsai'.format(frame)) for frame in sub_sampled_frames]
        out_gcp = [os.path.join(outdir,'{}_frame_idx.gcp'.format(frame)) for frame in sub_sampled_frames]
        frame_index = [frame_index]*n
        camera = [None]*n
        gcp_factor = 4
    elif mode == 'triplet':
        df = pd.read_pickle(overlap_pkl)
        img_list = list(np.unique(np.array(list(df.img1.values)+list(df.img2.values))))
        img_list = [os.path.splitext(os.path.basename(img))[0] for img in img_list]
        cam_list = [glob.glob(os.path.join(img_folder,'{}*.tif'.format(img)))[0] for img in img_list]
        n = len(img_list)
        if product_level == 'l1b':
            pitch = [0.8]*n
        else:
            pitch = [1.0]*n
        out_fn = [os.path.join(outdir,'{}_rpc.tsai'.format(frame)) for frame in img_list]
        out_gcp = [os.path.join(outdir,'{}_rpc.gcp'.format(frame)) for frame in img_list]
        camera = cam_list
        frame_index = [None]*n
        img_list = cam_list
        gcp_factor = 8
    fl = [553846.153846]*n
    cx = [1280]*n
    cy = [540]*n
    dem
    ht_datum = [np.nanmedian(iolib.fn_getma(dem).data)]*n # use this value for height where DEM has no-data
    gcp_std = [1]*n
    datum = ['WGS84']*n
    refdem = [dem]*n
    n_proc = iolib.cpu_count()
    if n_proc > 2:
        n_proc -= 2
    print("Starting camera resection procedure")
    cam_gen_log = p_map(asp.cam_gen,img_list,fl,cx,cy,pitch,ht_datum,gcp_std,out_fn,out_gcp,datum,refdem,camera,frame_index,num_cpus = n_proc)
    print("writing gcp with basename removed")
    # count expexted gcp 
    print(f"Total expected GCP {gcp_factor*n}")    
    asp.clean_gcp(out_gcp,outdir)
    
    
    return cam_gen_log   


def construct_land_cover_masks(multispec_mosaic_fn, out_dir, ndvi_threshold=0.5, ndsi_threshold=0.1, plot_results=True):
    # Define output files
    trees_mask_fn = os.path.join(out_dir, 'trees_mask.tif')
    snow_mask_fn = os.path.join(out_dir, 'snow_mask.tif')
    ss_mask_fn = os.path.join(out_dir, 'stable_surfaces_mask.tif')
    fig_fn = os.path.join(out_dir, 'land_cover_masks.png')

    # Function to load 4-band mosaic if needed
    mosaic = None
    def load_mosaic(multispec_mosaic_fn):
        mosaic_xr = rxr.open_rasterio(multispec_mosaic_fn).astype(float)
        crs = f'EPSG:{mosaic_xr.rio.crs.to_epsg()}'
        mosaic = xr.Dataset(coords={'y': mosaic_xr.y, 'x':mosaic_xr.x})
        bands = ['blue', 'green', 'red', 'NIR']
        for i, b in enumerate(bands):
            mosaic[b] = mosaic_xr.isel(band=i)
        mosaic = mosaic / 1e4
        mosaic = xr.where(mosaic==0, np.nan, mosaic)
        mosaic.rio.write_crs(crs, inplace=True)
        return mosaic, crs

    # Construct trees mask
    if not os.path.exists(trees_mask_fn):
        print('Constructing trees mask...')
        if not mosaic:
            mosaic, crs = load_mosaic(multispec_mosaic_fn)
        # Calculate NDVI
        ndvi = (mosaic.NIR - mosaic.red) / (mosaic.NIR + mosaic.red)
        # Apply threshold
        trees_mask = (ndvi >= ndvi_threshold).astype(int)
        # Save to file
        trees_mask = xr.where(np.isnan(mosaic.blue), -9999, trees_mask) # set no data values to -9999
        trees_mask = trees_mask.assign_attrs({'Description': 'Trees mask constructing by thresholding the NDVI of the 4-band mosaic image.',
                                            '_FillValue': -9999,
                                            'NDVI bands': 'NIR, green',
                                            'NDVI threshold': ndvi_threshold})
        trees_mask.rio.write_crs(crs, inplace=True)
        trees_mask.rio.to_raster(trees_mask_fn, dtype='int16')
        print('Trees mask saved to file:', trees_mask_fn)
    else:
        print('Trees mask exists in directory, skipping.')

    # Construct snow mask
    if not os.path.exists(snow_mask_fn):
        print('Constructing snow mask...')
        if not mosaic:
            mosaic, crs = load_mosaic(multispec_mosaic_fn)
        # Calculate NDSI
        ndsi = (mosaic.red - mosaic.NIR) / (mosaic.red + mosaic.NIR)
        # Apply threshold
        snow_mask = (ndsi >= ndsi_threshold).astype(int)
        # Save to file
        snow_mask = xr.where(np.isnan(mosaic.blue), -9999, snow_mask) # set no data values to -9999
        snow_mask = snow_mask.assign_attrs({'Description': 'Snow mask constructed by thresholding the NDSI of the orthomosaic image',
                                            '_FillValue': -9999,
                                            'NDSI bands': 'red, NIR',
                                            'NDSI threshold': ndsi_threshold})
        snow_mask.rio.write_crs(crs, inplace=True)
        snow_mask.rio.to_raster(snow_mask_fn, dtype='int16')
        print('Snow mask saved to file:', snow_mask_fn)
    else:
        print('Snow mask exists in directory, skipping.')

    # Construct stable surfaces (snow-free and tree-free) mask
    if not os.path.exists(ss_mask_fn):
        print('Constructing stable surfaces mask...')
        if not mosaic:
            mosaic, crs = load_mosaic(multispec_mosaic_fn)
        # Load trees and snow masks
        trees_mask = rxr.open_rasterio(trees_mask_fn).squeeze()
        snow_mask = rxr.open_rasterio(snow_mask_fn).squeeze()
        # Stable surfaces = snow-free and tree-free
        ss_mask = xr.where((trees_mask==0) & (snow_mask==0), 1, 0)
        # Save to file
        ss_mask = xr.where(np.isnan(mosaic.blue), -9999, ss_mask)
        ss_mask = ss_mask.assign_attrs({'Description': 'Stable surfaces (snow-free and tree-free) mask.',
                                        '_FillValue': -9999})
        ss_mask.rio.write_crs(crs, inplace=True)
        ss_mask.rio.to_raster(ss_mask_fn, dtype='int16')
        print('Stable surfaces mask saved to file:', ss_mask_fn)
    else: 
        print('Stable surfaces mask exists in directory, skipping.')

    # Plot land cover masks
    if plot_results & (not os.path.exists(fig_fn)):
        print('Plotting land cover masks...')
        if not mosaic:
            mosaic, crs = load_mosaic(multispec_mosaic_fn)
        # Load masks
        trees_mask = rxr.open_rasterio(trees_mask_fn).squeeze()
        snow_mask = rxr.open_rasterio(snow_mask_fn).squeeze()
        ss_mask = rxr.open_rasterio(ss_mask_fn).squeeze()
        # Define land cover colors
        colors_dict = {'trees': '#006d2c', 
                       'snow': '#4292c6', 
                       'stable_surfaces': '#bdbdbd'}
        # Plot
        fig, ax = plt.subplots(1, 2, figsize=(10,5))
        # RGB image
        ax[0].imshow(np.dstack([mosaic.red.data, mosaic.green.data, mosaic.blue.data]),
                     extent=(np.min(mosaic.x)/1e3, np.max(mosaic.x)/1e3, np.min(mosaic.y)/1e3, np.max(mosaic.y)/1e3))
        ax[0].set_title('RGB mosaic')
        ax[0].set_xlabel('Easting [km]')
        ax[0].set_ylabel('Northing [km]')
        # Land cover masks
        
        for mask, label in zip([trees_mask, snow_mask, ss_mask], list(colors_dict.keys())):
            cmap = matplotlib.colors.ListedColormap([(0,0,0,0), matplotlib.colors.to_rgb(colors_dict[label])])
            ax[1].imshow(mask, cmap=cmap, clim=(0,1),
                        extent=(np.min(mask.x)/1e3, np.max(mask.x)/1e3, np.min(mask.y)/1e3, np.max(mask.y)/1e3))
            # dummy point for legend
            ax[1].plot(0, 0, 's', color=colors_dict[label], markersize=10, label=label) 
        ax[1].set_xlim(ax[0].get_xlim())
        ax[1].set_ylim(ax[0].get_ylim())
        ax[1].legend(loc='best')
        ax[1].set_xlabel('Easting [km]')

        # Save figure
        fig.savefig(fig_fn, dpi=300, bbox_inches='tight')
        print('Figure saved to file:', fig_fn)
        plt.close()
    elif plot_results:
        print('Masks figure exists in directory, skipping plotting.')

    return trees_mask_fn, snow_mask_fn, ss_mask_fn

def prepare_reference_elevations(coreg_dem_fn, ortho_dem_fn, multispec_dir, bound_fn, bound_buffer_fn, out_dir, 
                                 ndvi_threshold=0.5, ndsi_threshold=0.4, coreg_stable_only=False):
    # Check if output directory exists
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Determine optimal UTM zone and clipping bounds
    print("Computing Target UTM zones for orthorectification")
    gdf = gpd.read_file(bound_fn)
    clon, clat = [gdf.centroid.x.values, gdf.centroid.y.values]
    epsg_code = f'EPSG:{geo.compute_epsg(clon, clat)}'
    print(f"Detected UTM zone is {epsg_code}")
    if not os.path.exists(bound_buffer_fn):
        print("Creating buffered shapefile")
        gdf_proj = gdf.to_crs(epsg_code)
        gdf_proj['geometry'] = gdf_proj.buffer(1000)
        gdf_proj.to_file(bound_buffer_fn, driver='GPKG')
    
    # Check if multispec_dir is a directory of images or a single file
    if os.path.isdir(multispec_dir):
        multispec_mosaic_fn = os.path.join(out_dir, '4band_mosaic.tif')
        # Check if mosaic exists
        if not os.path.exists(multispec_mosaic_fn):
            print('Mosacking 4-band SR images...')
            # Grab all 4-band SR image file names
            multispec_fns = sorted(glob.glob(os.path.join(multispec_dir, '*_SR.tif')))
            # Construct gdal_merge arguments
            merge_args = multispec_fns + ['--ot', 'Int16', '--no-bigtiff', '-o', multispec_mosaic_fn]
            # Run command
            run_cmd('image_mosaic', merge_args)
        else:
            print('4-band mosaic already exists, skipping gdal_merge.')
    elif os.path.isfile(multispec_dir):
        multispec_mosaic_fn = multispec_dir
    
    # Construct stable surfaces mask
    trees_mask_fn, snow_mask_fn, ss_mask_fn = construct_land_cover_masks(multispec_mosaic_fn, out_dir, ndvi_threshold, ndsi_threshold, plot_results=True)

    # Trim DEMs to SkySat footprint + 1km buffer to speed up computations
    def trim_dem(dem_fn, bound_fn, out_dir):
        # copy DEM to output directory
        shutil.copy2(dem_fn, out_dir)
        dem_fn = os.path.join(out_dir, os.path.basename(dem_fn))
        # Trim DEM to SkySat footprint + 1 km buffer
        dem_trim_fn = os.path.join(out_dir, os.path.splitext(os.path.basename(dem_fn))[0] + '_shpclip_trim.tif')
        if not os.path.exists(dem_trim_fn):
            print("Cropping reference DEM to extent of SkySat footprint + 1 km buffer")
            misc.clip_raster_by_shp_disk(dem_fn, bound_fn)
            misc.ndvtrim_function(os.path.splitext(dem_fn)[0] + '_shpclip.tif')
            # Use the median DEM value where there's no coverage in the images footprint
            # dem_trim = rxr.open_rasterio(dem_trim_fn).squeeze()
            # crs = dem_trim.rio.crs
            # dem_trim = xr.where(dem_trim <= 0, np.nan, dem_trim)
            # dem_trim_median = np.nanmedian(dem_trim.data)
            # dem_trim = xr.where(np.isnan(dem_trim), dem_trim_median, dem_trim)
            # dem_trim = dem_trim.rio.write_crs(crs)
            # dem_trim.rio.to_raster(dem_trim_fn)
        return dem_trim_fn
    
    coreg_dem_fn = trim_dem(coreg_dem_fn, bound_buffer_fn, out_dir)
    if coreg_dem_fn != ortho_dem_fn:
        ortho_dem_fn = trim_dem(ortho_dem_fn, bound_buffer_fn, out_dir)
    
    # Mask coreg DEM to stable surfaces
    if coreg_stable_only:
        def mask_coreg_dem(dem_fn, ss_mask_fn, out_dir):
            dem_stable_fn = os.path.join(out_dir, os.path.splitext(os.path.basename(dem_fn))[0] + '_stable.tif')
            if not os.path.exists(dem_stable_fn):
                print("Masking coreg DEM to stable surfaces")
                dem = rxr.open_rasterio(dem_fn).squeeze()
                ss_mask = rxr.open_rasterio(ss_mask_fn).squeeze().rio.reproject_match(dem)
                dem_masked = xr.where(ss_mask==1, dem, np.nan)
                dem_masked.rio.write_crs(dem.rio.crs, inplace=True)
                dem_masked.rio.to_raster(dem_stable_fn, dtype='float32', nodata=np.nan)
                print('Stable surfaces DEM saved to file:', dem_stable_fn)
            return dem_stable_fn
        coreg_dem_fn = mask_coreg_dem(coreg_dem_fn, ss_mask_fn, out_dir)
    
    return coreg_dem_fn, ortho_dem_fn, epsg_code
    

def execute_skysat_orthorectification(images,outdir,data='triplet',dem='WGS84',tr=None,tsrs=None,del_opt=False,cam_folder=None,ba_prefix=None,
    mode='science',session=None,overlap_list=None,frame_index_fn=None,copy_rpc=1,orthomosaic=0):
    """
    """
    if mode == 'browse':
        """
        this block creates low-res orthomosaics from RPC info for browsing purpose only
        """
        for_img_list,nadir_img_list,aft_img_list,for_time,nadir_time,aft_time = skysat.sort_img_list(images)
        for_out_dir = os.path.join(outdir,'for_map_browse')
        nadir_out_dir = os.path.join(outdir,'nadir_map_browse')
        aft_out_dir = os.path.join(outdir,'aft_map_browse')
        for_out_list = [os.path.join(for_out_dir,os.path.splitext(os.path.basename(img))[0]+'_browse_map.tif') for img in for_img_list]
        nadir_out_list = [os.path.join(nadir_out_dir,os.path.splitext(os.path.basename(img))[0]+'_browse_map.tif') for img in nadir_img_list]
        aft_out_list = [os.path.join(aft_out_dir,os.path.splitext(os.path.basename(img))[0]+'_browse_map.tif') for img in aft_img_list]
        for_count,nadir_count,aft_count = [len(for_img_list), len(nadir_img_list), len(aft_img_list)]
        print("Performing orthorectification for forward images {}".format(for_time))
        for_map_log = p_map(asp.mapproject,for_img_list,for_out_list,[session]*for_count,['WGS84']*for_count,[None]*for_count,
            ['EPSG:4326']*for_count,[None]*for_count,[None]*for_count,[None]*for_count)
        print("Performing orthorectification for nadir images {}".format(nadir_time))
        nadir_map_log = p_map(asp.mapproject,nadir_img_list,nadir_out_list,[session]*nadir_count,['WGS84']*nadir_count,[None]*nadir_count,
            ['EPSG:4326']*nadir_count,[None]*nadir_count,[None]*nadir_count,[None]*nadir_count)
        print("Performing orthorectification for aft images {}".format(aft_time))
        aft_map_log = p_map(asp.mapproject,aft_img_list,aft_out_list,[session]*aft_count,['WGS84']*aft_count,[None]*aft_count,
            ['EPSG:4326']*aft_count,[None]*aft_count,[None]*aft_count,[None]*aft_count)
        ortho_log = os.path.join(outdir,'low_res_ortho.log')
        print("Orthorectification log saved at {}".format(ortho_log))
        with open(ortho_log,'w') as f:
            total_ortho_log = for_map_log+nadir_map_log+aft_map_log
            for log in itertools.chain.from_iterable(total_ortho_log):
                f.write(log)

        # after orthorectification, now do mosaic
        for_out_mos = os.path.join(outdir,'for_map_mos_{}m.tif'.format(tr))
        for_map_list = sorted(glob.glob(os.path.join(for_out_dir,'*.tif')))
        nadir_out_mos = os.path.join(outdir,'nadir_map_mos_{}m.tif'.format(tr))
        nadir_map_list = sorted(glob.glob(os.path.join(nadir_out_dir,'*.tif')))
        aft_out_mos = os.path.join(outdir,'aft_map_mos_{}m.tif'.format(tr))
        aft_map_list = sorted(glob.glob(os.path.join(aft_out_dir,'*.tif')))
        print("Preparing forward browse orthomosaic")
        for_mos_log = asp.dem_mosaic(for_map_list,for_out_mos,tr,tsrs,stats='first',tile_size=None)
        print("Preparing nadir browse orthomosaic")
        nadir_mos_log = asp.dem_mosaic(nadir_map_list, nadir_out_mos, tr, tsrs,stats='first',tile_size=None)
        print("Preparing aft browse orthomosaic")
        aft_mos_log = asp.dem_mosaic(aft_map_list, aft_out_mos, tr, tsrs,stats='first',tile_size=None)
        ## delete temporary files
        if del_opt:
            [shutil.rmtree(x) for x in [for_out_dir,nadir_out_dir,aft_out_dir]]
        #Save figure to jpeg ?
        fig_title = os.path.basename(images[0]).split('_',15)[0]+'_'+for_time+'_'+nadir_time+'_'+aft_time
        fig,ax = plt.subplots(1,3,figsize=(10,10))
        # pltlib.iv_fn(for_out_mos,full=True,ax=ax[0],cmap='gray',scalebar=True,title='Forward')
        # pltlib.iv_fn(nadir_out_mos,full=True,ax=ax[1],cmap='gray',scalebar=True,title='NADIR')
        # pltlib.iv_fn(aft_out_mos,full=True,ax=ax[2],cmap='gray',scalebar=True,title='Aft')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.suptitle(fig_title)
        browse_img_fn = os.path.join(outdir,'browse_img_{}_{}m.jpg'.format(fig_title,tr))
        fig.savefig(browse_img_fn,dpi=300,bbox_inches='tight',pad_inches=0.1)
        print("Browse figure saved at {}".format(browse_img_fn))
        
    if mode == 'science':
        img_list = images
        if overlap_list is not None:
            # need to remove images and cameras which are not optimised during bundle adjustment
            # read pairs from input overlap list
            initial_count = len(img_list)
            with open(overlap_list) as f:
                content = f.readlines()
            content = [x.strip() for x in content]
            l_img = [x.split(' ')[0] for x in content]
            r_img = [x.split(' ')[1] for x in content]
            total_img = l_img + r_img
            uniq_idx = np.unique(total_img, return_index=True)[1]
            img_list = [total_img[idx] for idx in sorted(uniq_idx)]
            # Filter to images with cameras
            if ba_prefix is not None:
                img_list_new = []
                for img_fn in img_list:
                    cam_fn = glob.glob(os.path.abspath(ba_prefix)+'-'+os.path.splitext(os.path.basename(img_fn))[0]+'*.tsai')
                    if len(cam_fn) > 0:
                        img_list_new.append(img_fn)
                img_list = img_list_new
            print(f"Out of the initial {initial_count} images, {len(img_list)} will be orthorectified using adjusted cameras")

        if frame_index_fn is not None:
            frame_index = skysat.parse_frame_index(frame_index_fn)
            img_list = [glob.glob(os.path.join(dir,'{}*.tiff'.format(frame)))[0] for frame in frame_index.name.values]
            print("no of images is {}".format(len(img_list)))
        img_prefix = [os.path.splitext(os.path.basename(img))[0] for img in img_list]
        out_list = [os.path.join(outdir,img+'_map.tif') for img in img_prefix]
        session_list = [session]*len(img_list)
        dem_list = [dem]*len(img_list)
        # dem_list = ['WGS84']*len(img_list)
        tr_list = [tr]*len(img_list)
        if frame_index_fn is not None:
            # this hack is for video
            df = skysat.parse_frame_index(frame_index_fn)
            trunc_df = df[df['name'].isin(img_prefix)]
            tr_list = [str(gsd) for gsd in trunc_df.gsd.values]
        srs_list = [tsrs]*len(img_list)

        if session == 'pinhole':
            if ba_prefix:
                cam_list = [glob.glob(os.path.abspath(ba_prefix)+'-'+os.path.splitext(os.path.basename(x))[0]+'*.tsai')[0] for x in img_list]
                print("No of cameras is {}".format(len(cam_list)))
            else:
                print(os.path.join(os.path.abspath(cam_folder),os.path.splitext(os.path.basename(img_list[0]))[0]+'*.tsai'))
                cam_list = [glob.glob(os.path.join(os.path.abspath(cam_folder),os.path.splitext(os.path.basename(x))[0]+'*.tsai'))[0] for x in img_list]
        else:
            cam_list = [None]*len(img_list)
            if ba_prefix:
                # not yet implemented
                ba_prefix_list = [ba_prefix]*len(img_list)

        print("Mapping given images")
        ortho_logs = p_map(asp.mapproject,img_list,out_list,session_list,dem_list,tr_list,srs_list,cam_list,
            [None]*len(img_list),[None]*len(img_list),num_cpus=int(iolib.cpu_count()/4))
        ortho_log = os.path.join(outdir,'ortho.log')
        print("Saving Orthorectification log at {}".format(ortho_log))
        with open(ortho_log,'w') as f:
            for log in ortho_logs:
                f.write(log)
        if copy_rpc == 1:
            print("Copying RPC from native image to orthoimage in parallel")
            try:
                copy_rpc_out = p_map(skysat.copy_rpc,img_list,out_list,num_cpus=iolib.cpu_count())
            except Exception as e:
                print(e)
        if orthomosaic == 1:
            print("Will also produce median, weighted average and highest resolution orthomosaic")
            if data == 'triplet':
                # sort images based on timestamps and resolutions
                out_list = glob.glob(os.path.join(outdir, '*_map.tif'))
                img_list, time_list = skysat.sort_img_list(out_list)
                res_sorted_list = skysat.res_sort(out_list)

                # define mosaic prefix containing timestamps of inputs
                mos_prefix = '_'.join(np.unique([t.split('_')[0] for t in time_list]))+'__'+'_'.join(np.unique([t.split('_')[1] for t in time_list]))

                # define output filenames
                res_sorted_mosaic = os.path.join(outdir,'{}_finest_orthomosaic.tif'.format(mos_prefix))
                median_mosaic = os.path.join(outdir,'{}_median_orthomosaic.tif'.format(mos_prefix))
                wt_avg_mosaic = os.path.join(outdir,'{}_wt_avg_orthomosaic.tif'.format(mos_prefix))
                indi_mos_list = [os.path.join(outdir,f'{time}_first_orthomosaic.tif') for time in time_list]


                print("producing finest resolution on top mosaic, per-pixel median and wt_avg mosaic")
                all_3_view_mos_logs = p_map(asp.dem_mosaic, [res_sorted_list]*3, [res_sorted_mosaic,median_mosaic,wt_avg_mosaic], 
                                          ['None']*3, [None]*3, ['first','median',None],[None]*3,num_cpus=4)

                print("producing idependent mosaic for different views in parallel")
                indi_mos_count = len(time_list)
                if indi_mos_count>3:
                    tile_size = 400
                else:
                    tile_size = None

                indi_mos_log = p_map(asp.dem_mosaic,img_list, indi_mos_list, ['None']*indi_mos_count, [None]*indi_mos_count, 
                    ['first']*indi_mos_count,[tile_size]*indi_mos_count)

                # write out log files
                out_log = os.path.join(outdir,'science_mode_ortho_mos.log')
                total_mos_log = all_3_view_mos_logs+indi_mos_log
                print("Saving orthomosaic log at {}".format(out_log))
                with open(out_log,'w') as f:
                    for log in itertools.chain.from_iterable(total_mos_log):
                        f.write(log)
            if data == 'video':
                res_sorted_list = skysat.res_sort(out_list)
                print("producing orthomasaic with finest on top")
                res_sorted_mosaic = os.path.join(outdir,'video_finest_orthomosaic.tif')
                print("producing orthomasaic with per-pixel median stats")
                median_mosaic = os.path.join(outdir,'video_median_orthomosaic.tif')
                print("producing orthomosaic with weighted average statistics")
                wt_avg_mosaic = os.path.join(outdir,'video_wt_avg_orthomosaic.tif')
                print("Mosaicing will be done in parallel")
                all_3_view_mos_logs = p_map(asp.dem_mosaic, [res_sorted_list]*3, [res_sorted_mosaic,median_mosaic,wt_avg_mosaic], ['None']*3, [None]*3, ['first','median',None],[None]*3)
                out_log = os.path.join(outdir,'science_mode_ortho_mos.log')
                print("Saving orthomosaic log at {}".format(out_log))
                with open(out_log,'w') as f:
                    for log in all_3_view_mos_logs:
                        f.write(log)

def execute_skysat_stereo(img,outfol,mode,session='rpc',dem=None,texture='high',
    sampling_interval=None,cam_folder=None,ba_prefix=None,writeout_only=False,mvs=0,block=1,crop_map=0,
    full_extent=1,entry_point='pprc',threads=2,overlap_pkl=None,frame_index=None,job_fn=None,cross_track=False):
    """
    """
    img = os.path.abspath(img)
    try:
        img_list = sorted(glob.glob(os.path.join(img, '*.tif')))
        temp = img_list[1]
    except BaseException:
        img_list = sorted(glob.glob(os.path.join(img, '*.tiff')))
    if len(img_list) == 0:
        print("No images in the specified folder, exiting")
        sys.exit()

    if mode == 'video':
        # assume for now that we are still operating on a fixed image interval method
        # can accomodate different convergence angle function method here.
        frame_gdf = skysat.parse_frame_index(frame_index)
        # for now hardcording sgm,mgm,kernel params, should accept as inputs.
        # Maybe discuss with David with these issues/decisions when the overall
        # system is in place
        if mvs == 1:
            job_list = skysat.video_mvs(img,t=session,cam_fol=cam_folder,ba_prefix=ba_prefix,dem=dem,
                           sampling_interval=sampling_interval,texture=texture,
                           outfol=outfol,block=block,frame_index=frame_gdf)

        else:
            if full_extent == 1:
                full_extent = True
            else:
                full_extent = False
            job_list = skysat.prep_video_stereo_jobs(img,t=session,cam_fol=cam_folder,ba_prefix=ba_prefix,
                dem=dem,sampling_interval=sampling_interval,texture=texture,outfol=outfol,block=block,
                frame_index=frame_gdf,full_extent=full_extent,entry_point=entry_point)
    elif mode == 'triplet':
        if crop_map == 1:
            crop_map = True
        else: 
            crop_map = False
            
        job_list = skysat.triplet_stereo_job_list(cross_track=cross_track,t=session,
            threads = threads,overlap_list=overlap_pkl, img_list=img_list, ba_prefix=ba_prefix, 
            cam_fol=cam_folder, dem=dem, crop_map=crop_map,texture=texture, outfol=outfol, block=block,
            entry_point=entry_point)
        print(job_list[0])
    if not writeout_only:
        # decide on number of processes
        # if block matching, Plieades is able to handle 30-40 4 threaded jobs on bro node
        # if MGM/SGM, 25 . This stepup is arbitrariry, research on it more.
        # next build should accept no of jobs and stereo threads as inputs
    
        print(job_list[0])
        n_cpu = iolib.cpu_count()
        # no of parallel jobs with user specified threads per job
        #jobs = int(n_cpu/threads)
        # this seems to break with new paralle_stereo setup
        # setting hardcoded value of 20 for now
        # jobs = 20
        stereo_log = p_map(asp.run_cmd,['parallel_stereo']*len(job_list), job_list, num_cpus=n_cpu)
        print(stereo_log)
        stereo_log_fn = os.path.join(outfol,'stereo_log.log')      
        print("Consolidated stereo log saved at {}".format(stereo_log_fn))
    else:
        print(f"Writing jobs at {job_fn}")
        with open(job_fn,'w') as f:
            for idx,job in enumerate(tqdm(job_list)):
                try:                
                    job_str = 'stereo ' + ' '.join(job) + '\n'
                    f.write(job_str)
                except:
                    continue


def add_cam_intrinsics(input_file, output_file, distortion_model='TSAI',
                       intrinsics={'k1': -1e-12, 'k2': -1e-12, 'k3': -1e-12, 'p1': -1e-12, 'p2': -1e-12}):
    # Read the .tsai file
    with open(input_file, 'r') as file:
        lines = file.readlines()
    updated_lines = lines
    # Remove NULL line
    updated_lines = [line for line in updated_lines if line!='NULL\n']
    # Add the distortion model
    if f"{distortion_model}\n" not in updated_lines:
        updated_lines.append(f"{distortion_model}\n")
    # Add distortion parameters
    for key in intrinsics.keys():
        addition = f"{key} = {intrinsics[key]:.8f}\n"
        if addition not in updated_lines:
            updated_lines.append(addition)
    # Save the updated file
    with open(output_file, 'w') as file:
        file.writelines(updated_lines)
    # print('Saved updated camera file:', output_file)


def bundle_adjustment(img_folder, ba_prefix, cam, overlap_list=None, 
                      dem=None, dem_in_ba=False, dem_uncertainty=10, num_iter=1000, num_pass=2):
    # Create output directory
    out_dir = os.path.dirname(ba_prefix)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    # Grab list of images
    img_list = sorted(glob.glob(os.path.join(img_folder,'*.tif')) + glob.glob(os.path.join(img_folder, '*.tiff')))
    if len(img_list) < 2:
        if os.path.islink(img_list[0]):
            img_list = [os.readlink(x) for x in img_list] 
    
    # Filter to images with cameras (in case any failed in previous step)
    cam_list = sorted(glob.glob(os.path.join(cam, '*.tsai')))
    img_list_new = [os.path.join(img_folder, os.path.basename(cam_fn).replace('_rpc.tsai', '.tif')) for cam_fn in cam_list]
    print(f"Out of the initial {len(img_list)} images, {len(img_list_new)} will be orthorectified using adjusted cameras")
    img_list = img_list_new
    # Check if there are enough images
    if len(img_list) < 2:
        raise Exception('Not enough images with cameras found')
    
    ## ROUND 1: fix first 3 cameras
    # the concept is simple: first 3 cameras, and then corresponding first three cameras from next collection are fixed in the first go.
    # these serve as a kind of #GCP, preventing a large drift in the triangulated points/camera extrinsics during optimization
    img_time_identifier_list = np.array([os.path.basename(img).split('_')[1] for img in img_list])
    img_time_unique_list = np.unique(img_time_identifier_list)
    second_collection_list = np.where(img_time_identifier_list == img_time_unique_list[1])[0][[0,1,2]]
    fix_cam_idx = np.array([0,1,2]+list(second_collection_list))    
    fix_cam_idx_str = ' '.join(fix_cam_idx.astype(str))
    # Define bundle adjustment arguments
    ba_args = ['--fixed-camera-indices', fix_cam_idx_str,
               '--overlap-list', overlap_list,
               '--threads', str(iolib.cpu_count()),
               '--num-iterations', str(num_iter),
               '--num-passes', str(num_pass),
               '--camera-position-weight', '0',
               '--remove-outliers-params', "75 3 20 20",
               '--min-matches', '4',
               '--ip-per-tile', '4000',
               '--ip-inlier-factor', '0.2',
               '--individually-normalize',
               '--inline-adjustments',
               '--min-triangulation-angle', '0.0001',
               '--save-cnet-as-csv',
               '-o', ba_prefix] 
    if dem_in_ba:
        ba_args.extend(['--heights-from-dem', dem,
                        '--heights-from-dem-uncertainty', str(dem_uncertainty)])
    ba_args += img_list + cam_list
    print('ROUND 1')
    # print(ba_args)
    run_cmd('bundle_adjust', ba_args)
    
    ### ROUND 2: refine with DEM, (optional) solve intrinsics
    identifier = os.path.basename(cam_list[0]).split('_',14)[0][:2]
    print(ba_prefix+'-{}*.tsai'.format(identifier))
    cam_list = sorted(glob.glob(os.path.join(ba_prefix+ '-{}*.tsai'.format(identifier))))
    fix_cam_idx2 = np.delete(np.arange(len(img_list),dtype=int),fix_cam_idx)
    fix_cam_idx2_str = ' '.join(fix_cam_idx2.astype(str))
    # Define bundle adjustment arguments
    ba_args = ['--fixed-camera-indices', fix_cam_idx2_str,
               '--force-reuse-match-files',
               '--overlap-list', overlap_list,
               '--threads', str(iolib.cpu_count()),
               '--num-iterations', str(num_iter),
               '--num-passes', str(num_pass),
               '--camera-position-weight', '0',
               '--remove-outliers-params', "75 3 20 20",
               '--min-matches', '4',
               '--ip-per-tile', '4000',
               '--ip-inlier-factor', '0.2',
               '--individually-normalize',
               '--inline-adjustments',
               '--min-triangulation-angle', '0.0001',
               '--save-cnet-as-csv',
               '-o', ba_prefix]
    if dem_in_ba:
        ba_args.extend(['--heights-from-dem', dem,
                        '--heights-from-dem-uncertainty', str(dem_uncertainty)])
    ba_args += img_list + cam_list
    print('ROUND 2')
    # print(ba_args)
    run_cmd('bundle_adjust', ba_args)


# def bundle_adjustment(img_folder, ba_prefix, cam, overlap_list=None, 
#                       dem=None, num_iter=1000, num_pass=2, float_intrinsics=True):
#     # Create output directory
#     out_dir = os.path.dirname(ba_prefix)
#     if not os.path.exists(out_dir):
#         os.mkdir(out_dir)
    
#     # Grab list of images
#     img_list = sorted(glob.glob(os.path.join(img_folder,'*.tif')) + glob.glob(os.path.join(img_folder, '*.tiff')))
#     if len(img_list) < 2:
#         if os.path.islink(img_list[0]):
#             img_list = [os.readlink(x) for x in img_list] 
    
#     # Filter to images with cameras (some may have failed in previous step)
#     cam_list = sorted(glob.glob(os.path.join(cam, '*.tsai')))
#     img_list_new = [os.path.join(img_folder, os.path.basename(cam_fn).replace('_rpc.tsai', '.tif')) for cam_fn in cam_list]
#     print(f"Out of the initial {len(img_list)} images, {len(img_list_new)} will be orthorectified using adjusted cameras")
#     img_list = img_list_new
#     # Check if there are enough images
#     if len(img_list) < 2:
#         raise Exception('Not enough images with cameras found')
    
#     ## ROUND 1: Solve extrinsics, fixed intrinsics
#     # Sort image and camera lists by camera number to align with order in txt files later
#     img_list.sort(key=lambda x: os.path.basename(x).split('_')[2])
#     cam_list.sort(key=lambda x: os.path.basename(x).split('_')[2])
#     # Write images file names to txt file
#     img_txt_fn1 = os.path.join(out_dir, f'round1_images.txt')
#     with open(img_txt_fn1, 'w') as file:
#         for image_fn in img_list:
#             file.write(f'{image_fn}\n')
#     # Write camera file names to text file
#     cam_txt_fn1 = os.path.join(out_dir, f'round1_cameras.txt')
#     with open(cam_txt_fn1, 'w') as file:
#         for cam_fn in cam_list:
#             file.write(f'{cam_fn}\n')
        
#     # Define bundle adjustment arguments
#     ba_args = ['--image-list', img_txt_fn1,
#                '--camera-list', cam_txt_fn1,
#                '--overlap-list', overlap_list,
#                '--threads', str(iolib.cpu_count()),
#                '--num-iterations', str(num_iter),
#                '--num-passes', str(num_pass),
#                '--camera-position-weight', '0',
#                '--remove-outliers-params', "75 3 20 20",
#                '--min-matches', '4',
#                '--ip-per-tile', '2000',
#                '--individually-normalize',
#                '--inline-adjustments',
#                '--min-triangulation-angle', '0.0001',
#                '--save-cnet-as-csv',
#                '-o', ba_prefix]
#     # print(ba_args)
#     print("Running bundle adjustment round 1")
#     run_cmd('bundle_adjust', ba_args)
    
#     ### ROUND 2: refine with DEM, (optional) solve intrinsics
#     if float_intrinsics:
#         # Split cameras into individual files
#         def extract_unique_cam_strings(file_list):
#             ssc_pattern = re.compile(r'ssc\d+d\d+')
#             unique_ssc_strings = set()
#             for file_name in file_list:
#                 match = ssc_pattern.search(file_name)
#                 if match:
#                     unique_ssc_strings.add(match.group())
#             return list(unique_ssc_strings)
#         cam_strings = extract_unique_cam_strings(cam_list)
#         print('Number of unique cameras:', len(cam_strings))
    
#         # Add initial intrinsics to cameras
#         print('Adding initial intrinsics parameters to camera files')
#         cam_list_out = [os.path.join(out_dir, os.path.basename(cam_fn)) for cam_fn in cam_list]
#         p_map(add_cam_intrinsics, cam_list, cam_list_out, num_cpus=iolib.cpu_count())
    
#         # Write separate text files for each camera
#         for cam_string in cam_strings:
#             cam_fns_string = [cam_fn for cam_fn in cam_list_out if cam_string in cam_fn]
#             img_fns = [os.path.join(img_folder, os.path.basename(cam_fn).replace('_rpc.tsai', '.tif')) 
#                     for cam_fn in cam_fns_string]
#             # Write image file names to text file
#             with open(os.path.join(out_dir, f'round2_{cam_string}_images.txt'), 'w') as file:
#                 for image_fn in img_fns:
#                     file.write(f'{image_fn}\n')
#             # Write camera file names to text file
#             with open(os.path.join(out_dir, f'round2_{cam_string}_cameras.txt'), 'w') as file:
#                 for cam_fn in cam_fns_string:
#                     file.write(f'{cam_fn}\n') 
#         img_txt_fns = sorted(glob.glob(os.path.join(out_dir, 'round2*images.txt')))
#         cam_txt_fns = sorted(glob.glob(os.path.join(out_dir, 'round2*cameras.txt')))
#         img_txt_fn2 = ','.join(img_txt_fns)
#         cam_txt_fn2 = ','.join(cam_txt_fns)
#         cam_wt = '100'
    
#     else:
#         img_txt_fn2 = img_txt_fn1
#         cam_txt_fn2 = cam_txt_fn1
#         cam_wt = '0'
        
#     # Define bundle adjustment arguments
#     ba_args = ['--image-list', img_txt_fn2,
#                '--camera-list', cam_txt_fn2,
#                '--overlap-list', overlap_list,
#                '--heights-from-dem', dem,
#                '--heights-from-dem-uncertainty', '6',
#                '--force-reuse-match-files',
#                '--match-files-prefix', ba_prefix,
#                '--threads', str(iolib.cpu_count()),
#                '--num-iterations', str(num_iter),
#                '--num-passes', str(num_pass),
#                '--camera-position-weight', cam_wt,
#                '--remove-outliers-params', "75 3 20 20",
#                '--min-matches', '4',
#                '--ip-per-tile', '4000',
#                '--individually-normalize',
#                '--inline-adjustments',
#                '--min-triangulation-angle', '0.0001',
#                '--save-cnet-as-csv',
#                '-o', ba_prefix + '-run']

#     if float_intrinsics:
#         ba_args.extend(['--solve-intrinsics',
#                         '--intrinsics-to-float', "optical_center,focal_length,other_intrinsics"])
#     # print(ba_args)
#     print('Running bundle adjustment round 2')
#     run_cmd('bundle_adjust', ba_args)


def gridding_wrapper(pc_list,tr,tsrs=None):
    if tsrs is None:
        print("Projected Target CRS not provided, reading from the first point cloud")
        
        #fetch the PC-center.txt file instead
        # should probably make this default after more tests and confirmation with Oleg
        pc_center = os.path.splitext(pc_list[0])[0]+'-center.txt'
        with open(pc_center,'r') as f:
            content = f.readlines()
        X,Y,Z = [float(x) for x in content[0].split(' ')[:-1]]
        ecef_proj = 'EPSG:4978'
        geo_proj = 'EPSG:4326'
        ecef2wgs = Transformer.from_crs(ecef_proj,geo_proj)
        clat,clon,h = ecef2wgs.transform(X,Y,Z)
        epsg_code = f'EPSG:{geo.compute_epsg(clon,clat)}'
        print(f"Detected EPSG code from point cloud {epsg_code}") 
        tsrs = epsg_code
    n_cpu = iolib.cpu_count()    
    point2dem_opts = asp.get_point2dem_opts(tr=tr, tsrs=tsrs,threads=1)
    job_list = [point2dem_opts + [pc] for pc in pc_list]
    p2dem_log = p_map(asp.run_cmd,['point2dem'] * len(job_list), job_list, num_cpus = n_cpu)
    print(p2dem_log)
    
    
def alignment_wrapper_single(ref_dem,source_dem,max_displacement,outprefix,
                             align='point-to-plane',trans_only=0,initial_align=None):
    if trans_only == 0:
        trans_only = False
    else:
        trans_only = True
    asp.dem_align(ref_dem, source_dem, max_displacement, outprefix, align,
                  trans_only, threads=iolib.cpu_count(), initial_align=initial_align)
    
def alignment_wrapper_multi(ref_dem,source_dem_list,max_displacement,align='point-to-plane',initial_align=None,
                            trans_only=0):
    outprefix_list=['{}_aligned_to{}'.format(os.path.splitext(source_dem)[0],os.path.splitext(os.path.basename(ref_dem))[0]) for source_dem in source_dem_list]
    if trans_only == 0:
        trans_only = False
    else:
        trans_only = True
    n_source = len(source_dem_list)
    
    initial_align = [initial_align]*n_source
    ref_dem_list=[ref_dem] * n_source
    max_disp_list=[max_displacement] * n_source
    align_list=[align] * n_source
    trans_list=[trans_only] * n_source
    p_umap(asp.dem_align,ref_dem_list,source_dem_list,max_disp_list,outprefix_list,
           align_list,trans_list,[1]*n_source,initial_align,num_cpus = iolib.cpu_count())
    
def align_cameras_wrapper(input_camera_list,transform_txt,outfolder,rpc=0,dem='None',img_list=None):
    n_cam=len(input_camera_list)
    if (rpc == 1) & (dem != 'None'):
        print("Will also write RPC files")
        rpc = True
    else:
        dem = None
        img_list = [None] * n_cam
        rpc = False
    transform_list = [transform_txt]*n_cam
    outfolder = [outfolder] * n_cam
    write = [True] * n_cam
    rpc = [rpc] * n_cam
    dem = [dem] * n_cam
    
    p_umap(asp.align_cameras,input_camera_list,transform_list,outfolder,write,rpc,dem,
           img_list,num_cpus = iolib.cpu_count())
    

def plot_composite_fig(ortho_fn, georegistered_median_dem_fn, count_fn, nmad_fn, outfn=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    import rasterio as rio
    from matplotlib import colors
    import numpy as np
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Read files
    def read_data(raster):
        data = raster.read(1)
        data = np.ma.masked_where(data == raster.nodata, data)
        return data
    ortho = rio.open(ortho_fn)
    ortho_data = read_data(ortho)
    dem = rio.open(georegistered_median_dem_fn)
    dem_data = read_data(dem)
    count = rio.open(count_fn)
    count_data = read_data(count)
    nmad = rio.open(nmad_fn)
    nmad_data = read_data(nmad)

    def remove_ticks(axes):
        axes.set_xticks([])
        axes.set_yticks([])

    def add_colorbar(image, axes):
        x0, width = axes.get_position().x0, axes.get_position().width
        cax = fig.add_axes([x0, 0.1, width, 0.03])
        cbar = plt.colorbar(image, cax=cax, orientation='horizontal')
        return cbar

    # Plot
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    fig.subplots_adjust(bottom=0.15)
    # Orthoimage
    ax[0].imshow(ortho_data, cmap='Greys_r',
                extent=(ortho.bounds.left/1e3, ortho.bounds.right/1e3, ortho.bounds.bottom/1e3, ortho.bounds.top/1e3))
    ax[0].set_title('Orthoimage')
    remove_ticks(ax[0])
    # Add scale bar
    scalebar = AnchoredSizeBar(ax[0].transData,
                            1, '1 km', 'lower right', 
                            pad=0.3,
                            color='black',
                            frameon=False,
                            size_vertical=0.01)
    ax[0].add_artist(scalebar)
    # Shaded relief
    ls = colors.LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(dem_data, vert_exag=5)
    ax[1].imshow(hs, cmap='Greys_r',
                extent=(dem.bounds.left/1e3, dem.bounds.right/1e3, dem.bounds.bottom/1e3, dem.bounds.top/1e3))
    im = ax[1].imshow(dem_data, cmap='terrain', alpha=0.7, 
                    extent=(dem.bounds.left/1e3, dem.bounds.right/1e3, dem.bounds.bottom/1e3, dem.bounds.top/1e3))
    remove_ticks(ax[1])
    add_colorbar(im, ax[1])
    ax[1].set_title('DEM')
    # Count map
    im = ax[2].imshow(count_data, cmap='viridis', #clim=(0,np.nanpercentile(count_data, 95)),
                    extent=(count.bounds.left/1e3, count.bounds.right/1e3, count.bounds.bottom/1e3, count.bounds.top/1e3))
    remove_ticks(ax[2])
    add_colorbar(im, ax[2])
    ax[2].set_title('DEM counts')
    # NMAD
    # cmap = colors.LinearSegmentedColormap.from_list('custom', ['w', '#67000d'])
    im = ax[3].imshow(nmad_data, cmap='Reds', #clim=(0, np.nanpercentile(nmad_data, 95)),
                extent=(nmad.bounds.left/1e3, nmad.bounds.right/1e3, nmad.bounds.bottom/1e3, nmad.bounds.top/1e3))
    remove_ticks(ax[3])
    add_colorbar(im, ax[3])
    ax[3].set_title('NMAD')

    fig.savefig(outfn, dpi=300, bbox_inches='tight')
    print('Final figure saved to file:', outfn)
    plt.close()

    
def dem_mosaic_wrapper(dir,mode='triplet',out_folder=None,identifier=None,tile_size=None,filter_dem=1,min_video_count=2,max_video_nmad=5):
    if out_folder is None:
        out_folder = os.path.join(dir,'composite_dems')
    
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    if identifier is None:
        identifier = ''
    if mode == 'triplet':
        dir_list = sorted(glob.glob(os.path.join(dir,'20*/')))
        print(f"Number of combinations {len(dir_list)}")
        
        def valid_disp(direc):
            try:
                D_sub = iolib.fn_getma(os.path.join(direc,'run-D_sub.tif'),3)
                stats = np.percentile(D_sub.compressed(),(2,98))
                direc_out = glob.glob(os.path.join(direc,'run*{}*-DEM.tif'.format(identifier)))[0]
                out = (direc_out,True)
            except:
                out = (0,False)
            return out

    
        # find all valid DEMs
        total_cpu = iolib.cpu_count()
        n_proc = total_cpu - np.arange(len(dir_list))
        #this setup is so that old pools are discarded and new ones with new number of workers are created
        valid_dem_dir_list = []
        for idx,direc in enumerate(tqdm(dir_list)):
            comb_dir_list = sorted(glob.glob(os.path.join(direc,'*/')))
            results = p_map(valid_disp,comb_dir_list,num_cpus=n_proc[idx])
            t_val = [r[1] for r in results]
            valid_dirs = list(itertools.compress([r[0] for r in results],t_val))
            valid_dem_dir_list.append(valid_dirs)
        
        # naming logic for pairwise and triplet/multiview composites
        mdt1,t1,mdt2,t2 = [[],[],[],[]]
        combination_out_list = []
        if len(dir_list)>3:
            print("Input is cross-track")
        if dir_list[0][-1] == '/':
            dir_list = [x[:-1] for x in dir_list]
        for direc in dir_list:
            combination_out_list.append(os.path.join(out_folder,os.path.basename(direc)+'_wt_avg_mos.tif'))
            a,b,c,d = os.path.basename(direc).split('_')
            mdt1.append(a) #master date (year month date)
            t1.append(b) # time of day in seconds
            mdt2.append(c)
            t2.append(d)
        if len(direc)>3:
            composite_prefix = 'multiview_'+'_'.join(np.unique(mdt1+mdt2))+'__'+'_'.join(np.unique(t1+t2))
        elif len(direc) < 1:
            composite_prefix = 'stereo_'+'_'.join(np.unique(mdt1+mdt2))+'__'+'_'.join(np.unique(t1+t2))
        else:
            composite_prefix = 'triplet_'+'_'.join(np.unique(mdt1+mdt2))+'__'+'_'.join(np.unique(t1+t2))

        # produce bistereo pairwise mosaics
        len_combinations = len(combination_out_list)
        
        
        if len_combinations > 3:
            # force tiled processing in case of multiview mosaicking
            if not tile_size:
                tile_size = 400
        mos_log = p_map(asp.dem_mosaic,valid_dem_dir_list,combination_out_list,
                ['None']*len_combinations,[None]*len_combinations,
                [None]*len_combinations,[tile_size]*len_combinations)

        if len_combinations >= 1:
            print("Producing triplet/multiview composites")
            total_dem_list = list(itertools.chain.from_iterable(valid_dem_dir_list))
            print(f"Mosaicing {len(total_dem_list)} DEM strips using median, wt_avg, count, nmad operators")
            stats_list = [None,'median','nmad','count']
            out_fn_list = [os.path.join(out_folder,
                            '{}_{}_mos.tif'.format(composite_prefix,stat)) for stat in ['wt_avg','median','nmad','count']]
            composite_mos_log = p_map(asp.dem_mosaic,[total_dem_list]*4,out_fn_list,['None']*4,[None]*4,stats_list,
                                [tile_size]*4,num_cpus=4)            
        out_log_fn = os.path.join(out_folder,'skysat_triplet_dem_mos.log')
        print("Saving triplet DEM mosaic log at {}".format(out_log_fn))
        with open(out_log_fn,'w') as f:
            for log in mos_log+composite_mos_log:
                f.write(log) 
    if mode == 'video':
        dir_list = sorted(glob.glob(os.path.join(dir,'1*/')))
        valid_video_dir = []
        for video_dir in dir_list:
            try:
                D_sub = iolib.fn_getma(os.path.join(video_dir,'run-D_sub.tif'),3)
                stats = [np.percentile(D_sub.compressed(),(2,98)),np.mean(D_sub.compressed())]
                DEM = glob.glob(os.path.join(video_dir,'run*{}*-DEM.tif'.format(identifier)))[0]
                valid_video_dir.append(video_dir)
            except:
                continue 
        video_dem_list = [glob.glob(os.path.join(dir,f'run*{identifier}*-DEM.tif'))[0] for dir in valid_video_dir]
        stats_list = ['median','count','nmad']
        print('total dems are {}'.format(len(video_dem_list)))
        out_fn_list = [os.path.join(out_folder,'video_{}_mos.tif'.format(stat)) for stat in stats_list]
        dem_mos_log = p_map(asp.dem_mosaic,[video_dem_list]*3,out_fn_list,['None']*3,[None]*3,stats_list,[None]*3) 
        out_log_fn = os.path.join(out_folder,'skysat_video_dem_mos.log')
        with open(out_log_fn,'w') as f:
            for log in dem_mos_log:
                f.write(log)
        if filter_dem == 1:
            print("Filtering DEM using NMAD and count metrics")
            print(f"Filter will use min count of {min_video_count} and max NMAD of {max_video_nmad}")
            mos_ds_list = warplib.memwarp_multi_fn(out_fn_list)
            # Filtered array list contains dem_filtered,nmad_filtered, count_filtered in order
            filtered_array_list = skysat.filter_video_dem_by_nmad(mos_ds_list,min_video_count,max_video_nmad)
            trailing_str = f'_filt_max_nmad{max_video_nmad}_min_count{min_video_count}.tif'
            out_filter_fn_list = [os.path.splitext(fn)[0]+trailing_str for fn in out_fn_list]
            for idx,fn in enumerate(out_filter_fn_list):
                iolib.writeGTiff(filtered_array_list[idx],fn,mos_ds_list[idx])
                

                
                
def dense_match_wrapper(stereo_master_dir,ba_dir,modify_overlap=0,img_fol=None,orig_pickle=None,dense_match_pickle=None,stereo_dir=None,out_overlap_fn=None):
    """
    """
    triplet_stereo_matches = sorted(glob.glob(os.path.join(stereo_master_dir,'20*/*/run*-*disp*.match')))
    print('Found {} dense matches'.format(len(triplet_stereo_matches)))
    if  not os.path.isdir(ba_dir):
        os.makedirs(ba_dir)
    out_dense_match_list = [os.path.join(ba_dir,'run-'+os.path.basename(match).split('run-disp-',15)[1]) for match in triplet_stereo_matches]
    for idx,match in tqdm(enumerate(triplet_stereo_matches)):
        shutil.copy2(match, out_dense_match_list[idx])
    print("Copied all files successfully")
    
    if modify_overlap == 1:
        orig_df = pd.read_pickle(orig_pickle)
        dense_df = pd.read_pickle(dense_match_pickle)
        dense_img1 = list(dense_df.img1.values)
        dense_img2 = list(dense_df.img2.values)
        prioirty_list = list(zip(dense_img1,dense_img2))
        regular_img1 = [os.path.basename(x) for x in orig_df.img1.values]
        regular_img2 = [os.path.basename(x) for x in orig_df.img2.values]
        secondary_list = list(zip(regular_img1,regular_img2))
        # adapted from https://www.geeksforgeeks.org/python-extract-unique-tuples-from-list-order-irrespective/
        # note that I am using the more inefficient answer on purpose, because I want to use image pair order from the dense match overlap list
        total_list = priority_list + secondary_list
        final_overlap_set = set()
        temp = [final_overlap_set.add((a, b)) for (a, b) in total_list
              if (a, b) and (b, a) not in final_overlap_set]
        new_img1 = [os.path.join(img_fol,pair[0]) for pair in list(final_overlap_set)]
        new_img2 = [os.path.join(img_fol,pair[1]) for pair in list(final_overlap_set)]
        if not out_overlap_fn:
            out_overlap = os.path.join(ba_dir,'overlap_list_adapted_from_dense_matches.txt')
        else:
            out_overlap = os.path.join(ba_dir,out_overlap_fn)
        
        print("Saving adjusted overlap list at {}".format(out_overlap))
        with open(out_overlap,'w') as foo:
            for idx,img1 in enumerate(new_img1):
                out_str = '{} {}\n'.format(img1,new_img2[idx])
                f.write(out_str)
                

            
