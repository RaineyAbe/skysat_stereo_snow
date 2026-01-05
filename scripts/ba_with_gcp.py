#! /usr/bin/env python
# bundle adjust with GCP from ortho and stable surfaces

import os
from glob import glob
import rioxarray as rxr
import sys
import shutil
import subprocess
import xarray as xr


def run_cmd(bin: str = None, 
            args: list = None, **kw) -> str:
    binpath = shutil.which(bin)
    if binpath.endswith('.py'):
        call = ['python', binpath,]
    else:
        call = [binpath,]
    if args is not None: 
        call.extend(args)
    try:
        out = subprocess.run(call,check=True,capture_output=True,encoding='UTF-8').stdout
    except:
        out = "the command {} failed to run, see corresponding asp log".format(call)
    return out


def merge_rasters(raster_files: str = None,
                  out_file: str = None,
                  t_res: float | int = None,
                  t_nodata: float | int = 0,
                  t_dtype: str = 'UInt16'
                   ):
    # Make sure output directory exists
    if not os.path.exists(os.path.dirname(out_file)):
        os.mkdir(os.path.dirname(out_file))

    # Check if output file already exists
    if os.path.exists(out_file):
        print("Raster mosaic already exists in file, skipping.")
        return

    # Set up mosaic arguments
    mos_args = ['-ot', t_dtype,
                '-a_nodata', str(t_nodata)]
    if t_res:
        mos_args.extend(['-ps', str(t_res), str(t_res)])
    mos_args.extend(['-o', out_file])
    mos_args.extend(raster_files)

    # Run image mosaic
    out = run_cmd('gdal_merge.py', mos_args)    
    print(out)
    return


def create_land_cover_masks(raster_file, out_folder=None):

    # Define output files
    snow_mask_file = os.path.join(out_folder, os.path.splitext(os.path.basename(raster_file))[0] + "_snow_mask.tif")
    veg_mask_file = os.path.join(out_folder, os.path.splitext(os.path.basename(raster_file))[0] + "_vegetation_mask.tif")
    ss_mask_file = os.path.join(out_folder, os.path.splitext(os.path.basename(raster_file))[0] + "_stable_surfaces_mask.tif")
    classified_file = os.path.join(out_folder, os.path.splitext(os.path.basename(raster_file))[0] + "_classified.tif")


    def save_raster(r, r_file, crs, name="Mask"):
        r = r.rio.write_crs(crs)
        r.rio.to_raster(
        r_file,
            dtype='uint8',
            driver="GTiff",
            compress='lzw'
        )
        print(f"{name} saved to:\n{r_file}")

    with rxr.open_rasterio(raster_file) as raster:
        crs = raster.rio.crs # save CRS for later

        # Account for image scalar
        if raster.isel(band=0).mean().values > 1e3:
            raster = raster / 1e5

        # define image bands
        B = raster.isel(band=0)
        NIR = raster.isel(band=3)

        # classify snow and save mask
        NDSI = (B - NIR) / (B + NIR)
        snow_mask = xr.where((NDSI > 0.15) & (B > 0.1), 1, 0).astype(int)
        save_raster(snow_mask, snow_mask_file, crs, "Snow mask")

        # classify vegetation
        veg_mask = xr.where((NDSI > 0.4) & (B < 0.1), 1, 0).astype(int)
        save_raster(veg_mask, veg_mask_file, crs, "Vegetation mask")

        # classify stable surfaces as all snow- and tree-free pixels
        ss_mask = xr.where((snow_mask==0) & (veg_mask==0), 1, 0).astype(int)
        save_raster(ss_mask, ss_mask_file, crs, name="Stable surfaces mask")

        # create classified image
        # 1 = snow
        # 1.5 = snow & vegetation
        # 2 = vegetation
        # 3 = stable
        classified = xr.where(snow_mask==1, 1,
                      xr.where((snow_mask==1) & (veg_mask==1), 1.5,
                      xr.where(veg_mask==1, 2,
                      xr.where(ss_mask==1, 3, 0))))
        save_raster(classified, classified_file, crs, "Classified image")


    return


def main():
    # Get input files
    data_folder = "/Users/rdcrlrka/Research/SkySat-Stereo/study-sites/MCS"
    date = "20240420"
    refdem_file = os.path.join(data_folder, "refdem", "MCS_refdem_lidar_COPDEM_merged.tif")
    refortho_files = sorted(glob(os.path.join(data_folder, date, "SkySatCollect*", "*_analytic.tif")))
    image_list = sorted(glob(os.path.join(data_folder, date, "SkySatScene", "*_analytic.tif")))
    print(f"Detected {len(image_list)} images and {len(refortho_files)} reference orthoimages.")

    # Add path to ASP
    asp_path = "/Users/rdcrlrka/Research/StereoPipeline-3.6.0-2025-12-31-arm64-OSX/bin"
    sys.path.append(asp_path)

    # Define outputs
    out_folder = os.path.join(data_folder, date, "ba_with_gcp")
    os.makedirs(out_folder, exist_ok=True)
    masks_folder = os.path.join(out_folder, "land_cover_masks")

    # Mosaic ortho files
    refortho_file = os.path.join(masks_folder, "refortho.tif")
    merge_rasters(refortho_files, refortho_file)

    # Construct land cover masks
    create_land_cover_masks(refortho_file, masks_folder)


if __name__=="__main__":
    main()