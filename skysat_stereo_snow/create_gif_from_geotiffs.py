# /usr/bin/env python

"""
Create a GIF from a folder of GeoTIFFs. 
"""

import os
from glob import glob
import rioxarray as rxr
from PIL import Image
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# ---> MODIFY HERE: Define inputs and outputs
# folder containing images
image_folder = '/Volumes/LaCie/raineyaberle/Research/Hubbard/ArcticDEM_PGC_filtered/mosaics'
# output folder where individual images and GIF will be saved
out_folder = os.path.join(image_folder, 'gif')
# CRS to reproject each image if needed
out_crs = "EPSG:3338"
# set font size and type
plt.rcParams.update({'font.size': 12, 'font.sans-serif': 'Verdana'})

# make sure out_folder exists
os.makedirs(out_folder, exist_ok=True)

# get image file names
image_files = sorted(glob(os.path.join(image_folder, '*.tif')))
print(f'Found {len(image_files)} images in folder.')

# get min and max bounds from all images to set a uniform extent in each figure
# ---> CAN ALSO SPECIFY MANUALLY BY COMMENTING THIS OUT AND SETTING xmin, xmax, ymin, ymax 
# (currently assumes units = meters, sets extent in kilometers)
print('\nDetermining uniform extent for figures from all image bounds...')
bounds_list = []
for image in image_files:
    image = rxr.open_rasterio(image)
    if image.rio.crs != out_crs:
        image = image.rio.reproject(out_crs)
    bounds_list += [image.rio.bounds()]
bounds_list = np.array(bounds_list)
xmin = np.min(bounds_list[:,0]) / 1e3
xmax = np.max(bounds_list[:,2]) / 1e3
ymin = np.min(bounds_list[:,1]) / 1e3
ymax = np.max(bounds_list[:,3]) / 1e3
print(f'\txmin = {xmin} km'
      f'\txmax = {xmax} km'
      f'\tymin = {ymin} km'
      f'\tymax = {ymax} km')

# plot and save individual images
print('\nPlotting each GeoTIFF individually...')
for image_file in tqdm(image_files):
    # open image
    image = rxr.open_rasterio(image_file).squeeze()
    
    # reproject if needed
    if image.rio.crs != out_crs:
        image = image.rio.reproject(out_crs)
    
    # get the image bounds
    bounds = image.rio.bounds()

    # define output file name
    fig_file = os.path.join(out_folder, os.path.basename(image_file).replace('.tif', '.png'))

    # set up figure
    fig, ax = plt.subplots(1, 1, figsize=(8,8))

    # ASSUMES: 
    #   - band 1 = blue, band 2 = green, band 3 = red
    #   - CRS units = meters 
    # ---> FOR RGB IMAGE, USE THIS 
    ax.imshow(np.dstack([image.isel(band=2).data, image.isel(band=1).data, image.isel(band=0).data]),
              extent = [bounds[0] / 1e3, bounds[2] / 1e3, bounds[1] / 1e3, bounds[3] / 1e3])

    # ---> FOR SINGLE BAND IMAGE, USE THIS. CHANGE cmap AND clim AS NEEDED. 
    # im = ax.imshow(image.data, cmap='terrain', clim=(0, 1e3),
    #                extent = [bounds[0] / 1e3, bounds[2] / 1e3, bounds[1] / 1e3, bounds[3] / 1e3])
    # fig.colorbar(im, 
    #              label='Elevation [m]',
    #              shrink=0.8 # relative length of the colorbar w.r.t. the axis length. shrink = 1 for full length.
    #              )
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Easting [km]')
    ax.set_ylabel('Northing [km]')

    # ---> SET TITLE FROM FIRST 8 DIGITS OF FILE NAME
    ax.set_title(os.path.basename(image_file)[0:8])

    # save to file
    fig.savefig(fig_file, dpi=300, bbox_inches='tight')
    plt.close()

# Compile PNGs into a GIF
print('\nCompiling PNGs into a GIF file...')

# get figure file names
fig_files = sorted(glob(os.path.join(out_folder, '*.png')))

# open figures as PIL.Image
fig_image_list = [Image.open(file) for file in fig_files]

# save the first image as a GIF file
gif_file = os.path.join(out_folder, 'animation.gif')
fig_image_list[0].save(
    gif_file,
    save_all = True,
    append_images = fig_image_list[1:], # append the rest of the images
    duration = 1000, # for each frame in milliseconds
    loop = 1 # = 1 to keep looping, 0 to stop once its done
    )

print('Animation saved to file:', gif_file)