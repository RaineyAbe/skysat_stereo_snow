'''
Script for coregistering individual DEMs to reference DEM before mosaicking.
'''

import subprocess
from tqdm import tqdm
import os
import glob
from p_tqdm import p_map
import argparse
import shutil

def getparser():
    parser = argparse.ArgumentParser(description='Wrapper script to run individual DEM coregistration')
    parser.add_argument('-in_fol',default=None,type=str,help='path to Folder containing individual DEMs')
    parser.add_argument('-refdem_fn',default=None,type=str,help='path to reference DEM')
    parser.add_argument('-out_fol',default=None, type=str,help='path for output files')
    return parser


def main():
    parser = getparser()
    args = parser.parse_args()
    in_fol = args.in_fol
    refdem_fn = args.refdem_fn
    out_fol = args.out_fol

    if not os.path.exists(out_fol):
        os.mkdir(out_fol)
        print('Made output directory:', out_fol)

    # Get DEM fns
    dem_list = sorted(glob.glob(os.path.join(in_fol, '20*', '20*', '*run-DEM.tif')))
    print(f'Found {len(dem_list)} source DEMs to coregister')
    
    # Rename and move to out_fol
    print('Renaming and moving to out folder')
    for dem_fn in tqdm(dem_list):
        dem_out_fn = os.path.join(out_fol, os.path.dirname(dem_fn).split('/')[-1] + '_run-DEM.tif')
        print(dem_out_fn)
        if not os.path.exists(dem_out_fn):
            shutil.copyfile(dem_fn, dem_out_fn)
    # reset DEM list
    dem_list = sorted(glob.glob(os.path.join(out_fol, '*_run-DEM.tif')))

    # Run alignment
    print('Running alignment')
    for dem_fn in tqdm(dem_list):
        align_out_fn = dem_fn.replace('.tif', '_coreg-trans_source.tif')
        dem_out_fn = dem_fn.replace('.tif', '_coreg-trans_source-DEM.tif')
        if not os.path.exists(dem_out_fn):
            # run pc_align
            out_prefix = dem_fn.replace('.tif', '_coreg')
            cmd = ['pc_align', 
                    '--max-displacement', '100',
                    '--save-transformed-source-points',
                    '--highest-accuracy',
                    '--threads', '48',
                    '-o', out_prefix,
                    refdem_fn, dem_fn]
            out = subprocess.run(cmd, shell=False, capture_output=True)
            print(out)
            # run point2dem
            cmd = ['point2dem',
                   '--threads', '48',
                   '-o', os.path.splitext(align_out_fn)[0],
                   align_out_fn]
            out = subprocess.run(cmd, shell=False, capture_output=True)
            print(out)

    dem_out_list = sorted(glob.glob(os.path.join(mos_dem_dir, '*run-DEM_coreg-trans_source-DEM.tif')))
    print(f'{len(dem_out_list)} DEMs succesfully coregistered')

    def merge_dems(dems, stat, out_fn):
        cmd = ['dem_mosaic',
                '--tr', '2', 
                f'--{stat}',
                '--threads', '48', 
                '-o', out_fn] + dems 
        out = subprocess.run(cmd, shell=False, capture_output=True)
        print(out)

    print('Merging DEMs using the median, NMAD, and count statistics')
    stats = ['median', 'nmad', 'count']
    for stat in tqdm(stats):
        out_fn = os.path.join(mos_dem_dir, stat + '_mos.tif') 
        if not os.path.exists(out_fn):
            merge_dems(dem_out_list, stat, out_fn)


if __name__ == '__main__':
    main()
