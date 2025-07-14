#! /usr/bin/env python

try:
    import Metashape
except ImportError:
    raise ImportError('Could not import the Metashape Python library. Check your licence and installation.')

from glob import glob
import os
import pyproj
import re
import csv
from tqdm import tqdm
import rioxarray as rxr
from typing import List
from collections import defaultdict


def ecef_to_geodetic(x: float = None, 
                     y: float = None, 
                     z: float = None) -> tuple[float, float, float]:
    """
    Convert ECEF coordinates to geodetic coordinates (latitude, longitude, height).
    
    Parameters
    ----------
    x : float
        ECEF X coordinate.
    y : float
        ECEF Y coordinate.
    z : float
        ECEF Z coordinate.

    Returns
    ----------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees.
    h : float
        Height above the ellipsoid in meters.
    """
    transformer = pyproj.Transformer.from_crs("EPSG:4978", "EPSG:4326", always_xy=True)
    lon, lat, h = transformer.transform(x, y, z)
    return lat, lon, h


def load_tsai_camera(tsai_fn: str = None) -> tuple[Metashape.Matrix, float, float, float]:
    """
    Load a TSAI camera model from a .tsai file and convert it to a Metashape camera transform.

    Parameters
    ----------
    tsai_fn : str
        Path to the TSAI camera file.

    Returns
    ----------
    T : Metashape.Matrix
        Transformation matrix for the camera.
    lon : float
        Longitude of the camera position in degrees.
    lat : float
        Latitude of the camera position in degrees.
    h : float
        Height of the camera position above the ellipsoid in meters.
    """
    with open(tsai_fn, 'r') as f:
        lines = f.readlines()
    C_line = [l for l in lines if l.startswith("C =")]
    R_line = [l for l in lines if l.startswith("R =")]
    if C_line and R_line:
        pos = list(map(float, C_line[0].strip().split('=')[1].split()))
        R_vals = list(map(float, R_line[0].strip().split('=')[1].split()))
        R = [
            [R_vals[0], R_vals[1], R_vals[2]],
            [R_vals[3], R_vals[4], R_vals[5]],
            [R_vals[6], R_vals[7], R_vals[8]]
        ]
        T = Metashape.Matrix([
            [R[0][0], R[0][1], R[0][2], pos[0]],
            [R[1][0], R[1][1], R[1][2], pos[1]],
            [R[2][0], R[2][1], R[2][2], pos[2]],
            [0, 0, 0, 1]
        ])
        lat, lon, h = ecef_to_geodetic(*pos)

    return T, lon, lat, h


def load_rpc_camera(rpc_fn: str = None) -> Metashape.RPCModel:
    """
    Load RPC camera model from a .txt file and convert it to a Metashape RPCModel object.

    Parameters
    ----------
    rpc_fn : str
        Path to the RPC camera file.

    Returns
    ----------
    rpc : Metashape.RPCModel
        Metashape RPC model object containing the camera parameters.
    """

    # Load RPC data from text file
    rpc_data = {}
    with open(rpc_fn, 'r') as f:
        for line in f:
            if ':' in line:
                key, val = line.strip().split(':', 1)
                key = key.strip()
                val = val.strip()
                try:
                    if '_' in key and key[-1].isdigit():
                        prefix = '_'.join(key.split('_')[:-1])
                        rpc_data.setdefault(prefix, []).append(float(val))
                    else:
                        rpc_data[key] = float(val)
                except ValueError:
                    rpc_data[key] = val

    # Construct Metashape RPCModel object
    rpc = Metashape.RPCModel()

    # Image offset and scale
    rpc.image_offset = Metashape.Vector([
        rpc_data["SAMP_OFF"],  # X (sample)
        rpc_data["LINE_OFF"]   # Y (line)
    ])
    rpc.image_scale = Metashape.Vector([
        rpc_data["SAMP_SCALE"],
        rpc_data["LINE_SCALE"]
    ])

    # Object offset and scale (lon, lat, height)
    rpc.object_offset = Metashape.Vector([
        rpc_data["LONG_OFF"],  # X (longitude)
        rpc_data["LAT_OFF"],   # Y (latitude)
        rpc_data["HEIGHT_OFF"]
    ])
    rpc.object_scale = Metashape.Vector([
        rpc_data["LONG_SCALE"],
        rpc_data["LAT_SCALE"],
        rpc_data["HEIGHT_SCALE"]
    ])

    # Coefficients: should be 20 elements each
    rpc.line_num_coeff = Metashape.Vector(rpc_data["LINE_NUM_COEFF"])
    rpc.line_den_coeff = Metashape.Vector(rpc_data["LINE_DEN_COEFF"])
    rpc.samp_num_coeff = Metashape.Vector(rpc_data["SAMP_NUM_COEFF"])
    rpc.samp_den_coeff = Metashape.Vector(rpc_data["SAMP_DEN_COEFF"])

    return rpc


def align_photos(img_list: List[str] = None, 
                 crs: str = "EPSG:4326", 
                 out_folder: str = None, 
                 project_fn: str = None, 
                 cam_folder: str= None, 
                 gcp_csv: str = None) -> str:
    """
    Align photos in Metashape and export aligned cameras to XML and PDF report.

    Parameters
    ----------
    img_list : list of str
        List of image file paths to be aligned.
    crs : str, optional
        Coordinate reference system for the project (default is "EPSG:4326").
    out_folder : str, optional
        Output folder to save the Metashape project and results.
    project_name : str, optional
        Name of the Metashape project (default is "align_photos").
    cam_folder : str, optional
        Folder containing refined ASP .tsai camera models to initialize camera positions.
    gcp_csv : str, optional
        Path to tab-delimited CSV file with GCPs: columns = index, lat, lon, elevation.

    Returns
    -------
    aligned_cams_fn : str
        File path to the exported aligned cameras XML file.
    """
    # Make sure output directory exists
    os.makedirs(out_folder, exist_ok=True) 
    
    # Define outputs
    if not project_fn:
        project_name = 'align_photos'
        project_fn = os.path.join(out_folder, project_name + '.psx')
    else:
        project_name = os.path.splitext(os.path.basename(project_fn))[0]
    aligned_cams_fn = os.path.join(out_folder, project_name + "_aligned_cameras.xml")
    report_fn = os.path.join(out_folder, project_name  + "_report.pdf")
    
    # Start a project
    doc = Metashape.Document()
    doc.save(project_fn, chunks=doc.chunks)
    chunk = doc.addChunk()
    
    # Set project CRS
    chunk.crs = Metashape.CoordinateSystem(crs)
    
    # Check if RPC TXT files exist
    if cam_folder:
        load_rpc = False
    elif len(glob(os.path.join(os.path.dirname(img_list[0]), '*RPC.TXT'))) > 0:
        load_rpc = True
    else:
        load_rpc = False
    
    # Add photos
    chunk.addPhotos(img_list, load_rpc_txt=load_rpc)
    doc.save(project_fn, chunks=doc.chunks)

    # Load cameras
    if cam_folder:
        for camera in chunk.cameras:
            base = os.path.basename(camera.photo.path)
            identifier = '_'.join(base.split('_')[0:4])
            # check if TSAI camera
            tsai_fn = glob(os.path.join(cam_folder, identifier + '*.tsai'))
            if len(tsai_fn) > 0:
                tsai_fn = tsai_fn[0]
                T, lon, lat, h = load_tsai_camera(tsai_fn)
                camera.transform = T
                camera.reference.enabled = True
                camera.reference.location = Metashape.Vector([lon, lat, h])
                camera.reference.accuracy = Metashape.Vector([10, 10, 10])
        
            # check if RPC camera
            rpc_fn = glob(os.path.join(cam_folder, identifier + '*RPC.TXT'))
            if len(rpc_fn) > 0:
                rpc_fn = rpc_fn[0]
                # load RPC data
                rpc = load_rpc_camera(rpc_fn)
                # assign RPC to camera
                lon, lat, h = rpc.object_offset
                camera.reference.location = Metashape.Vector([lon, lat, h])
                camera.reference.enabled = True
                camera.reference.accuracy = Metashape.Vector([10, 10, 10])

        doc.save(project_fn, chunks=doc.chunks)

    # Load GCPs
    if gcp_csv:
        marker_dict = {} 
        with open(gcp_csv, 'r') as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) < 10:
                    continue
                marker_id = row[0]
                lat = float(row[1])
                lon = float(row[2])
                elev = float(row[3])
                # Create marker if it doesn't exist
                if marker_id not in marker_dict:
                    marker = chunk.addMarker()
                    marker.label = f"marker_{marker_id}"
                    marker.reference.location = Metashape.Vector([lon, lat, elev])
                    marker_dict[marker_id] = marker
                else:
                    marker = marker_dict[marker_id]
                
        doc.save(project_fn, chunks=doc.chunks)

    # Match and align photos
    chunk.matchPhotos(downscale = 2,
                      generic_preselection=True,
                      reference_preselection=False,
                      keypoint_limit=40000,
                      tiepoint_limit=10000,
                      reset_matches=True)
    chunk.alignCameras()
    doc.save(project_fn, chunks=doc.chunks)

    # Export aligned cameras
    chunk.exportCameras(path=aligned_cams_fn, 
                        use_labels=True, 
                        save_points=True, 
                        save_markers=False,
                        save_invalid_matches=False)

    # Export report
    chunk.exportReport(report_fn)
    
    return aligned_cams_fn


def build_dem(project_fn: str = None, 
              dem_resolution: float = 2, 
              out_dir: str = None,
              out_nodata=-9999) -> tuple[str, str, str]:
    """
    Build a dense point cloud, DEM, and orthomosaic from a Metashape project containing aligned photos.

    Parameters
    ----------
    project_fn : str
        Path to the Metashape project file (.psx).
    dem_resolution: float
        Spatial resolution of the output DEM file. 
    out_dir : str
        Output directory to save the dense point cloud and DEM.
    out_crs: str
        Output Coordinate Reference System of DEM and orthomosaic. 

    Returns
    ---------
    None
    """
    # Make sure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # Check if project file exists
    if not os.path.exists(project_fn):
        raise FileNotFoundError(f"Project file {project_fn} does not exist.")
    
    # Load project
    doc = Metashape.Document()
    doc.open(project_fn)
    project_name = os.path.splitext(os.path.basename(project_fn))[0]    

    # Check if project has a chunk
    if not doc.chunks:
        raise ValueError("No chunks found in the project.")
    chunk = doc.chunks[0]

    # Group cameras by date
    # def extract_date(label):
    #     match = re.match(r"(\d{8})", label)
    #     return match.group(1) if match else None
    # cameras_by_date = defaultdict(list)
    # for camera in chunk.cameras:
    #     date = extract_date(camera.label)
    #     if date:
    #         cameras_by_date[date].append(camera)

    # # Create a chunk for each date and process
    # dem_fns, ortho_fns = [], [] # initialize list of files
    # for date, cameras in cameras_by_date.items():
    #     print(f"Processing date: {date} with {len(cameras)} images")

        # # Duplicate chunk
        # date_chunk = chunk.copy()
        # date_chunk.label = date

        # # Disable images not captured on date
        # for cam in date_chunk.cameras:
        #     if cam not in cameras:
        #         cam.enabled = False

    # Define outputs
    pc_fn = os.path.join(out_dir, project_name + f'_point_cloud.laz')
    dem_fn = os.path.join(out_dir, project_name + f'_DEM.tif')
    ortho_fn = os.path.join(out_dir, project_name + f'_orthomosaic.tif')
        
    # Build depth maps
    print("\nBuilding depth maps...")
    chunk.buildDepthMaps(
        downscale = 2, # 2 = high quality
        filter_mode = Metashape.MildFiltering
        )
    doc.save(project_fn, chunks=doc.chunks)

    # Build dense point cloud
    print("\nBuilding dense point cloud...")
    chunk.buildPointCloud(
        point_colors = True, 
        point_confidence = True
        )
    doc.save(project_fn, chunks=doc.chunks)
    # export to file
    chunk.exportPointCloud(path=pc_fn)

    # Build DEM
    print("\nBuilding DEM...")
    # for whatever reason, you need to save and reload before building DEM...
    doc.save(project_fn, chunks=doc.chunks)
    chunk = doc.chunks[0]
    chunk.buildDem(
        source_data=Metashape.PointCloudData,
        resolution=dem_resolution,
        interpolation=Metashape.Interpolation.DisabledInterpolation
        )
    doc.save(project_fn, chunks=doc.chunks)
    # export to file
    chunk.exportRaster(
        dem_fn, 
        source_data = Metashape.ElevationData,
        image_format = Metashape.ImageFormatTIFF, 
        format = Metashape.RasterFormatTiles, 
        nodata_value = out_nodata, 
        save_kml = False, 
        save_world = False,
        resolution = dem_resolution
        )

    # Build orthomosaic
    print('\nBuilding orthomosaic...')
    chunk.buildOrthomosaic(surface_data=Metashape.ElevationData)
    doc.save(project_fn, chunks=doc.chunks)
    # export to fiile
    chunk.exportRaster(
        ortho_fn,
        source_data = Metashape.OrthomosaicData,
        split_in_blocks = False,
        nodata_value = out_nodata
        )

    return dem_fn, ortho_fn


def plot_results_fig(dem_fn: str = None, 
                     ortho_fn: str = None, 
                     fig_fn: str = None) -> None:
    """
    Plot a figure showing the orthomosaic and DEM side by side.
    
    Parameters
    ----------
    dem_fn : str
        Path to the DEM file.
    ortho_fn : str
        Path to the orthomosaic file.
    fig_fn : str
        Path to save the output figure.

    Returns
    ----------
    None
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    from matplotlib import colors
    import numpy as np
    import xarray as xr

    # Load inputs
    def load_raster(raster_fn, scaler=1):
        raster = rxr.open_rasterio(raster_fn).squeeze()
        crs = raster.rio.crs
        nodata = raster.rio.nodata
        if not nodata:
            nodata = raster.isel(band=0).data[0][0]
        raster = xr.where(raster==nodata, np.nan, raster / scaler)
        raster = raster.rio.write_crs(crs)
        return raster
    dem = load_raster(dem_fn, scaler=1)
    ortho = load_raster(ortho_fn, scaler=1e4)

    # Plot
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.subplots_adjust(bottom=0.15)
    # Orthoimage
    ax[0].imshow(np.dstack([ortho.isel(band=2).data, ortho.isel(band=1).data, ortho.isel(band=0).data]),
                 extent=(ortho.rio.bounds()[0]/1e3, ortho.rio.bounds()[2]/1e3, 
                         ortho.rio.bounds()[1]/1e3, ortho.rio.bounds()[3]/1e3))
    ax[0].set_title('RGB orthomosaic')
    ax[0].set_xlabel('Easting [km]')
    ax[0].set_ylabel('Northing [km]')
    # Add scale bar
    scalebar = AnchoredSizeBar(ax[0].transData,
                               1, '1 km', 'lower right', 
                               pad=0.3,
                               color='black',
                               frameon=True,
                               size_vertical=0.01)
    ax[0].add_artist(scalebar)
    # Shaded relief
    ls = colors.LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(dem.data, vert_exag=5)
    ax[1].imshow(hs, cmap='Greys_r',
                 extent=(dem.rio.bounds()[0], dem.rio.bounds()[2], 
                         dem.rio.bounds()[1], dem.rio.bounds()[3]))
    im = ax[1].imshow(dem.data, cmap='terrain', alpha=0.7, 
                      extent=(dem.rio.bounds()[0]/1e3, dem.rio.bounds()[2]/1e3, 
                              dem.rio.bounds()[1]/1e3, dem.rio.bounds()[3]/1e3))
    ax[1].set_xlabel('Easting [km]')
    x0, width = ax[1].get_position().x0, ax[1].get_position().width
    cax = fig.add_axes([x0, 0.0, width, 0.03])
    plt.colorbar(im, cax=cax, orientation='horizontal', label='Elevation [m]')
    ax[1].set_title('DEM')
    ax[1].set_xlim(ax[0].get_xlim())
    ax[1].set_ylim(ax[0].get_ylim())
    plt.close()

    # Save to file
    fig.savefig(fig_fn, dpi=300, bbox_inches='tight')
    print('Results figure saved to file:', fig_fn)


def xml_cameras_to_rpc_txt(xml_fn: str = None, 
                           out_folder: str = None) -> None:
    """
    Convert Metashape aligned cameras XML to individual RPC.TXT files.
    
    Parameters
    ----------
    xml_path : str
        Path to the Metashape aligned cameras XML file.
    out_folder : str
        Output folder to save the RPC.TXT files.

    Returns
    ----------
    None
    """
    # Make sure output directory exists
    os.makedirs(out_folder, exist_ok=True)
    # Read entire XML as plain text
    with open(xml_fn, "r") as file:
        xml_text = file.read()
    # Find all camera blocks
    cameras = re.findall(r"<camera\b[^>]*?>.*?</camera>", xml_text, flags=re.DOTALL)
    print('Saving Metashape XML cameras as separate RPC TXT files...')
    for cam in tqdm(cameras):
        # Get label
        label_match = re.search(r'label="([^"]+)"', cam)
        label = label_match.group(1) if label_match else "unknown_label"
        # Resolution
        res = re.search(r'<resolution[^>]*width="(\d+)"[^>]*height="(\d+)"', cam)
        width, height = map(float, res.groups())
        samp_off = width / 2
        line_off = height / 2
        # image_scale
        image_scale = re.search(r"<image_scale>(.*?)</image_scale>", cam).group(1)
        samp_scale, line_scale = map(float, image_scale.strip().split())
        # object_scale
        object_scale = re.search(r"<object_scale>(.*?)</object_scale>", cam).group(1)
        long_scale, lat_scale, height_scale = map(float, object_scale.strip().split())
        # reference (center coordinates)
        ref = re.search(r'<reference\b[^>]*x="([-.\d]+)"[^>]*y="([-.\d]+)"[^>]*z="([-.\d]+)"', cam)
        long_off, lat_off, height_off = map(float, ref.groups())
        # Coefficients
        def extract_coeff(tag):
            m = re.search(f"<{tag}>(.*?)</{tag}>", cam, flags=re.DOTALL)
            return list(map(float, m.group(1).split())) if m else []
        def coeff_lines(prefix, values):
            return [f"{prefix}_{i+1}: {v}" for i, v in enumerate(values)]
        content = [
            f"LINE_OFF: {line_off}",
            f"SAMP_OFF: {samp_off}",
            f"LAT_OFF: {lat_off}",
            f"LONG_OFF: {long_off}",
            f"HEIGHT_OFF: {height_off}",
            f"LINE_SCALE: {line_scale}",
            f"SAMP_SCALE: {samp_scale}",
            f"LAT_SCALE: {lat_scale}",
            f"LONG_SCALE: {long_scale}",
            f"HEIGHT_SCALE: {height_scale}",
            *coeff_lines("LINE_NUM_COEFF", extract_coeff("line_num_coeff")),
            *coeff_lines("LINE_DEN_COEFF", extract_coeff("line_den_coeff")),
            *coeff_lines("SAMP_NUM_COEFF", extract_coeff("samp_num_coeff")),
            *coeff_lines("SAMP_DEN_COEFF", extract_coeff("samp_den_coeff")),
        ]
        # Save to file
        out_path = os.path.join(out_folder, f"{label}_RPC.TXT")
        with open(out_path, "w") as f:
            f.write("\n".join(content) + "\n")

