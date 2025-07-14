# Installation and pipeline instructions for `skysat_stereo_snow` using Agisoft Metashape

1. Clone (or fork then clone) this GitHub repository by running the following in the command line:

```
git clone https://github.com/RaineyAbe/skysat_stereo_snow
```

2. Install the required packages. We recommend using [Mamba or Micromamba](https://mamba.readthedocs.io/en/latest/index.html) for package management: 

```
cd skysat_stereo_snow
mamba env create -f environment.yml
mamba activate skysat_stereo_snow
pip install -e ../skysat_stereo_snow
```

3. Install the Agisoft Metashape Python 3 Module:
- Download the appropriate installer for your OS: https://www.agisoft.com/downloads/installer/
- `pip install Metashape-*.whl` while in the skysat_stereo_snow environment. 

4. Optional: For querying and downloading a reference DEM from Google Earth Engine, authenticate your GEE account. In the command line, run the following to start the authentication process. 

```
earthengine authenticate
```

5. Download Basic Analytic (TOAR) unrectified assets from your SkySat capture(s). 

6. To run the full pipeline: See the `scripts/skysat_triplet_pipeline_Metashape.py` script. NOTE: If you'd like to all align photos over the same region for different dates, then construct a separate DEM and orthomosaic for each date separately, place all images into a single folder. 

__Notes for Borah / HPC users:__ 

- Make sure your account has access to an Agisoft Metashape Professional License to authorize use of the Python API. (HP has a license on Borah you can request to be added to.)

- To download the Metashape Python 3 Module, you can log into your Borah account, copy the URL to the Linux download link, then run `wget [COPIED LINK]` in the command line on your Borah account. 

- See `skysat_triplet_pipeline_Metashape_slurm_example.bash` in the `scripts` folder for an example job submission file. 