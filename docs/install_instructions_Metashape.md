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
pip install skysat_stereo_snow
```

3. Install the Agisoft Metashape Python 3 Module:
- Download the appropriate installer for OS: https://www.agisoft.com/downloads/installer/
- `pip install` the installer while in the skysat_stereo_snow environment. 

4. Download Basic Analytic (TOAR) unrectified assets for your SkySat capture. 

5. To run the full pipeline: See the `scripts/skysat_triplet_pipeline_Metashape.py` script. Borah users: see `scripts/skysat_triplet_pipeline_Metashape_slurm_example.bash` for an example job submission file. 