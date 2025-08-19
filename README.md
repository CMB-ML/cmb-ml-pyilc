This repo is a work-in-progress attempt to hold the PyILC portion of CMB-ML, separately from the rest. 

# Installation

## Short version:

- Download the CMB-ML repository
    - `git clone git@github.com:CMB-ML/cmb-ml.git`
    - `cd cmb-ml`
    - `git switch whatever` (dev-rm-pyilc)
    - Follow the README instructions there
- Get [PyILC](https://github.com/jcolinhill/pyilc)
  - Simply clone the repository
  - No installation is needed, CMB-ML runs the code as its own
  - This was run and tested with [the version from April 30, 2024](https://github.com/jcolinhill/pyilc/tree/7ced3ec392a520977b3c672a2a7af62064dcc296)
- Configure your local system
  - See [Setting up your environment](https://github.com/CMB-ML/cmb-ml-tutorials/blob/main/C_setting_up_local.ipynb) for more information
  - Set your CMB_ML_DATA environment variable, and ensure that the directory exists
    - E.g., `export CMB_ML_DATA=/data/jim/CMB_Data`
  - In pyilc_redir, edit the `__init__.py` file to point to the directory containing your local installation of pyilc (containing the pyilc `inputs.py` and `wavelets.py`)

## Installation instructions (full):

- Download the CMB-ML repository
    - `git clone git@github.com:CMB-ML/cmb-ml.git`
    - `cd cmb-ml`
    - `git switch whatever` (dev-rm-pyilc)
- Create the conda environment 
    - remove old conda installations (and Poetry... which can be gotten rid of as a whole)
        - `conda remove -n cmb-ml --all`
    - still required due to either namaster or torch... this could be fixed soon, possibly
    - `conda env create -f env.yaml`
    - To change the name of the environment, edit the file or use a different command.
- Activate the conda environment
    - `conda activate cmb-ml`
- Install CMB-ML
    - `which pip` (ensure that the response is within the conda environment)
    - `pip install .`
- Get [PyILC](https://github.com/jcolinhill/pyilc)
  - Simply clone the repository
  - No installation is needed, CMB-ML runs the code as its own
  - This was run and tested with [the version from April 30, 2024](https://github.com/jcolinhill/pyilc/tree/7ced3ec392a520977b3c672a2a7af62064dcc296)
- Configure your local system
  - See [Setting up your environment](https://github.com/CMB-ML/cmb-ml-tutorials/blob/main/C_setting_up_local.ipynb) for more information
  - Set your CMB_ML_DATA environment variable, and ensure that the directory exists
    - E.g., `export CMB_ML_DATA=/data/jim/CMB_Data`
  - In pyilc_redir, edit the `__init__.py` file to point to the directory containing your local installation of pyilc (containing the pyilc `inputs.py` and `wavelets.py`)
- Download some external science assets and the CMB-ML assets
  - See [Setting up your environment](https://github.com/CMB-ML/cmb-ml-tutorials/blob/main/C_setting_up_local.ipynb) for more information
  - External science assets include Planck's observations maps (from which we get information for producing noise) and Planck's NILC prediction map (for the mask; NILC is a parameter)
  - These are available from the original sources and a mirror set up for this purpose
  - CMB-ML assets include the substitute detector information and information required for downloading datasets
  - If you are not creating simulations, you only need one external science asset: "COM_CMB_IQU-nilc_2048_R3.00_full.fits" (for the mask)
  - Scripts are available in the `get_data` folder, which will download all files.
    - [Downloads from original sources](./get_data/get_assets.py) gets files from the official sources (and the CMB-ML files from this repo)
    - If you prefer to download fewer files, adjust [this executor](get_data/stage_executors/A_get_assets.py) (not recommended)



# PyILC Debias

## How to Run
PyILC debias can currently be run with the same `main_pyilc_predict.py` and prediction_executor as the usual PyILC. So do this, change the `debias` key in `config_pyilc.yaml` to `True`.

Before running with debiasing, run `half_mission_construction.py` and enter in the dataset folder of your choice (must have at least two simulations) in the `.py` file. This will create a new folder in the datasetfolder labeled Simulation_Half which will contain two half mission simulations (a and b) for every Simulation in the dataset. Half mission pairs have the same CMB and foregrounds but different noise. (Currently noise is pulled randomly from the simulation dataset and does reuse each noise simulation twice. In the future this will be done by `main_sims.py`)

When run for debiasing, a PyILC_CNILC folder will be created as usual and will hold a `config_all_dets.yaml` file. This yaml file has a few extra keys that are only used by pyilc debias and it will be fed into the debiasing code. An `NILC_C_Debias` folder will be created and will contain a seperate folder for each simulation. Each simulation folder will contain all cleaned maps generated as well as `.txt` files of the final estimated power spectra and the noise cross spectra. Most crucially, it will contain `plotted_cmb_ps.png` which compares the final estimate to the fiduial power spectra.

## Debias Method Explained
The Debias method works as follows:
- First, the observation maps of half mission A are fed through the ILC and a cleaned map is generated. 
  - Weights are created during this cleaning and saved in the working folder.
      - While the needlet coefficient maps are deleted from the working folder, these weights are always kept. They will remain in the folder for the rest of the cleaning.
- Next, in the `Simulation_Half` folder, a set of residual maps between the half mission A and B is created. These are then fed through the ILC.
    - When the ILC detects the weights from earlier in the working folder it does not compute new ones and uses the already existing ones.
- As a part of the assets of this method, there exists a folder called `debiasing_simulations`. This (currently) contains 11 sets of simulations (10 noise, 1 foreground).
    - These 11 sets are run through the ILC sequentially using the same weights.
- All possible pairs of cross spectra between 10 cleaned noise maps are computed (45 in total) and a new power spectrum is generated by taking the average of all 45 for each ell. This serves as the noise cross spectra estimate, N.
- Lastly, the power spectra of the clean observation map, the cleaned residual map, and the cleaned foreground map are computed. We denote them O, R, and F respectively.
- The debiased power spectra, C, is given by C = O - R/2 - F - N
    - This serves as the estimated power spectra of Half Mission Map A.