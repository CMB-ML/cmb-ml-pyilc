# CMB-ML: PyILC

This repository implements the NILC component separation method used with the [CMB-ML](https://github.com/CMB-ML/cmb-ml) framework, using the CNILC implementation provided by [PyILC](https://github.com/jcolinhill/pyilc).

It includes the code and configurations for prediction and analysis, as well as instructions for reproducing the CNILC results from the Direct-to-Power-Spectrum ([D2PS](https://github.com/CMB-ML/e-d2ps)) paper, including half-mission cross-power spectrum
estimation and foreground debiasing.

## Installation

Follow the installation instructions in the [CMB-ML repository](https://github.com/CMB-ML/cmb-ml), using release `v0.3.0-d2ps`.

With the CMB-ML conda environment activated, clone and install the required version of PyILC:

```bash
git clone https://github.com/jcolinhill/pyilc.git
cd pyilc
git switch --detach f527e35fc6ab738d1a3c2724daab62430e35cc12
python -m pip install pixell==0.32.7
python -m pip install -e .
cd ..
```

PyILC must be installed in editable mode to ensure that required YAML files are accessible.

Clone this repository:

```bash
git clone https://github.com/CMB-ML/cmb-ml-pyilc.git
cd cmb-ml-pyilc
git switch --detach v0.2.0-d2ps
python -m pip check
```

## Data Preparation

To reproduce the results reported in the D2PS paper, use the `I_512_1850_UnifWMAP_Wide` dataset, which can be obtained by following the data preparation instructions in the CMB-ML repository.

In addition to the simulated observations, the PyILC pipeline requires the noise model and intermediate simulation products. These are included in the distributed dataset.

Ensure that the `CMB_ML_DATA` environment variable is set as described in the CMB-ML documentation.

The default configuration, `cfg/config_pyilc.yaml`, is set up to use this dataset.

## Running PyILC

To reproduce the CNILC predictions and debiased power spectra used in the D2PS paper, run:

```bash
python main_pyilc_predict.py
```

The pipeline generates half-mission observations and foreground debiasing realizations, computes CNILC weights, applies the weights, and estimates the debiased power spectra.

By default, the pipeline runs on the validation and test splits.

To test the pipeline on a smaller number of simulations per split, use the `n_infer_cap` configuration option:

```bash
python main_pyilc_predict.py n_infer_cap=12
```

The default working directory within the dataset is `PyILC_CNILC/`. The resulting power spectra are stored in `PyILC_CNILC/Analysis_K_Power_Spectra/` within the dataset directory. Further post-processing (polynomial fitting) is applied in D2PS.

## Attribution

If you use this repository in your research, please cite the relevant publications:

- **NILC method:** [J. Delabrouille, J.-F. Cardoso, M. Le Jeune, et al. 2009, A&A, 493, 835–857](https://doi.org/10.1051/0004-6361:200810514).
- **PyILC implementation:** [Fiona McCarthy and J. Colin Hill 2024, Phys. Rev. D, 109, 023528](https://doi.org/10.1103/PhysRevD.109.023528).
- **CMB-ML simulation framework:** [James Amato, Yunan Xie, Leonel Medina-Varela, et al. 2025, ICCV, 9418–9430](https://doi.org/10.1109/ICCV51701.2025.00879).

The configurations and analysis procedures provided with this release were developed for the following work:

- **D2PS:** Citation forthcoming.
