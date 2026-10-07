# spatial_pps_for_waiting

This repository contains a dataset of replicated spatial patterns of waiting pedestrians at a train station, as well as the accompanying code to estimate the parameters of determinantal (DPP) and Gibbs (GPP) spatial point process models for modeling. It accompanies our preprint [[1]](#ref1).

## Requirements

The code in this repository consists of scripts written in `python` and scripts written in `R`. For the `python` code, we use several standard packages for scientific computing, as well as `matplotlib` and `seaborn` for plotting and `dppy` for DPP simulation. The required packages (along with standard packages typically included in most `python` setups) are:

- `pandas`
- `numpy`
- `cupy`
- `pyarrow`
- `joblib`
- `matplotlib`
- `seaborn`
- `scipy`
- `scikit-learn`
- `numba`
- `dppy`

Note that `cupy` is designed to work with NVIDIA GPUs, with only experimental builds for AMD GPUs. It is required for all DPP computations, but GPP code works without it. In `R`, we make heavy use of the `spatstat` package, which is specifically suited to spatial statistical analysis. In addition, we use `ggplot2` for plotting.

## Data

The data included in this repo is an excerpt of the publicly available Zenodo dataset [[2]](#ref2) under CC BY 4.0. Columns align with the format described in [[2]](#ref2). See `demo_data.ipynb` for more details.

## Demos

In order to guide the reader through some of this repository, `python` code used for data analysis and inference has been showcased in two `jupyter` notebooks, namely `demo_data.ipynb` and `demo_inference.ipynb`. Since simulation and validation are mainly done in `R` (with the exception of simulation for DPPs), we briefly outline how to use the code here:

### Simulation

For all three models, as well as for inhomogeneous Poisson processes and a discrete analogue (CSR - completely spatially random), scripts with corresponding names have been created in the `simulation` folder. Running each of these scripts creates a `.csv` file (for DPPs and CSR) or `.rds` file (for GPPs and Poisson) of simulations, ready for validation in a subfolder `data_simulation`. Model parameters to be changed are at the top of each script.

### Validation

In the `validation` folder, simply run the corresponding validation script for each model to generate the plots that compare each model with the data, which are saved to the `validation_plots` subfolder. Calculated metrics are saved to a `metrics` folder. In addition, running either of the comparison scripts compares the DPP model to the CSR model and data, or the two GPP models to the Poisson model and the data. Note that for the plot showing the average nearest neighbor distances as a function of the occupancy, enough data must be used, as only occupancies with `min_reps` scenarios are counted.

### Disclaimer

A large language model (Claude Opus 5.5) was used to organize, merge, and debug code written over the course of a long-running project for this repository. All AI-assisted output was validated and reviewed by the authors.

## References

<a id="ref1"></a>[1] L. Sickert Karam, R. M. Castro, M. Schoukens, and A. Corbetta, "Modeling inhomogeneous spatial point configurations with applications to replicated patterns in waiting crowds," arXiv preprint [arXiv:2606.14532](https://arxiv.org/abs/2606.14532) [physics.soc-ph], 2026.

<a id="ref2"></a>[2] L. Sickert Karam, R. M. Castro, M. Schoukens, and A. Corbetta, "Modeling inhomogeneous spatial point configurations with applications to replicated patterns in real-life waiting crowds - dataset: snapshots of pedestrians at Eindhoven train station," Dataset.(https://doi.org/10.5281/zenodo.23060582), 2026.