# optoIDR MS2 data analysis

This repository contains source data and analysis scripts for part of the optoIDR-MCP experiments in Wei et al. 2020. Nucleated transcriptional condensates amplify gene expression. _Nature Cell Biology_ XX: XXXX. See Jupyter notebooks in `/notebook` (or [here](notebook/)) for demostrations of 

1. Correlation of optoIDR and MCP signals (Figure 5E-G in the paper) in [this notebook](notebook/droplet-mcp-correlation.ipynb)
2. Averaged images of optoIDR droplets (Figure 5H in the paper) in [this notebook](notebook/droplet-mcp-airyscan-overlay.ipynb)


## Environment setup
_Note: You do not need to set up this if all you would like to do is to view the contents of the notebooks. Just open those notebooks within GitHub._

We recommend use of [Conda](https://conda.io/projects/conda/en/latest/) to manage dependencies, which are specified in [`environment.yml`](environment.yml). The default environment name is `optoidr-mcp`. Before running the Jupyter notebooks, please install all dependencies by the following steps.

1. Create env by `conda env create -f environment.yml`
2. Activate conda environment by `conda activate optoidr-mcp`

## Reference
Wei MT, Chang YC, Shimobayashi SF, Shin Y, Strom AR, Brangwynne CP. 2020. Nucleated transcriptional condensates amplify gene expression. _Nature Cell Biology_ XX: XXXX