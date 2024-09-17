Exploring sEMG data from ["A generic noninvasive neuromotor interface
for human-computer interaction"](https://www.biorxiv.org/content/10.1101/2024.02.23.581779v1.full.pdf)
=======


This repo is for loading and plotting surface electromyography (sEMG) data associated with the paper ["A generic noninvasive neuromotor interface for human-computer interaction"](https://www.biorxiv.org/content/10.1101/2024.02.23.581779v1.full.pdf).

The dataset contains 100 sEMG recordings for each of the three tasks described in the paper: `discrete_gestures`, `handwriting`, and `wrist_angles`. Each recording is packaged in an `hdf5` file. This repo contains utility functions for loading and plotting these recordings.

![Figure 1 from the paper](images/figure_1.png)

# Download the data
> NOTE: The AWS addresses below will be updated as soon as data are approved for open sourcing.

Download the full dataset (XXX GiB) with following command.

```
cd ~ && wget https://fb-ctrl-oss.s3.amazonaws.com/...
```

Alternatively, you can download a smaller version of the dataset with only three recordings per task (XXX MiB) to quickly get started.

```
cd ~ && wget https://fb-ctrl-oss.s3.amazonaws.com/...
```

# Setup
> NOTE: The github repo described in this section will be made available upon publication of the paper.

First, clone the repo, setup the conda environment, and install the local package.

```
# Clone the repo
git clone git@github.com:facebookresearch/generic-neuromotor-interface-data.git ~/generic-neuromotor-interface-data

# Setup and activate the environment
cd ~/generic-neuromotor-interface-data
conda env create -f environment.yml
conda activate neuromotordata

# Install the generic-neuromotor-inferface-data package
pip install -e .
```


# Run the notebook
Finally, use the `loading_emg_data.ipynb` notebook to see how data can be loaded and plotted.

```
jupyter lab loading_emg_data.ipynb
```
