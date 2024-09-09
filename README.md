Exploring sEMG data from ["A generic noninvasive neuromotor interface
for human-computer interaction"](https://www.biorxiv.org/content/10.1101/2024.02.23.581779v1.full.pdf)
=======


This repo is for loading and plotting surface electromyography (sEMG) data associated with the paper ["A generic noninvasive neuromotor interface for human-computer interaction"](https://www.biorxiv.org/content/10.1101/2024.02.23.581779v1.full.pdf).

The dataset contains 100 sEMG recordings for each of the three tasks described in the paper: `discrete_gestures`, `handwriting`, and `wrist_pose`. Each recording is packaged in an `hdf5` file. This repo contains utility functions for loading and plotting these recordings.

![Figure 1 from the paper](images/figure_1.png)

# Setup
> NOTE: The github repo described in this section will be made available upon publication of the paper.

First, clone the repo and install the package.

```
git clone git@github.com:facebookresearch/generic-neuromotor-interface-data.git ~/generic-neuromotor-interface-data

pip install -e .
```

# Download the data
> NOTE: The AWS addresses below will be updated as soon as the data are approved for open sourcing.

Download the full dataset (XXX GiB) with following command.

```
cd ~ && wget https://fb-ctrl-oss.s3.amazonaws.com/...
```

To quickly get started, you can alternatively download a smaller version of the dataset with only three recordings per task (XXX MiB).

```
cd ~ && wget https://fb-ctrl-oss.s3.amazonaws.com/...
```

# Run the notebook
Finally, use the `loading_emg_data.ipynb` notebook to see how data can be loaded and plotted.
