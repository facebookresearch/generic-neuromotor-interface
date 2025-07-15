# A generic noninvasive neuromotor interface for human-computer interaction

[ [`Paper`](https://www.biorxiv.org/content/10.1101/2024.02.23.581779v2) ] [ [`BibTeX`](#citation) ]

This repo is for exploring surface electromyography (sEMG) data and training models associated with the paper ["A generic noninvasive neuromotor interface for human-computer interaction"](https://www.biorxiv.org/content/10.1101/2024.02.23.581779v2).

The dataset contains sEMG recordings from 100 participants in each of the three tasks described in the paper: `discrete_gestures`, `handwriting`, and `wrist`. This repo contains implementations of the models in the paper as well as code for training and evaluating the models.

![Figure 1 from the paper](images/figure_1.png)

## Download the data and models

> NOTE: The instructions on this page are for MacOS.

Clone the `generic-neuromotor-interface` repo and move it to your home directory.

To download the full dataset to `~/emg_data` for a given task, run:

```bash
cd ~/generic-neuromotor-interface
./scripts/download_data.sh <TASK_NAME> full_data ~/emg_data
```

where `<TASK_NAME>` is one of `discrete_gestures, handwriting, wrist`.

Alternatively, you can download and extract a smaller version of the dataset with only 3 participants per task to quickly get started:

```bash
cd ~/generic-neuromotor-interface
./scripts/download_data.sh <TASK_NAME> small_subset ~/emg_data
```

To download the models for a task, run:

```bash
./scripts/download_models.sh <TASK_NAME> ~/emg_models
```

## Setup

Now setup the conda environment and install the local package.

```bash
# Setup and activate the environment
cd ~/generic-neuromotor-interface
conda env create -f environment.yml
conda activate neuromotor

# Install this repository as a package
pip install -e .
```

## Explore the data in a notebook

Use the `explore_data.ipynb` notebook to see how data can be loaded and plotted:

```bash
jupyter lab notebooks/explore_data.ipynb
```

## Train a model

Train a model via:

```bash
python -m generic_neuromotor_interface.train --config-name=<TASK_NAME>
```

Note that this requires downloading the `full_data` dataset as described earlier.

You can also launch a small test run (1 epoch on the `small_subset` dataset) via:

```bash
python -m generic_neuromotor_interface.train --config-name=<TASK_NAME> trainer.max_epochs=1 trainer.accelerator=cpu data_module/data_split=<TASK_NAME>_mini_split
```

After training, the model checkpoint will be available at `./logs/<DATE>/<TIME>/lightning_logs/<VERSION>/checkpoints/`, and the model config will be available at `./logs/<DATE>/<TIME>/hydra_configs/config.yaml`.

## Evaluate a model

Model evaluation on the test set is automatically performed in the training script after training is complete. But if you want to run evaluation ad-hoc on any given trained model, please see the evaluation notebooks provided at `notebooks/<TASK_NAME>-eval.ipynb` for code to run inference with each model and run a full evaluation on the test set. These notebooks also provide some visualizations of the model outputs.

## Dataset details

We are releasing data from 100 data collection participants for each task: 80 train, 10 validation, and 10 test participants. Train participants correspond to those from the 80 participant data point in Figures 2e-g (except for Handwriting, where we randomly selected 80 participants from the 100 participant data point). The 10 validation and train participants were randomly selected from the full set of validation and test participants.

Evaluation metrics may deviate slightly from the published results due to subsampling of the test participants (as there is considerable variability across participants) and variability across model seeds.

Each recording is stored in an `.hdf5` file, and there can be multiple recordings per participant. There is also a `.csv` file for each task (`<TASK_NAME>_corpus.csv`) documenting the recordings included for each participant, start and end times for each relevant "stage" from the experimental protocol (see below), and their assignment to the train / val / test splits. This `.csv` file is downloaded alongside the data by the `download_data.sh` script described above.

sEMG is recorded at 2 kHz and is high pass filtered at 40 Hz. Timestamps are expressed in seconds. A `stages` dataframe is included in each dataset that encodes the time of each stage of the experiment (see `explore_data.ipynb` for more details). Specifics for each task are as follows:

### Discrete gestures

Datasets include the `name` of each gesture and the `time` at which it occurred. Stage names include the types of gestures performed in each stage, as well as the posture (e.g. `static_arm_in_front`, `static_arm_in_lap`, ...)

### Handwriting

Handwriting datasets include the `start` and `end` time of each prompt. `start` is the time the prompt appears, and `end` is the time at which participants finished writing the prompt. Stage names describe the types of prompts in each stage (e.g. `words_with_backspace`, `three_digit_numbers`, ...).

### Wrist

Wrist angle datasets also include wrist angle measurements, which are upsampled to match the 2 kHz EMG sampling rate. Stage names include information about the type of task and movement in each stage (e.g. `cursor_to_target_task_horizontal_low_gain_screen_4`, `smooth_pursuit_task_high_gain_1`, ...).

## License

The dataset and the code are CC-BY-NC-4.0 licensed, as found in the LICENSE file.

## Citation

```
@article{generic_neuromotor_interface_2024,
 author = {CTRL-labs at Reality Labs},
 title = {A generic noninvasive neuromotor interface for human-computer interaction},
 elocation-id = {2024.02.23.581779},
 year = {2024},
 doi = {10.1101/2024.02.23.581779},
 publisher = {Cold Spring Harbor Laboratory},
 URL = {https://www.biorxiv.org/content/early/2024/07/23/2024.02.23.581779},
 eprint = {https://www.biorxiv.org/content/early/2024/07/23/2024.02.23.581779.full.pdf},
 journal = {bioRxiv}
}
```
