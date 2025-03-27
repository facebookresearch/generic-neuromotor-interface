#!/bin/bash
#
# Instructions:
#   Run in `generic_neuromotor_interface/fb_scripts` with `./dev_quickstart.sh TASK`
#   TASK should be one of(wrist, discrete_gestures, handwriting). wrist by default.
#
#   ->  You should `conda activate ctrldev` first.
#   ->  If running on a Meta devserver, make sure you have `fwd_proxy` enabled to reach
#           outside internet (i.e. AWS).


# TODO: delete before OSS

# Get the parent directory of the script
parent_dir="$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")"

# Change the current working directory to the parent directory
cd "$parent_dir" || exit

# Define a function for green color
printf_green() {
  printf "\033[92m%s\033[0m\n" "$*"
}

# Determine the task from command line arg
config_name=${1:-wrist}

# Step 1: install package
printf_green "---------------------------------------------------------------------------------"
printf_green "Step 1: Installing generic_neuromotor_interface package as editable (-e)"
printf_green "---------------------------------------------------------------------------------"
pip install -e .

# Step 2: train for 2 epochs (will run val / test as well)
printf_green "---------------------------------------------------------------------------------"
printf_green "Step 2: Running training script ($config_name) on subsampled data with max_epochs=2."
printf_green "---------------------------------------------------------------------------------"
python -m generic_neuromotor_interface.train --config-name="$config_name" download_subset=true trainer.max_epochs=2 trainer.accelerator=cpu data_module/data_split="$config_name"_mini_split
printf_green "Done with local training!"


# launch via local dag
cd "fb_scripts" || exit

# Step 3: train for 2 epochs using CTRL-DAG (local)
printf_green "---------------------------------------------------------------------------------"
printf_green "Step 3: Running local CTRL-DAG training"
printf_green "---------------------------------------------------------------------------------"
python launch_ctrl_dag.py --config-name="$config_name" local=true download_subset=true trainer.max_epochs=2 trainer.accelerator=cpu data_module/data_split="$config_name"_mini_split
printf_green "Done with local CTRL-DAG training!"

# Step 4: train for 2 epochs using CTRL-DAG (remote)
printf_green "---------------------------------------------------------------------------------"
printf_green "Step 4: Now launching to remote (CTRL's AWS cluster) via CTRL-DAG (max_epochs=10)"
printf_green "---------------------------------------------------------------------------------"
python launch_ctrl_dag.py --config-name="$config_name" local=false download_subset=true trainer.max_epochs=10 data_module/data_split="$config_name"_mini_split
printf_green "Done with remote CTRL-DAG launch!"
printf_green "Make sure the model is running at the ARGO link printed above ^"

printf_green "----------------------"
printf_green "All done. Good bye! 😀"
printf_green "----------------------"
