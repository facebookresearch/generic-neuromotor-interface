#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Check if required arguments are provided
if [ $# -ne 3 ]; then
    echo "Usage: $0 <TASK_NAME> <DATASET> <EMG_DATA_DIR>"
    exit 1
fi

TASK_NAME=$1  # handwriting, discrete_gestures, or wrist
DATASET=$2  # full_data or small_subset
EMG_DATA_DIR=$3  # The directory where the data will be stored

URL="https://fb-ctrl-oss.s3.amazonaws.com/neuromotor-data"

echo "Download the data..."
mkdir -p "$EMG_DATA_DIR"
curl "$URL/data/$TASK_NAME/$DATASET.tar" -o "$EMG_DATA_DIR/${TASK_NAME}_$DATASET.tar"

echo "Unzipping the data..."
tar -xvf "$EMG_DATA_DIR/${TASK_NAME}_$DATASET.tar" -C "$EMG_DATA_DIR"

echo "Downloading the corpus spreadsheet..."
curl "$URL/data/$TASK_NAME/$TASK_NAME"_corpus.csv -o "$EMG_DATA_DIR/$TASK_NAME"_corpus.csv
