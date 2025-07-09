#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Check if required arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <TASK_NAME> <MODEL_DIR>"
    exit 1
fi

TASK_NAME=$1  # handwriting, discrete_gestures, or wrist
MODEL_DIR=$2  # The directory where the model will be stored

# Check that TASK_NAME is one of the expected values
if [ "$TASK_NAME" != "handwriting" ] && [ "$TASK_NAME" != "discrete_gestures" ] && [ "$TASK_NAME" != "wrist" ]; then
    echo "Invalid TASK_NAME value: $TASK_NAME"
    echo "Expected 'handwriting', 'discrete_gestures', or 'wrist'"
    exit 1
fi

URL="https://fb-ctrl-oss.s3.amazonaws.com/neuromotor-data"

echo "Downloading the pretrained model..."
mkdir -p "$MODEL_DIR"
mkdir -p "$MODEL_DIR/$TASK_NAME"
curl "$URL/checkpoints/$TASK_NAME/$TASK_NAME.tar" -o "$MODEL_DIR/$TASK_NAME/$TASK_NAME.tar"
tar -xvf "$MODEL_DIR/$TASK_NAME/$TASK_NAME.tar" -C "$MODEL_DIR/$TASK_NAME/"

echo "Done"
