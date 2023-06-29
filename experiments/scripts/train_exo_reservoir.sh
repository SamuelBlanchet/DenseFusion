#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./tools/train_exo.py --dataset exo_reservoir --dataset_root ./datasets/reservoir_transparent_a/Reservoir_preprocessed
