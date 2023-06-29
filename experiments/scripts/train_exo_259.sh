#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./tools/train_exo.py --dataset exo_259 --dataset_root ./datasets/part_259/259_preprocessed
