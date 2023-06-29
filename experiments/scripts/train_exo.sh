#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0,1

python3 ./tools/train_exo.py --dataset exo --dataset_root ./datasets/exo/Exo_preprocessed
