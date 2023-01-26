#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=1

python3 ./tools/train.py --dataset linemod --dataset_root ./datasets/linemod/Linemod_preprocessed
