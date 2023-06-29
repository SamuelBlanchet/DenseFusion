#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=1

python3 ./tools/eval_exo.py --dataset_root datasets/exo/Exo_preprocessed\
    --model trained_models/exo/pose_model_55_0.012997642964667952.pth\
    --refine_model trained_models/exo/pose_refine_model_215_0.007262987860788901.pth
