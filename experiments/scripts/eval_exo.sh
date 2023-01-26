#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=1

python3 ./tools/eval_exo.py --dataset_root ./datasets/exo/Exo_preprocessed\
    --model trained_models/exo/pose_model_current.pth\
    --refine_model trained_models/exo/pose_refine_model_current.pth
#--refine_model trained_checkpoints/exo/pose_refine_model_from_linemod.pth
#--model trained_checkpoints/exo/pose_model_268_0.41909691212432726.pth\
