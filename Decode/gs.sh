#!/usr/bin/env bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate marknull

python ../Baselines/Gaussian-Shading/decode.py --path "$1"
