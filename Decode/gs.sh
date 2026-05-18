#!/usr/bin/env bash
cd ../Baselines/Gaussian-Shading

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate marknull

python decode.py --path "$1"
