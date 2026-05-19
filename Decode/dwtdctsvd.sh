#!/usr/bin/env bash

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate marknull

python ../Baselines/blind_watermark/dwtDctSvd.py --path "$1"
