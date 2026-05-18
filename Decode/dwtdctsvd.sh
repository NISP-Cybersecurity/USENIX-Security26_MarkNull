#!/usr/bin/env bash
cd ../Baselines/blind_watermark

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate marknull

python dwtDctSvd.py --path "$1"
