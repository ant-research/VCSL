#!/usr/bin/env bash
FEAT=dino
VTA=SPD

# The following "--pred-file" is obtained by running test_video_vta.sh

python run_video_eval.py \
       --pool-size 16 \
       --pred-file result/output/${FEAT}-${VTA}-pred.json  \
       --pred-store local \
       --split test