#!/usr/bin/env bash
FEAT=ISC
VTA=DTW

# The following "--pred-file" is obtained by running test_video_vta.sh

python run_video_eval.py \
       --pool-size 16 \
       --pred-file result/output/${FEAT}-${VTA}-pred-new.json  \
       --pred-store local \
       --split test