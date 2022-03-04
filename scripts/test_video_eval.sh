#!/usr/bin/env bash
FEAT=rmac
VTA=DTW

python run_video_eval.py \
       --pool-size 16 \
       --pred-file ${FEAT}-${VTA}-pred.json  \
       --pred-store local \
       --split test