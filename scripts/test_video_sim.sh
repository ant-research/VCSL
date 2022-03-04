#!/usr/bin/env bash
FEAT=rmac
FEAT=dino_base_ps8_i224
DATASET=bili211015
SPLIT=test

python run_video_sim.py \
       --pair-file data/pair_file_${SPLIT}.csv \
       --data-file data/frames_all.csv \
       --input-root result/temper/video/${FEAT}/${DATASET}/ \
       --input-store oss \
       --output-root result/vcsl/video/sim/${FEAT}/${DATASET}/ \
       --output-store local \
       --data-workers 32 \
       --request-workers 8