#!/usr/bin/env bash

FEAT=dino
VTA=DTW
DATASET=VCSL
DATE=21112318

# Before executing following script, similarity maps need to be calculated by test_video_sim.sh, and
# similarity maps are put in the same folder as the following "--input-root".

python run_video_vta_tune.py \
        --pair-file data/pair_file_val.csv \
        --input-root result/sim/${FEAT}/${DATASET}/ \
        --input-store local \
        --batch-size 32 \
        --data-workers 32 \
        --request-workers 16 \
        --alignment-method ${VTA} \
        --output-workers 16 \
        --output-root result/vcsl/tune/${FEAT}-${VTA}-${DATE}/ \
        --tn-max-step="5:15:5" \
        --tn-top-K="5:15:5" \
        --min-sim="0.2:0.31:0.1" \
        --discontinue="9:11:1" \
        --sum-sim="-2:10:1" \
        --diagonal-thres="10:50:10" \
        --ave-sim="1.1:1.31:0.1"


python run_video_vta.py \
        --pair-file data/pair_file_test.csv \
        --input-root result/sim/${FEAT}/${DATASET}/ \
        --input-store local \
        --batch-size 32 \
        --data-workers 32 \
        --request-workers 16 \
        --alignment-method ${VTA} \
        --output-root result/best/ \
        --result-file ${FEAT}-${VTA}-pred.json \
        --params-file result/vcsl/tune/${FEAT}-${VTA}-${DATE}/result.json
