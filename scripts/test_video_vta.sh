#!/usr/bin/env bash
FEAT=dino
VTA=SPD
DATASET=VCSL

# Before executing following script, similarity maps need to be calculated by test_video_sim.sh, and
# similarity maps are put in the same folder as the following "--input-root".
# For SPD that needs an addition model as input, we provide SPD model (trained on VCSL_train) links in
# data/spd_models.txt, and you can download and put it in "--spd-model-path" indicated below.

python run_video_vta.py \
        --pair-file data/pair_file_test.csv \
        --input-root result/sim/${FEAT}/${DATASET}/ \
        --input-store local \
        --batch-size 32 \
        --data-workers 0 \
        --request-workers 4 \
        --alignment-method ${VTA} \
        --output-root result/output/ \
        --result-file ${FEAT}-${VTA}-pred.json \
        --spd-model-path data/spd_models/${FEAT}.pt \
        --device cuda:0 \
        --spd-conf-thres 0.32
