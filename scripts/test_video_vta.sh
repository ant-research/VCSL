#!/usr/bin/env bash

## Before executing following script, similarity maps need to be calculated by test_video_sim.sh, and
## similarity maps are put in the same folder as the following "--input-root".
## For SPD that needs an addition model as input, we provide SPD model (trained on VCSL_train) links in
## data/spd_models.txt, and you can download and put it in "--spd-model-path" indicated below.

#FEAT=ISC
#VTA=SPD
#DATASET=VCSL
#
#python run_video_vta.py \
#        --pair-file data/pair_file_test.csv \
#        --input-root result/sim/${FEAT}/${DATASET}/ \
#        --input-store local \
#        --batch-size 32 \
#        --data-workers 0 \
#        --request-workers 4 \
#        --alignment-method ${VTA} \
#        --output-root result/output/ \
#        --result-file ${FEAT}-${VTA}-pred-new.json \
#        --spd-model-path data/spd_models/${FEAT}.pt \
#        --device cuda:0 \
#        --spd-conf-thres 0.1

FEAT=ISC
VTA=DTW
DATASET=VCSL

python run_video_vta.py \
        --pair-file data/pair_file_test.csv \
        --input-root result/sim/${FEAT}/${DATASET}/ \
        --input-store local \
        --batch-size 32 \
        --data-workers 0 \
        --request-workers 4 \
        --alignment-method ${VTA} \
        --output-root result/output/ \
        --result-file ${FEAT}-${VTA}-pred-new.json \
        --discontinue 9 \
        --min-sim 0.3 \
        --min-length 5

#FEAT=ISC
#VTA=TN
#DATASET=VCSL
#
#python run_video_vta.py \
#        --pair-file data/pair_file_test.csv \
#        --input-root result/sim/${FEAT}/${DATASET}/ \
#        --input-store local \
#        --batch-size 32 \
#        --data-workers 0 \
#        --request-workers 4 \
#        --alignment-method ${VTA} \
#        --output-root result/output/ \
#        --result-file ${FEAT}-${VTA}-pred-new.json \
#        --tn-max-step 10 \
#        --tn-top-K 3 \
#        --min-sim 0.3


#FEAT=ISC
#VTA=HV
#DATASET=VCSL
#
#python run_video_vta.py \
#        --pair-file data/pair_file_test.csv \
#        --input-root result/sim/${FEAT}/${DATASET}/ \
#        --input-store local \
#        --batch-size 32 \
#        --data-workers 0 \
#        --request-workers 4 \
#        --alignment-method ${VTA} \
#        --output-root result/output/ \
#        --result-file ${FEAT}-${VTA}-pred-new.json \
#        --min-sim 0.7 \
#        --max-iou 0.9

#FEAT=ISC
#VTA=DP
#DATASET=VCSL
#
#python run_video_vta.py \
#        --pair-file data/pair_file_test.csv \
#        --input-root result/sim/${FEAT}/${DATASET}/ \
#        --input-store local \
#        --batch-size 32 \
#        --data-workers 0 \
#        --request-workers 4 \
#        --alignment-method ${VTA} \
#        --output-root result/output/ \
#        --result-file ${FEAT}-${VTA}-pred-new.json \
#        --discontinue 9 \
#        --min-sim 0.2 \
#        --ave-sim 1.3 \
#        --min-length 5 \
#        --diagonal-thres 10