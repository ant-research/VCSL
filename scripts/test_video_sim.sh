#!/usr/bin/env bash
FEAT=dino
DATASET=VCSL
SPLIT=test

# Before executing following script, you need to put video frame features in root "result/${FEAT}/${DATASET}/".
# You can download videos in "data/videos_url_uuid.csv" and extract frames and features by yourself,
# or you can directly download extracted features in "data/vcsl_features.txt". All the feature npy files
# need to be put under folder "--input-root" indicated below. The folder can be located in local dir or
# aliyun OSS(https://www.aliyun.com/product/oss).

python run_video_sim.py \
       --pair-file data/pair_file_${SPLIT}.csv \
       --data-file data/frames_all.csv \
       --input-root result/features/${FEAT}/${DATASET}/ \
       --input-store local \
       --output-root result/sim/${FEAT}/${DATASET}/ \
       --output-store local \
       --data-workers 32 \
       --request-workers 8