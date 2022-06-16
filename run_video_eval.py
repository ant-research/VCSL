#!/usr/bin/env python3
# Copyright (c) Ant Group, Inc.
"""
Codes for [CVPR2022] VCSL paper [https://github.com/alipay/VCSL].
This is the script for evaluating the performance of video copy detection algorithm result.
Before running this script, you need to go to run_video_sim.py and run_video_vta.py files
to get similarity map and obtain copied segments (i.e., temporal alignment results).
This script will give the following three evaluation metrics:
- Overall segment-level precision/recall performance
- Video-level FRR/FAR performance
- Segment-level precision/recall performance on positive samples over query set
We recommend using the first metric to reflect segment-level alignment accuracy, while it
is also influenced by video-level results.

Please cite the following publications if you plan to use our codes or the results for your research:
{
    1. He S, Yang X, Jiang C, et al. A Large-scale Comprehensive Dataset and Copy-overlap Aware Evaluation
    Protocol for Segment-level Video Copy Detection[C]//Proceedings of the IEEE/CVF Conference on Computer
    Vision and Pattern Recognition. 2022: 21086-21095.
    2. Jiang C, Huang K, He S, et al. Learning segment similarity and alignment in large-scale content based
    video retrieval[C]//Proceedings of the 29th ACM International Conference on Multimedia. 2021: 1618-1626.
}
@author: Sifeng He and Xudong Yang
@email [sifeng.hsf@antgroup.com, jiegang.yxd@antgroup.com]
"""


import argparse
import pandas as pd
from vcsl import *
from vcsl import build_reader
from multiprocessing import Pool
from loguru import logger


def run_eval(input_dict):
    gt_box = np.array(input_dict["gt"])
    pred_box = np.array(input_dict["pred"])
    result_dict = precision_recall(pred_box, gt_box)
    result_dict["name"] = input_dict["name"]
    return result_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno-file", default="data/label_file_uuid_total.json", type=str, help="gt label file")
    parser.add_argument("--pair-group-file", default="data/split_meta_pairs.json",
                        type=str, help="meta pair corresponding relation")
    parser.add_argument("--meta-info", default="data/meta_info.json",
                        type=str, help="meta info with name and categories")
    parser.add_argument("--pred-file", type=str, help="result dir of video segment prediction")
    parser.add_argument("--pred-store", type=str, help="store of pred data: oss|local", default="oss")
    parser.add_argument("--split", type=str, default='test', help="store of pred data: train|val|test|all")
    parser.add_argument("--oss-result-file", type=str, help="save result json file on oss, endswith '.json' ")
    parser.add_argument("--local-set-csv", type=str, help="save result csv file with query set, endswith '.csv' ")
    parser.add_argument("--pool-size", type=int, default=16, help="multiprocess pool size of evaluation")
    parser.add_argument("--oss-config", type=str, default='~/ossutilconfig-copyright', help="url path")
    parser.add_argument("--oss-workers", type=int, default=1, help="oss upload workers")
    parser.add_argument("--ratio-pos-neg", type=float, default=1, help="ratio between positive and negative samples")

    args = parser.parse_args()

    split = args.split if args.split is not None else "all"
    if args.split not in ['all', 'train', 'val', 'test']:
        raise ValueError(f"Unknown dataset split {args.split}, must be one of train|val|test")

    config = dict()
    if args.pred_store == 'oss':
        config['oss_config'] = args.oss_config
    reader = build_reader(args.pred_store, "json", **config)

    logger.info(f"start loading...")

    gt = json.load(open(args.anno_file))
    key_list = [key for key in gt]

    meta_pairs = json.load(open(args.pair_group_file))

    root_dir = os.path.dirname(args.anno_file)
    split_file = os.path.join(root_dir, f"pair_file_{args.split}.csv")
    df = pd.read_csv(split_file)
    split_pairs = set([f"{q}-{r}" for q, r in zip(df.query_id.values, df.reference_id.values)])
    logger.info("{} contains pairs {}", args.split, len(split_pairs))

    key_list = [key for key in key_list if key in split_pairs]
    logger.info("Copied video data (positive) to evaluate: {}", len(key_list))

    pred_dict = reader.read(args.pred_file)
    eval_list = []
    for key in split_pairs:
        if key in gt:
            eval_list += [{"name": key, "gt": gt[key], "pred": pred_dict[key]}]
        else:
            eval_list += [{"name": key, "gt": [], "pred": pred_dict[key]}]

    logger.info(f"finish loading files, start evaluation...")

    process_pool = Pool(args.pool_size)
    result_list = process_pool.map(run_eval, eval_list)

    result_dict = {i['name']: i for i in result_list}

    if args.split != 'all':
        meta_pairs = meta_pairs[args.split]
    else:
        meta_pairs = {**meta_pairs['train'], **meta_pairs['val'], **meta_pairs['test']}

    try:
        feat, vta = args.pred_file.split('-')[:2]
    except:
        feat, vta = 'My-FEAT', 'My-VTA'

    # Evaluated result on all video pairs including positive and negative copied pairs.
    # The segment-level precision/recall can also indicate video-level performance since
    # missing or false alarm lead to decrease on segment recall or precision.
    r, p, frr, far = evaluate_micro(result_dict, args.ratio_pos_neg)
    logger.info(f"Feature {feat} & VTA {vta}: ")
    logger.info(f"Overall segment-level performance, "
                f"Recall: {r:.2%}, "
                f"Precision: {p:.2%}, "
                f"F1: {2 * r * p / (r + p):.2%}, "
                )
    logger.info(f"video-level performance, "
                f"FRR: {frr:.2%}, "
                f"FAR: {far:.2%}, "
                )

    # Evaluated macro result over each query set.
    r, p, cnt = evaluate_macro(result_dict, meta_pairs)

    logger.info(f"query set cnt {cnt}, "
                f"query macro-Recall: {r:.2%}, "
                f"query macro-Precision: {p:.2%}, "
                f"F1: {2 * r * p / (r + p):.2%}, "
                )

