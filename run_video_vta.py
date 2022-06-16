#!/usr/bin/env python3
# Copyright (c) Ant Group, Inc.
"""
Codes for [CVPR2022] VCSL paper [https://github.com/alipay/VCSL].
This is the script for obtaining copied segments (i.e., temporal alignment results) between video pairs.
Before running this script, you need to go to run_video_sim.py file to get similarity map.
The hyper-parameters in these vta(video temporal alignment) methods can be directly indicated or
tuned by validation set of VCSL (recommend, details in run_video_vta_tune.py).

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
from torch.utils.data import DataLoader
from loguru import logger
from itertools import product, islice


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--query-file", "-Q", type=str, help="data file")
    parser.add_argument("--reference-file", "-G", type=str, help="data file")
    parser.add_argument("--pair-file", type=str, help="data file")

    parser.add_argument("--input-store", type=str, help="store of input data: oss|local", default="oss")
    parser.add_argument("--input-root", type=str, help="root path of input data", default="")

    parser.add_argument("--oss-config", type=str, default='~/ossutilconfig-copyright', help="url path")
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="batch size")
    parser.add_argument("--data-workers", type=int, default=16, help="data workers")
    parser.add_argument("--request-workers", type=int, default=4, help="data workers")
    parser.add_argument("--output-root", type=str, help="output root")
    parser.add_argument("--output-store", type=str, help="store of output data: oss|local")

    # Hyper parameters or input model
    parser.add_argument("--alignment-method", type=str, default="DTW", help="DTW, DP, TN alignment method")

    parser.add_argument("--min-length",  type=int, default=5, help="minimum length of one segment")
    parser.add_argument("--sum-sim", type=float, default=10., help="minimum accumulated sim of one segment")
    parser.add_argument("--ave-sim", type=float, default=0.3, help="average sim of one segment")
    parser.add_argument("--min-sim", type=float, default=0.2, help="minimum average sim of one segment")

    parser.add_argument("--max-path", type=int, default=10, help="maximum number of paths to predict")
    parser.add_argument("--discontinue", type=int, default=3, help="max discontinue point in path")
    parser.add_argument("--max-iou", type=float, default=0.3, help="max iou to filter bboxes")

    parser.add_argument("--diagonal-thres", type=int, default=10, help="threshold for discarding a vertical/horizontal part of a segment for DP")

    parser.add_argument("--tn-top-K", type=int, default=5, help="top k nearest for TN")
    parser.add_argument("--tn-max-step", type=int, default=10, help="max step for TN")

    parser.add_argument("--spd-model-path", type=str, help="SPD model path")
    parser.add_argument("--device", type=str, help="cpu or cuda:0 or others, only valid to SPD inference")
    parser.add_argument("--spd-conf-thres", type=float, default=0.5, help="bounding box conf filter for SPD inference")


    parser.add_argument("--params-file", type=str)

    parser.add_argument("--result-file", default="pred.json", type=str, help="result path")

    args = parser.parse_args()

    pairs, files_dict, query, reference = None, None, None, None
    if args.pair_file:
        df = pd.read_csv(args.pair_file)
        pairs = df[['query_id', 'reference_id']].values.tolist()

        data_list = [(f"{p[0]}-{p[1]}", f"{p[0]}-{p[1]}") for p in pairs]
    else:
        query = pd.read_csv(args.query_file)
        query = query[['uuid']].values.tolist()

        reference = pd.read_csv(args.reference_file)
        reference = reference[['uuid']].values.tolist()

        pairs = product(query, reference)
        data_list = [(f"{p[0]}-{p[1]}", f"{p[0]}-{p[1]}") for p in pairs]

    config = dict()
    if args.input_store == 'oss':
        config['oss_config'] = args.oss_config

    dataset = ItemDataset(data_list,
                          store_type=args.input_store,
                          data_type=DataType.NUMPY.type_name,
                          root=args.input_root,
                          trans_key_func=lambda x: x + '.npy',
                          **config)

    logger.info(f"Data to run {len(dataset)}")

    loader = DataLoader(dataset, collate_fn=lambda x: x,
                        batch_size=args.batch_size,
                        num_workers=args.data_workers)

    model_config = dict()
    if args.alignment_method.startswith('DTW'):
        model_config = dict(
            discontinue=args.discontinue,
            min_sim=args.min_sim,
            min_length=args.min_length,
            max_iou=args.max_iou
        )
    elif args.alignment_method.startswith('TN'):
        model_config = dict(
            tn_max_step=args.tn_max_step, tn_top_k=args.tn_top_K, max_path=args.max_path,
            min_sim=args.min_sim, min_length=args.min_length, max_iou=args.max_iou
        )
    elif args.alignment_method.startswith('DP'):
        model_config = dict(discontinue=args.discontinue,
                            min_sim=args.min_sim,
                            ave_sim=args.ave_sim,
                            min_length=args.min_length,
                            diagonal_thres=args.diagonal_thres)
    elif args.alignment_method.startswith('HV'):
        model_config = dict(min_sim=args.min_sim, iou_thresh=args.max_iou)
    elif args.alignment_method.startswith('SPD'):
        model_config = dict(model_path=args.spd_model_path,
                            conf_thresh=args.spd_conf_thres,
                            device=args.device)
    else:
        raise ValueError(f"Unknown VTA method: {args.alignment_method}")

    # override model config with param file
    if args.params_file:
        reader = build_reader(args.input_store, DataType.JSON.type_name, **config)
        param_result = reader.read(args.params_file)
        best_params = param_result['best']
        logger.info("best param {}", best_params)
        model_config = best_params['param']

    model = build_vta_model(method=args.alignment_method, concurrency=args.request_workers, **model_config)

    total_result = dict()
    for batch_data in islice(loader, 0, None):
        logger.info("data cnt: {}, {}", len(batch_data), batch_data[0][0])
        batch_result = model.forward_sim(batch_data)
        logger.info("result cnt: {}", len(batch_result))

        for pair_id, result in batch_result:
            total_result[pair_id] = result

    output_store = args.input_store if args.output_store is None else args.output_store
    if output_store == 'local' and not os.path.exists(args.output_root):
        os.makedirs(args.output_root, exist_ok=True)
    writer = build_writer(output_store, DataType.JSON.type_name, **config)
    writer.write(os.path.join(args.output_root, args.result_file), total_result)
