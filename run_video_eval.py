import argparse
import pandas as pd
import os
import json

from vcsl import *
from itertools import islice
from vcsl import build_reader
from multiprocessing import Pool
from loguru import logger


def gen_input(key):
    return {"name": key, "gt": gt[key], "pred": json.loads(reader.read(os.path.join(args.pred_root, key + ".json")))}


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

    args = parser.parse_args()

    split = args.split if args.split is not None else "all"
    if args.split not in ['all', 'train', 'val', 'test']:
        raise ValueError(f"Unknown dataset split {args.split}, must be one of train|val|test")

    config = dict()
    if args.pred_store == 'oss':
        config['oss_config'] = args.oss_config
    # reader = build_reader(args.pred_store, "bytes", **config)
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
    logger.info("data to evaluate: {}", len(key_list))

    pred_dict = reader.read(args.pred_file)
    eval_list = [{"name": key, "gt": gt[key], "pred": pred_dict[key]} for key in key_list]

    logger.info(f"finish loading files, start evaluation...")

    process_pool = Pool(args.pool_size)
    result_list = process_pool.map(run_eval, eval_list)

    result_dict = {i['name']: i for i in result_list}

    if args.split != 'all':
        meta_pairs = meta_pairs[args.split]
    else:
        meta_pairs = {**meta_pairs['train'], **meta_pairs['val'], **meta_pairs['test']}

    r, p, cnt = evaluate(result_dict, meta_pairs)

    try:
        feat, vta = args.pred_file.split('-')[:2]
    except:
        feat, vta = 'My-FEAT', 'My-VTA'
    logger.info(f"query set cnt {cnt}, "
                f"query macro-Recall: {r:.2%}, "
                f"query macro-Precision: {p:.2%}, "
                f"F1: {2 * r * p / (r + p):.2%}, "
                )

