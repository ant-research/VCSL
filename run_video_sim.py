import argparse
import os
import pandas as pd
import numpy as np
import json

from vcsl import *
from torch.utils.data import DataLoader
from loguru import logger

from itertools import islice


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--query-file", "-Q", type=str, help="data file")
    parser.add_argument("--reference-file", "-G", type=str, help="data file")
    parser.add_argument("--pair-file", type=str, help="data file")
    parser.add_argument("--data-file", type=str, help="data file")

    parser.add_argument("--input-store", type=str, help="store of input data: oss|local", default="oss")
    parser.add_argument("--input-root", type=str, help="root path of input data", default="")

    parser.add_argument("--oss-config", type=str, default='~/ossutilconfig-copyright', help="url path")
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="batch size")
    parser.add_argument("--data-workers", type=int, default=16, help="data workers")
    parser.add_argument("--request-workers", type=int, default=4, help="data workers")

    parser.add_argument("--output-workers", type=int, default=4, help="oss upload workers")
    parser.add_argument("--output-store", type=str, help="store of output data: oss|local")
    parser.add_argument("--output-root", type=str, help="output root")

    parser.add_argument("--similarity-type", default='cos', type=str, help="cos or chamfer")
    parser.add_argument('--consume', action='store_false', help="find exist npy file and consume")
    parser.add_argument('--device', type=int, default=0, help="cuda device, available for gpu")

    args = parser.parse_args()

    pairs, files_dict, query, reference = None, None, None, None
    bucket = create_oss_bucket(args.oss_config)
    if args.pair_file and args.data_file:
        df = pd.read_csv(args.pair_file)
        pairs = df[['query_id', 'reference_id']].values.tolist()
        files_dict = pd.read_csv(args.data_file, usecols=['uuid', 'path'], index_col='uuid')
        files_dict = {idx: r['path'] for idx, r in files_dict.iterrows()}
    else:
        query = pd.read_csv(args.query_file)
        query = query[['uuid', 'path']].values.tolist()

        reference = pd.read_csv(args.reference_file)
        reference = reference[['uuid', 'path']].values.tolist()

    config = dict()
    if args.input_store == 'oss':
        config['oss_config'] = args.oss_config

    dataset = PairDataset(query_list=query,
                          gallery_list=reference,
                          pair_list=pairs,
                          file_dict=files_dict,
                          root=args.input_root,
                          store_type=args.input_store,
                          trans_key_func=lambda x: x + ".npy",
                          data_type="numpy",
                          **config)

    logger.info(f"Data to run {len(dataset)}")

    loader = DataLoader(dataset, collate_fn=lambda x: x,
                        batch_size=args.batch_size,
                        num_workers=args.data_workers)

    model = VideoSimMapModel(concurrency=args.request_workers,
                             )

    output_store = args.input_store if args.output_store is None else args.output_store
    output_config = dict(oss_config=args.oss_config) if output_store == 'oss' else dict()
    writer_pool = AsyncWriter(pool_size=args.output_workers,
                              store_type=output_store,
                              data_type=DataType.NUMPY.type_name,
                              **output_config)

    if output_store == 'local' and not os.path.exists(args.output_root):
        os.makedirs(args.output_root, exist_ok=True)

    for batch_data in islice(loader, 0, None):
        logger.info("data cnt: {}", len(batch_data))
        batch_result = model.forward(batch_data, normalize_input=False, similarity_type=args.similarity_type, device=args.device)
        logger.info("result cnt: {}", len(batch_result))
        #
        for r_id, q_id, result in batch_result:
            key = os.path.join(args.output_root, f"{r_id}-{q_id}.npy")
            writer_pool.consume((key, result))

    writer_pool.stop()
