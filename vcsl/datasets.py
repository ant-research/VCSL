#!/usr/bin/env python3
# Copyright (c) Ant Group, Inc.
"""
Codes for [CVPR2022] VCSL paper [https://github.com/alipay/VCSL].
Data wraping with torch-like Dataset classes

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

from typing import Sequence, Callable
from torch.utils.data import Dataset
import base64
from .utils import *


def base64_encode_image(data: np.ndarray) -> str:
    ret, data_bytes = cv2.imencode('.png', data)
    data_base64 = str(base64.b64encode(data_bytes, b"-_"), "utf-8")
    return data_base64


class ItemDataset(Dataset):
    def __init__(self,
                 data_list: Sequence[Tuple[str, str]],
                 root: str = "",
                 store_type: str = StoreType.LOCAL.type_name,
                 data_type: str = DataType.BYTES.type_name,
                 trans_key_func: Callable = lambda x: x,
                 use_cache: bool = False,
                 **kwargs):
        self.uuids, self.files = zip(*data_list)
        self.root = root
        self.trans_key_func = trans_key_func

        self.reader = build_reader(store_type, data_type, **kwargs)

        self.use_cache = use_cache

    def read_item(self, idx):
        key = self.files[idx]
        path = self.trans_key_func(key)
        path = os.path.join(self.root, path) if self.root else path
        return self.uuids[idx], self.reader.read(path)

    def __getitem__(self, item) -> Tuple[str, Any]:
        if self.use_cache:
            logger.info(f"{os.getpid()} cache hit")
            return self.cache[item]
        else:
            return self.read_item(item)

    def __len__(self):
        return len(self.files)


class PairDataset(Dataset):
    def __init__(self,
                 query_list: Sequence[Tuple[str, str]] = None,
                 gallery_list: Sequence[Tuple[str, str]] = None,
                 pair_list: Sequence[Tuple[str, str]] = None,
                 file_dict: Dict[str, str] = None,
                 root: str = "",
                 store_type: str = StoreType.LOCAL.type_name,
                 data_type: str = DataType.BYTES.type_name,
                 trans_key_func: Callable = lambda x: x,
                 **kwargs):

        self.query_list = query_list
        self.gallery_list = gallery_list
        self.pair_list = pair_list
        self.file_dict = file_dict

        self.root = root
        self.trans_key_func = trans_key_func
        self.reader = build_reader(store_type, data_type, **kwargs)

    def __getitem__(self, item) -> Tuple[str, str, Any, Any]:
        if self.pair_list:
            query_id, gallery_id = self.pair_list[item]

            query_file = self.file_dict[query_id]
            gallery_file = self.file_dict[gallery_id]
        else:
            # iterate the product of query_list X gallery_list in row-major order
            i, j = item // len(self.gallery_list), item % len(self.gallery_list)
            query_id, query_file = self.query_list[i]
            gallery_id, gallery_file = self.gallery_list[j]

        query_file = self.trans_key_func(query_file)
        gallery_file = self.trans_key_func(gallery_file)

        query_path = os.path.join(self.root, query_file) if self.root else query_file
        gallery_path = os.path.join(self.root, gallery_file) if self.root else gallery_file

        return query_id, gallery_id, self.reader.read(query_path), self.reader.read(gallery_path)

    def __len__(self):
        return len(self.pair_list) if self.pair_list else len(self.query_list) * len(self.gallery_list)



def inter_search(val: int, interval_list: List[int]):
    low_ind, high_ind = 0, len(interval_list) - 1

    while high_ind - low_ind > 1:
        mid_ind = (low_ind + high_ind) // 2

        if val > interval_list[mid_ind]:
            low_ind = mid_ind
        elif val < interval_list[mid_ind]:
            high_ind = mid_ind
        else:
            return mid_ind
    return low_ind


class VideoFramesDataset(Dataset):
    def __init__(self, video_list: List[Tuple[str, str, int]],
                 id_to_key_fn: Callable,
                 root: str = "",
                 transforms: List[Any] = None,
                 store_type: str = StoreType.LOCAL.type_name,
                 data_type: str = DataType.IMAGE.type_name,
                 **kwargs):
        super(VideoFramesDataset, self).__init__()
        self.root = root

        self.reader = build_reader(store_type, data_type, **kwargs)

        self.id_to_key_fn = id_to_key_fn
        self.transforms = transforms
        self.video_list = video_list
        frame_cnt_list = [0, *[v[-1] for v in video_list]]
        self.offset_list = np.cumsum(frame_cnt_list).tolist()

    def __getitem__(self, item):
        video_idx, frame_idx = self.offset_to_index(item)

        vid, vdir, _ = self.video_list[video_idx]
        path = self.id_to_key_fn(vdir, frame_idx)
        path = os.path.join(self.root, path) if self.root else path
        value = self.reader.read(path)

        if self.transforms:
            for t in self.transforms:
                value = t(value)
        return vid, frame_idx, value

    def __len__(self):
        return self.offset_list[-1]

    def offset_to_index(self, offset: int) -> (int, int):
        video_idx = inter_search(offset, self.offset_list)
        # the saved frames are 0-indexed
        frame_idx = offset - self.offset_list[video_idx]
        return video_idx, int(frame_idx)

    @staticmethod
    def idx2str(idx: int):
        return f"{idx:05d}"

    @staticmethod
    def build_image_key(vdir: str, frame_idx: int):
        return os.path.join(vdir, VideoFramesDataset.idx2str(frame_idx) + ".jpg")

    @staticmethod
    def build_feature_key(vdir: str, frame_idx: int):
        return os.path.join(vdir, VideoFramesDataset.idx2str(frame_idx) + ".npy")


