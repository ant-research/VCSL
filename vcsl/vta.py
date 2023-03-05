#!/usr/bin/env python3
# Copyright (c) Ant Group, Inc.
"""
Codes for [CVPR2022] VCSL paper [https://github.com/alipay/VCSL].
Video temporal alignment (VTA) methods for video copy detection including HV/TN/DP/DTW/SPD.

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

import numpy as np
import torch
import math

from multiprocessing import Pool, set_start_method
from typing import List, Tuple, Any
from functools import partial
from .yolov5 import attempt_load, letterbox, non_max_suppression, scale_coords

from tslearn.metrics import dtw_path_from_metric
import networkx as nx
from networkx.algorithms.dag import dag_longest_path
from loguru import logger
from numba import prange, njit


def chamfer_sim_cpu(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    chamfer similarity calculation for video feature pairs implemented on CPU.
    details can be referred from ViSiL paper [https://github.com/MKLab-ITI/visil]

    Parameters
    ----------
    q: query features with dimension: (VideoLength_q, RegionLevel, FeatureDims)
    r: reference features with dimension: (VideoLength_r, RegionLevel, FeatureDims)
    In Visil feature, RegionLevel is 9 and FeatureDims is 3840

    Returns
    -------
    similarity map, dimension is (VideoLength_q, VideoLength_r)
    """
    sim = np.tensordot(q, r.T, axes=1)
    chamfer_sim_1 = np.squeeze(np.mean(np.max(sim, axis=1, keepdims=True), axis=2, keepdims=True))
    chamfer_sim_2 = np.squeeze(np.mean(np.max(sim, axis=2, keepdims=True), axis=1, keepdims=True))
    return (chamfer_sim_1 + chamfer_sim_2) / 2


def chamfer_sim_gpu(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    chamfer similarity calculation for video feature pairs implemented on GPU.
    details can be referred from ViSiL paper [https://github.com/MKLab-ITI/visil]

    Parameters
    ----------
    q: query features with dimension: (VideoLength_q, RegionLevel, FeatureDims)
    r: reference features with dimension: (VideoLength_r, RegionLevel, FeatureDims)
    In Visil feature, RegionLevel is 9 and FeatureDims is 3840

    Returns
    -------
    similarity map, dimension is (VideoLength_q, VideoLength_r)
    """
    sim = torch.tensordot(q, r.T, 1)
    chamfer_sim_1 = torch.squeeze(torch.mean(torch.amax(sim, 1, True), 2, True))
    chamfer_sim_2 = torch.squeeze(torch.mean(torch.amax(sim, 2, True), 1, True))
    return ((chamfer_sim_1 + chamfer_sim_2) / 2).cpu().numpy()


def sim_norm(sim: np.ndarray, lower_bound=0, upper_bound=0.3) -> np.ndarray:
    """
    similarity normalization after chamfer similarity.
    This chamfer similarity is not distributed between 0 and 1 and sim_norm is needed.

    Parameters
    ----------
    sim: similarity map, dimension is (VideoLength_q, VideoLength_r)
    lower_bound: lower than this value is set to 0.
    upper_bound: higher than this value is set to 1.

    Returns
    -------
    similarity map after normalization with the same dimension as input
    """
    return np.clip(sim, lower_bound, upper_bound) / (upper_bound - lower_bound)


def sim_map_cpu(qid, rid, q: np.ndarray, r: np.ndarray, normalize_input=False, similarity_type="cos") -> Tuple[str, str, np.ndarray]:
    """
    similarity map calculation (CPU) for a pair of features as input.
    the output similarity map is used as input of VTA methods.

    Parameters
    ----------
    qid: query video uuid
    rid: reference video uuid
    q: query video features
    r: reference video features
    normalize_input: normalize the features or not
    similarity_type: cosine similarity or chamfer similarity (only for ViSiL feature)

    Returns
    -------
    (qid, rid, similarity map)
    """
    if normalize_input:
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        r = r / np.linalg.norm(r, axis=1, keepdims=True)
    if similarity_type == "cos":
        return qid, rid, np.dot(q, r.T)
    elif similarity_type == "chamfer":
        return qid, rid, sim_norm(chamfer_sim_cpu(q, r))
    else:
        raise ValueError(f"Unknown method {similarity_type}")


def sim_map_gpu(qid, rid, q: np.ndarray, r: np.ndarray, normalize_input=False, similarity_type="cos", device=0) -> Tuple[str, str, np.ndarray]:
    """
    similarity map calculation (GPU) for a pair of features as input.
    the output similarity map is used as input of VTA methods.

    Parameters
    ----------
    qid: query video uuid
    rid: reference video uuid
    q: query video features
    r: reference video features
    normalize_input: normalize the features or not
    similarity_type: cosine similarity or chamfer similarity (only for ViSiL feature)

    Returns
    -------
    (qid, rid, similarity map)
    """
    with torch.cuda.device(device):
        q = torch.from_numpy(q).cuda()
        r = torch.from_numpy(r).cuda()

    if normalize_input:
        q = torch.nn.functional.normalize(q, dim=1, p=2)
        r = torch.nn.functional.normalize(r, dim=1, p=2)

    if similarity_type == "cos":
        return qid, rid, torch.matmul(q, r.T).cpu().numpy()
    elif similarity_type == "chamfer":
        return qid, rid, sim_norm(chamfer_sim_gpu(q, r))
    else:
        raise ValueError(f"Unknown method {similarity_type}")


def segment_map_to_square(similarity_map: np.ndarray, segment_choice: bool=True, ratio_thrsh: int=3, slice_ratio: int=3) -> Tuple[List[np.ndarray], List[List[int]]]:
    """
    Segment similarity map for SPD inference.
    Similarity maps with high aspect ratio between height and width (e.g. 2000 * 150) are not suitable for pattern detection.
    Segment a high aspect ratio map into several square patches

    Parameters
    ----------
    similarity_map: input similarity map, dimension is VideoLength_q * VideoLength_r
    segment_choice: whether to segment similarity image or not
    ratio_thrsh: minimum aspect ratio between height and width
    slice_ratio: height-width ration of segmented patches

    Returns
    -------
    List of segmented similarity images
    List of starting position in original similarity map
    """
    (h, w) = similarity_map.shape[:2]
    ratio = h / w
    nonzero_thresh = 1
    if 1 / ratio_thrsh < ratio < ratio_thrsh or not segment_choice:
        return [similarity_map], [[0, 0]]
    else:
        images = []
        start_locations = []
        if ratio > 1:
            max_h_ratio = math.ceil(ratio / slice_ratio) * slice_ratio
            padding = np.zeros((max_h_ratio * w, w))
            padding[:h, :w] = similarity_map
            arrays = np.vsplit(padding, int(max_h_ratio / slice_ratio))
            locs = list(range(0, max_h_ratio * w, w * slice_ratio))
            for idx, array in enumerate(arrays):
                if np.count_nonzero(array) / w > nonzero_thresh:
                    images.append(array)
                    start_locations.append([0, locs[idx]])
        else:
            max_w_ratio = math.ceil(1 / ratio / slice_ratio) * slice_ratio
            padding = np.zeros((h, max_w_ratio * h))
            padding[:h, :w] = similarity_map
            arrays = np.hsplit(padding, int(max_w_ratio / slice_ratio))
            locs = list(range(0, max_w_ratio * h, h * slice_ratio))
            for idx, array in enumerate(arrays):
                if np.count_nonzero(array) / h > nonzero_thresh:
                    images.append(array)
                    start_locations.append([locs[idx], 0])
    return images, start_locations


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess the similarity map to images that will be suitable as detection network input

    Parameters
    ----------
    image: input similarity map, dimension is (VideoLength_q, VideoLength_r)

    Returns
    -------
    float similarity map, dimension is (VideoLength_q, VideoLength_r, 3)
    """
    image = letterbox(image, new_shape=(640, 640), color=(0, 0, 0), auto=False)[0]

    if len(image.shape) == 2:
        image = np.repeat(np.expand_dims(image, axis=2), repeats=3, axis=-1)
    else:
        image = image[:, :, ::-1]

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255

    return image


def merge_detections(detection_batch: List[np.ndarray], start_coords: List[List[int]], output_thres: float=0.01) -> np.ndarray:
    """
    Merge detection results from (optional) segmented SPD results.

    Parameters
    ----------
    detection_batch: detection results from SPD model
    start_coords: segmented starting coordinates.
    output_thres: threshold with conf

    Returns
    -------
    detection results with array of (x0, y0, x1, y1, conf), i.e., (query_min, ref_min, query_max, ref_max)
    """
    total_detections = []
    for detections, origin in zip(detection_batch, start_coords):
        x0, y0 = origin
        detections[:, 0:3:2] += x0
        detections[:, 1:4:2] += y0
        detections = detections[detections[:, 4] > output_thres]
        total_detections.append(detections)
    total_detections = np.vstack(total_detections)
    return total_detections


class VideoSimMapModel(object):
    """
    Calculate similarity map for a pair of features (query/reference) as input
    optimization for GPU and multi-processing
    """
    def __init__(self, concurrency=4):
        self.concurrency = concurrency
        self.pool = Pool(self.concurrency)

    def forward(self,
                data: List[Tuple[str, str, np.array, np.array]],
                use_cuda=True,
                normalize_input=False,
                similarity_type="cos",
                device=0) -> List[Any]:
        if use_cuda:
            func = partial(sim_map_gpu, normalize_input=normalize_input, similarity_type=similarity_type, device=device)
        else:
            func = partial(sim_map_cpu, normalize_input=normalize_input, similarity_type=similarity_type)

        return self.pool.starmap(func, data)


def iou(bbox: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """
    IoU calculation for next-step filtering

    Parameters
    ----------
    bbox: bounding box array (n, 4)
    gt: bounding box array (m, 4)

    Returns
    -------
    IoU results with dimension (n, m)
    """
    if len(bbox) == 0 or len(gt) == 0:
        return np.array(0)
    lt = np.maximum(bbox[:, None, :2], gt[:, :2])  # left_top (x, y)
    rb = np.minimum(bbox[:, None, 2:], gt[:, 2:])  # right_bottom (x, y)
    wh = np.maximum(rb - lt + 1, 0)  # inter_area (w, h)
    inter_areas = wh[:, :, 0] * wh[:, :, 1]  # shape: (n, m)
    box_areas = (bbox[:, 2] - bbox[:, 0] + 1) * (bbox[:, 3] - bbox[:, 1] + 1)
    gt_areas = (gt[:, 2] - gt[:, 0] + 1) * (gt[:, 3] - gt[:, 1] + 1)
    IoU = inter_areas / (box_areas[:, None] + gt_areas - inter_areas)
    return np.array(IoU)


def zero_runs(path_diff: np.ndarray) -> np.ndarray:
    """
    obtain start and end position for a continuous invariant sequence
    Parameters
    ----------
    path_diff: aligned path position after diff

    Returns
    -------
    start and end range
    """
    # Create an array that is 1 where path_diff is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(path_diff, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def cut_path(path: np.ndarray, diagonal_thres: int) -> np.ndarray:
    """
    Remove aligned paths that move vertically or horizontally continuously
    Parameters
    ----------
    path: aligned path by DTW or DP
    diagonal_thres: discontinuity upper bound

    Returns
    -------
    path start and end location
    """
    # range [start, end)
    vertical_ranges = zero_runs(np.diff(path[:, 0]))
    vertical_ranges[:, 1] += 1
    horizontal_ranges = zero_runs(np.diff(path[:, 1]))
    horizontal_ranges[:, 1] += 1

    vertical_ranges = vertical_ranges[np.diff(vertical_ranges, axis=-1).squeeze(axis=-1) > diagonal_thres]
    horizontal_ranges = horizontal_ranges[np.diff(horizontal_ranges, axis=-1).squeeze(axis=-1) > diagonal_thres]
    discard_ranges = np.concatenate([vertical_ranges, horizontal_ranges], axis=0)
    discard_ranges = discard_ranges[discard_ranges[:, 0].argsort()]

    endpoints = discard_ranges.ravel()
    if len(endpoints) == 0:
        keep_ranges = np.array([[0, len(path)]], dtype=np.int32)
    else:
        endpoints = endpoints[1:] if endpoints[0] == 0 else np.concatenate([[0], endpoints])
        endpoints = endpoints[:-1] if endpoints[-1] == len(path) else np.concatenate([endpoints, [len(path)]])

        keep_ranges = endpoints.reshape(-1, 2)
    return keep_ranges


def dtw(sim_matrix: np.ndarray, discontinue=3, min_sim=0.2, min_length=5) -> List[List[int]]:
    """
    DTW method for video temporal alignment.
    DTW is usually adopted to match two time sequence, and we simply modify it for adaption to our task.
    Parameters
    ----------
    sim_matrix: input similarity map computed from a copied video pair.
    discontinue: max discontinued location inside each aligned segment.
    min_sim: min average similarity score for each aligned segment.
    min_length: min segment length

    Returns
    -------
    list of temporal aligned copied segments, [query_min, ref_min, query_max, ref_max] for each segment
    """
    # sim_matrix represents similarity, we need a distance matrix
    path, sim_score = dtw_path_from_metric(1 - sim_matrix, metric="precomputed")
    path = np.array(path)

    # remove horizontal and vertical paths
    keep_ranges = cut_path(path, diagonal_thres=discontinue)
    keep_ranges = keep_ranges[np.diff(keep_ranges, axis=-1).squeeze(axis=-1) > min_length]

    result_list = []
    for s, e in keep_ranges:
        assert s < e, f"{s} < {e}, {sim_matrix.shape}"
        sub_path = path[s:e]
        mean_sim = np.mean(sim_matrix[sub_path[:, 0], sub_path[:, 1]])

        # constrains: 1. average similarity score, 2. path width and height length,
        if mean_sim > min_sim and (sub_path[-1][0] - sub_path[0][0]) > min_length and (
                sub_path[-1][1] - sub_path[0][1]) > min_length:
            result_list.append(
                [int(sub_path[0][0]), int(sub_path[0][1]), int(sub_path[-1][0]), int(sub_path[-1][1])])

    return result_list


def find_path(dp_mat: np.ndarray, back_trace_mat: np.ndarray) -> np.ndarray:
    """
    find the aligned path from similarity map and backtrace matrix
    Parameters
    ----------
    dp_mat: DP matrix calculated from original DP paper
    back_trace_mat: back trace matrix

    Returns
    -------
    aligned path array for all the back traced path
    """
    max_i, max_j = np.unravel_index(np.argmax(dp_mat), dp_mat.shape)
    path = [(max_i, max_j)]

    while back_trace_mat[max_i, max_j] != -1:
        if back_trace_mat[max_i, max_j] == 0:
            max_i, max_j = max_i - 1, max_j - 1
        elif back_trace_mat[max_i, max_j] == 1:
            max_i, max_j = max_i - 1, max_j
        else:
            max_i, max_j = max_i, max_j - 1

        if dp_mat[max_i, max_j] == np.NINF:
            break

        path.append((max_i, max_j))

    path = np.array(path, dtype=np.int32)[::-1, :]
    return path


@njit()
def njit_dp_matrix(sim_mat: np.ndarray, discontinue: int=3, min_sim: float=0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    DP algorithm from Equation.5 of paper "Pattern-Based Near-Duplicate Video Retrieval and Localization on Web-Scale Videos"
    Here we use JIT to accelerate for loop calculation
    Parameters
    ----------
    sim_mat: similarity map for DP
    discontinue: max accumulated unmatch
    min_sim: minimum similarity score

    Returns
    -------
    dp_mat: DP matrix in original DP paper Eq.5
    accu_unmatch_mat: accunmulated unmatch scores
    back_trace_mat: back trace matrix for obtaining next-step aligned path
    """
    dp_mat = sim_mat.copy()

    M, N = sim_mat.shape[:2]

    accu_unmatch_mat = np.zeros(dp_mat.shape, dtype=np.int32)
    back_trace_mat = -np.ones(dp_mat.shape, dtype=np.int8)

    for i in prange(1, M):
        for j in prange(1, N):
            cand_locs = [(i - 1, j - 1), (i - 1, j), (i, j - 1)]
            # (i-1, j-1) top-left
            top_left = dp_mat[i - 1, j - 1]
            # (i-1, j) top
            top = dp_mat[i - 1, j]
            # (i, j-1) left
            left = dp_mat[i, j - 1]

            values = np.array([top_left + sim_mat[i, j], top + 0.5 * sim_mat[i, j], left + 0.5 * sim_mat[i, j]])
            max_ind = np.argmax(values)
            max_value = values[max_ind]
            prev_loc = cand_locs[max_ind]

            # sim value is too small
            unmatch = (sim_mat[i, j] < min_sim)
            if unmatch:
                accu_unmatch_mat[i, j] = accu_unmatch_mat[prev_loc] + 1

            if accu_unmatch_mat[i, j] <= discontinue:
                back_trace_mat[i, j] = max_ind
                dp_mat[i, j] = max_value
    return dp_mat, accu_unmatch_mat, back_trace_mat


def dp(sim_matrix: np.ndarray, discontinue: int = 3, min_sim: float = 1., ave_sim: float = 1.3, min_length: int = 5,
       diagonal_thres: int = 30) -> List[List[int]]:
    """
    DP method for video temporal alignment.
    Reimplemented paper:
    {Chou C L, Chen H T, Lee S Y. Pattern-based near-duplicate video retrieval and localization on web-scale videos[J].
     IEEE Transactions on Multimedia, 2015, 17(3): 382-395.}
    Parameters
    ----------
    sim_matrix: input similarity map computed from a copied video pair.
    discontinue: max accumulated unmatch in DP paper
    min_sim: minimum similarity score of each point
    ave_sim: min average similarity score for each aligned segment.
    min_length: min segment length
    diagonal_thres: max discontinued location inside each aligned segment.

    Returns
    -------
    list of temporal aligned copied segments, [query_min, ref_min, query_max, ref_max] for each segment
    """
    # rescale to make cosine-similarity scores non-negative
    sim_matrix += 1

    dp_mat, accu_unmatch_mat, back_trace_mat = njit_dp_matrix(sim_matrix, discontinue=discontinue, min_sim=min_sim)

    result_list = []
    cnt = 100
    while cnt > 0:
        path = find_path(dp_mat, back_trace_mat)

        if dp_mat[path[-1][0], path[-1][1]] == np.NINF:
            break

        r1, c1 = int(path[0][0]), int(path[0][1])
        r2, c2 = int(path[-1][0]), int(path[-1][1])
        dp_mat[r1:r2 + 1, c1:c2 + 1] = np.NINF

        keep_ranges = cut_path(path, diagonal_thres=diagonal_thres)
        keep_ranges = keep_ranges[np.diff(keep_ranges, axis=-1).squeeze(axis=-1) > min_length]
        for s, e in keep_ranges:
            sub_path = path[s:e]
            mean_sim = np.mean(sim_matrix[sub_path[:, 0], sub_path[:, 1]])

            if mean_sim > ave_sim and (sub_path[-1][0] - sub_path[0][0]) > min_length and (
                    sub_path[-1][1] - sub_path[0][1]) > min_length:
                result_list.append(
                    [int(sub_path[0][0]), int(sub_path[0][1]), int(sub_path[-1][0]), int(sub_path[-1][1])])

        cnt -= 1

    return result_list


def tn(sims: np.ndarray,
       tn_max_step: int = 10, tn_top_k: int = 5, max_path: int = 10,
       min_sim: float = 0.2, min_length: int = 5, max_iou: float = 0.3) -> List[List[int]]:
    """
    TN method for video temporal alignment.
    Reimplemented paper:
    {Tan H K, Ngo C W, Hong R, et al. Scalable detection of partial near-duplicate videos by visual-temporal consistency
     [C]//Proceedings of the 17th ACM international conference on Multimedia. 2009: 145-154.}
    Parameters
    ----------
    sims: input similarity map computed from a copied video pair.
    tn_max_step: max step range in TN.
    tn_top_k: Top k frame similarity selection in TN.
    max_path: max loop for multiply segments detection.
    min_sim: min average similarity score for each aligned segment.
    min_length: min segment length.
    max_iou: max iou for filtering overlap segments (bbox).

    Returns
    -------
    list of temporal aligned copied segments, [query_min, ref_min, query_max, ref_max] for each segment

    """
    infringe_box_list = []
    path = 0
    node_pair2id = {}
    node_pair2id[(-1, -1)] = 0

    node_id2pair = {}
    node_id2pair[0] = (-1, -1)  # source

    node_num = 1

    DG = nx.DiGraph()
    DG.add_node(0)

    # get top-k values and indices, shape (Q_LEN, top_k)
    top = min(tn_top_k, sims.shape[1])

    topk_indices = np.argsort(-sims)[:, :top]
    topk_sims = np.take_along_axis(sims, topk_indices, axis=-1)

    # add nodes
    for qf_idx in range(sims.shape[0]):
        for k in range(top):
            rf_idx = topk_indices[qf_idx][k]

            node_id2pair[node_num] = (qf_idx, rf_idx)
            node_pair2id[(qf_idx, rf_idx)] = node_num

            DG.add_node(node_num)
            node_num += 1

    # create graph by adding edges
    for q_i in range(sims.shape[0]):
        r_i = topk_indices[q_i]

        intermediate_rs = np.empty((0,), dtype=np.int32)
        # implements Constraints C1 by limiting range end
        for q_j in range(q_i + 1, min(sims.shape[0], q_i + tn_max_step)):
            r_j = topk_indices[q_j]  # shape (top_k, )

            r_diff = r_j[:, None] - r_i  # dst - src, shape (top_k, top_k)

            # Constraints C2
            C2 = (r_diff > 0) & (r_diff < tn_max_step)

            # Constraints C3
            if len(intermediate_rs) == 0:
                C3 = np.ones(C2.shape, dtype=np.bool)
            else:
                # "the equal sign" in C3 in paper is wrong because it's contradictory to C2
                cond1 = intermediate_rs[None, :] > r_i[:, None]
                cond2 = intermediate_rs[None, :] < r_j[:, None]
                C3 = np.sum(cond2[:, None, :] & cond1, axis=-1) == 0

            # Constraints C4
            s_j = topk_sims[q_j]  # shape (top_k, )
            s_j = np.repeat(s_j.reshape(-1, 1), r_diff.shape[1], axis=1)  # shape (top_k, top_k)
            C4 = s_j >= min_sim

            val_rows, val_cols = np.where(C2 & C3 & C4)
            val_sims = s_j[val_rows, val_cols]
            # update intermediate_rs
            valid_r_j = r_j[val_rows]
            intermediate_rs = np.unique(np.concatenate([intermediate_rs, valid_r_j]))

            edges = [(node_pair2id[(q_i, r_i[c])], node_pair2id[(q_j, r_j[r])], dict(weight=s))
                     for c, r, s in zip(val_cols, val_rows, val_sims)]

            DG.add_edges_from(edges)

    logger.info("Graph N {} E {} for sim {}x{}", DG.number_of_nodes(), DG.number_of_edges(), sims.shape[0],
                sims.shape[1])

    # link sink node
    for i in range(0, node_num - 1):
        j = node_num - 1

        pair_i = node_id2pair[i]
        pair_j = node_id2pair[j]

        if (pair_j[0] > pair_i[0] and pair_j[1] > pair_i[1] and
                pair_j[0] - pair_i[0] <= tn_max_step and pair_j[1] - pair_i[1] <= tn_max_step):
            DG.add_edge(i, j, weight=0)

    while True:
        if path > max_path:
            break
        longest_path = dag_longest_path(DG)
        for i in range(1, len(longest_path)):
            DG.add_edge(longest_path[i - 1], longest_path[i], weight=0.0)
        if 0 in longest_path:
            longest_path.remove(0)  # remove source node
        if node_num - 1 in longest_path:
            longest_path.remove(node_num - 1)  # remove sink node
        path_query = [node_id2pair[node_id][0] for node_id in longest_path]
        path_refer = [node_id2pair[node_id][1] for node_id in longest_path]

        if len(path_query) == 0:
            break
        score = 0.0
        for (qf_idx, rf_idx) in zip(path_query, path_refer):
            score += sims[qf_idx][rf_idx]
        if score > 0:
            query_min, query_max = min(path_query), max(path_query)
            refer_min, refer_max = min(path_refer), max(path_refer)
        else:
            query_min, query_max = 0, 0
            refer_min, refer_max = 0, 0
        ave_length = (refer_max - refer_min + query_max - query_min) / 2
        ious = iou(np.expand_dims(np.array([query_min, refer_min, query_max, refer_max]), axis=0),
                   np.array(infringe_box_list))
        if score / ave_length > min_sim and min(refer_max - refer_min,
                                                query_max - query_min) > min_length and ious.max() < max_iou:
            infringe_box_list.append([int(query_min), int(refer_min), int(query_max), int(refer_max)])
        path += 1
    return infringe_box_list


def hv(sims: np.ndarray,
       iou_thresh: float=0.9,
       min_sim: float=0.2, max_peaks: int=100) -> List[List[int]]:
    """
    HV method for video temporal alignment.
    Reimplemented paper:
    {Douze M, Jégou H, Schmid C. An image-based approach to video copy detection with spatio-temporal post-filtering[J].
     IEEE Transactions on Multimedia, 2010, 12(4): 257-266.}
    Parameters
    ----------
    sims: input similarity map computed from a copied video pair.
    iou_thresh: max iou for filtering overlap segments (bbox).
    min_sim: min average similarity score for each aligned segment.
    max_peaks: max selected top K score.

    Returns
    -------
    list of temporal aligned copied segments, [query_min, ref_min, query_max, ref_max] for each segment

    """
    infringe_box_list = []

    ## step1: remove all pairs lower than min_sim
    sims[sims < min_sim] = 0.

    ## step2: calculate the time_bins histogram
    query_inds, refer_inds = np.where(sims >= min_sim)
    sigma_inds = np.unique(refer_inds - query_inds)
    sigma_hists = dict()
    for s_i in range(sigma_inds.shape[0]):
        sigma = sigma_inds[s_i]
        if sigma not in sigma_hists:
            sigma_hists[sigma] = dict()
            sigma_hists[sigma]['score'] = 0.
            sigma_hists[sigma]['matches'] = list()
        start_idx = -sigma if sigma < 0 else 0
        end_idx = sims.shape[1] - sigma  # if sigma>0 else sims.shape[1] - sigma
        end_idx = min(max(end_idx, 0), sims.shape[0])
        query_idx = range(start_idx, end_idx)
        refer_idx = range(start_idx + sigma, end_idx + sigma)
        sub_sims = sims[query_idx, refer_idx]
        sigma_hists[sigma]['score'] = float(np.sum(sub_sims))
        sigma_hists[sigma]['matches'] = [[query_idx[x], refer_idx[x], sub_sims[x]] for x in range(len(query_idx))]

    ## step3: refine the final matches
    sorted_sigma_hists = sorted(sigma_hists.items(), key=lambda x: x[1]['score'], reverse=True)
    del sigma_hists
    sorted_sigma_hists = sorted_sigma_hists[:max_peaks]

    ## step4: output the final infringe_box_list
    for sigma, sum in sorted_sigma_hists:
        if sum['score'] <= 0.: continue
        matches = sum['matches']
        query_ids = [x[0] for x in matches]
        refer_ids = [x[1] for x in matches]
        query_min = min(query_ids)
        query_max = max(query_ids)
        refer_min = min(refer_ids)
        refer_max = max(refer_ids)
        cur_box = [int(query_min), int(refer_min), int(query_max), int(refer_max)]
        ## add nms
        ious = iou(np.expand_dims(cur_box, axis=0), np.array(infringe_box_list, dtype=np.float32))
        if np.any(ious > iou_thresh): continue
        infringe_box_list.append(cur_box)
    return infringe_box_list


def spd(sims: np.ndarray, model, conf_thresh: float, device: str, iou_thresh=0.3, infer_batch: int=1) -> List[List[int]]:
    """
    SPD method for video temporal alignment.
    Original paper:
    {Jiang C, Huang K, He S, et al. Learning segment similarity and alignment in large-scale content based video retrieval
     [C]//Proceedings of the 29th ACM International Conference on Multimedia. 2021: 1618-1626.}
    Parameters
    ----------
    sims: input similarity map computed from a copied video pair.
    model: SPD model. (implement with YOLOv5)
    conf_thresh: min confidence score for filtering segments (bbox).
    device: model inference on device name. e.g. cuda:0 or cpu
    iou_thresh: max iou for filtering overlap segments (bbox).
    infer_batch: inference batch size

    Returns
    -------
    list of temporal aligned copied segments, [query_min, ref_min, query_max, ref_max] for each segment

    """
    similarity = sims.clip(0, 1)
    similar_img = np.where(np.argsort(np.argsort(similarity)) >= similarity.shape[1] - 20, similarity, 0)

    image_batch, start_coords = segment_map_to_square(similar_img)

    old_shapes = [img.shape[:2] for img in image_batch]

    # N x C x H x W
    image_batch = np.array([preprocess_image(image) for image in image_batch]).transpose([0, 3, 1, 2])

    new_shapes = [image_batch.shape[2:] for _ in range(len(image_batch))]

    infer_cnt = (len(image_batch) + infer_batch - 1) // infer_batch

    total_result = []
    for idx in range(infer_cnt):
        data = image_batch[idx * infer_batch: (idx + 1) * infer_batch, ...]
        # Inference
        data = torch.from_numpy(data).to(device, dtype=torch.float)
        pred = model(data, augment=False)[0]
        detections = non_max_suppression(pred, conf_thresh, iou_thresh, agnostic=False)
        # transform coords
        for d_idx, det in enumerate(detections):
            new_shape = new_shapes[idx * infer_batch + d_idx]
            old_shape = old_shapes[idx * infer_batch + d_idx]
            det[:, :4] = scale_coords(new_shape, det[:, :4], old_shape).round()

        total_result.extend([det.cpu().numpy() for det in detections])

    # merge detection results
    total_detections = merge_detections(total_result, start_coords)
    infringe_list = []
    for detection_box in total_detections:
        xmin = int(detection_box[0])
        xmax = int(detection_box[2])
        ymin = int(detection_box[1])
        ymax = int(detection_box[3])
        conf = float(detection_box[4])
        if conf > conf_thresh:
            infringe_list.append([ymin, xmin, ymax, xmax])
    return infringe_list


def func_wrapper_with_exception(rid, item, func):
    try:
        return rid, func(item)
    except Exception as e:
        logger.exception("Fail to run with rid {}", rid)
        raise RuntimeError(f"Fail to run with data with {rid}") from e


class BaseVtaModel(object):
    """
    Wrapper of base video temporal alignment (VTA) model
    """
    def __init__(self, concurrency, func_to_run):
        self.pool = Pool(concurrency)
        self.func_to_run = func_to_run

    def forward(self, data: List[Tuple[str, str, np.array, np.array]]) -> List[Any]:
        sim_func = partial(sim_map_cpu, normalize_input=False)
        sim_list = self.pool.map(sim_func, data)
        sim_list = [(f"{q}-{r}", v) for q, r, v in sim_list]
        return self.forward_sim(sim_list)

    def forward_sim(self, data: List[Tuple[str, np.array]]) -> List[Any]:
        algo = partial(func_wrapper_with_exception, func=self.func_to_run)
        results = self.pool.starmap(algo, data)
        return results


class DtwModel(BaseVtaModel):
    def __init__(self,
                 concurrency=4,
                 version="v1",
                 discontinue=3,
                 min_sim=0.2,
                 min_length=5,
                 max_iou=0.3):
        self.min_length = min_length
        self.min_sim = min_sim
        self.max_iou = max_iou
        self.discontinue = discontinue
        self.version = version
        func = partial(dtw,
                       discontinue=self.discontinue,
                       min_sim=self.min_sim,
                       min_length=self.min_length)

        super(DtwModel, self).__init__(concurrency=concurrency, func_to_run=func)


class DpVtaModel(BaseVtaModel):
    def __init__(self,
                 concurrency=4,
                 version="v1",
                 discontinue=3,
                 min_sim=0., min_length=5, max_iou=0.3, sum_sim=8, ave_sim=0.3, diagonal_thres=10):
        self.min_sim = min_sim
        self.min_length = min_length
        self.max_iou = max_iou
        self.sum_sim = sum_sim
        self.ave_sim = ave_sim
        self.discontinue = discontinue
        self.diagonal_thres = diagonal_thres
        self.version = version

        func = partial(dp,
                       discontinue=self.discontinue,
                       min_sim=self.min_sim, min_length=self.min_length,
                       ave_sim=self.ave_sim, diagonal_thres=self.diagonal_thres)

        super(DpVtaModel, self).__init__(concurrency=concurrency, func_to_run=func)


class TnVtaModel(BaseVtaModel):
    def __init__(self,
                 concurrency=4,
                 version="v1",
                 tn_max_step=10, tn_top_k=5, max_path=10,
                 min_sim=0.2, min_length=5, max_iou=0.3):
        self.tn_max_step = tn_max_step
        self.tn_top_k = tn_top_k
        self.max_path = max_path

        self.min_sim = min_sim
        self.min_length = min_length
        self.max_iou = max_iou

        self.version = version

        func = partial(tn,
                       tn_max_step=self.tn_max_step, tn_top_k=self.tn_top_k, max_path=self.max_path,
                       min_sim=self.min_sim, min_length=self.min_length, max_iou=self.max_iou)
        super(TnVtaModel, self).__init__(concurrency=concurrency, func_to_run=func)


class HvVtaModel(BaseVtaModel):
    def __init__(self,
                 concurrency=4,
                 version="v1",
                 iou_thresh=0.9,
                 min_sim=0.0, min_bins=1, max_peaks=100, min_peaks=10):
        self.min_sim = min_sim
        self.min_bins = min_bins
        self.max_peaks = max_peaks
        self.min_peaks = min_peaks
        self.iou_thresh = iou_thresh
        self.version = version

        func = partial(hv,
                       iou_thresh=self.iou_thresh,
                       min_sim=self.min_sim,
                       max_peaks=self.max_peaks)

        super(HvVtaModel, self).__init__(concurrency=concurrency, func_to_run=func)


class SpdVtaModel(BaseVtaModel):
    def __init__(self,
                 concurrency=4,
                 version="v1",
                 model_path=None,
                 conf_thresh=0.5,
                 device="cuda:0",
                 iou_thresh=0.3,
                 infer_batch=1
                 ):
        self.version = version
        self.conf_thresh = conf_thresh
        self.device = device
        self.iou_thresh = iou_thresh
        self.infer_batch = infer_batch
        model = attempt_load(model_path, map_location=self.device)
        func = partial(spd,
                       model=model,
                       conf_thresh=self.conf_thresh,
                       device=self.device,
                       iou_thresh=self.iou_thresh,
                       infer_batch=self.infer_batch)
        try:
            if self.device.startswith("cuda"):
                set_start_method('spawn')
        except RuntimeError as e:
            logger.info(f"{e} on starting spawn method for SPD.")
            pass
        super(SpdVtaModel, self).__init__(concurrency=concurrency, func_to_run=func)


def build_vta_model(method="DTW", concurrency=4, **config) -> BaseVtaModel:
    if method == 'DTW':
        return DtwModel(concurrency=concurrency, version="v1", **config)
    elif method == 'TN':
        return TnVtaModel(concurrency=concurrency, version="v1", **config)
    elif method == 'DP':
        return DpVtaModel(concurrency=concurrency, version="v1", **config)
    elif method == 'HV':
        return HvVtaModel(concurrency=concurrency, version="v1", **config)
    elif method == "SPD":
        return SpdVtaModel(concurrency=concurrency, version="v1", **config)
    else:
        raise ValueError(f"Unknown method {method}")
