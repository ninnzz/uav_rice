"""Utility functions."""
import os
import json
import datetime
import xmltodict
import numpy as np
from uav_utils.data_classes import Rectangle, ExperimentParams


def gen_folder(p: ExperimentParams, save_path: str) -> str:
    """
    Generate foldername based on experiment parameters.

    Parameters
    ----------
    p :
    save_path :
    Returns
    -------

    """
    ts = datetime.datetime.utcnow().timestamp()
    folder_name = f"experiment_{p.target_label}_{p.split_width}by{p.split_height}_{p.training_ratio}_{ts}"

    folder = os.path.join(save_path, folder_name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    return folder


def intersection(r1: Rectangle, r2: Rectangle) -> float:
    """
    Returns the area of intersection for rectangles.

    Parameters
    ----------
    r1 :
    r2 :

    Returns
    -------

    """
    dx = min(r1.xmax, r2.xmax) - max(r1.xmin, r2.xmin)
    dy = min(r1.ymax, r2.ymax) - max(r1.ymin, r2.ymin)
    if dx >= 0 and dy >= 0:
        return dx * dy

    return 0


def convert_xml(file_path: str) -> dict:
    """
    Convert xml file to dict.

    Parameters
    ----------
    file_path :

    Returns
    -------

    """
    with open(file_path) as xml_file:
        data_dict = xmltodict.parse(xml_file.read())

    return json.loads(json.dumps(data_dict))


def convert_to_tiles(data: list, img: np.ndarray, height: int = 50, width: int = 50):
    """
    Split images to tiles.

    Parameters
    ----------
    data :
    img :
    height :
    width :

    Returns
    -------

    """
    converted = []
    for x, y in data:
        y1 = y + height
        x1 = x + width
        tile = img[y:y1, x:x1]
        converted.append(tile)

    return converted

#
# def compute_threshold(data: np.ndarray) -> tuple:
#     """
#     Computes the optimal threshold base on training results.
#
#     Parameters
#     ----------
#     data :
#
#     Returns
#     -------
#
#     """
#     f1 = 0
#     acc = 0
#     f1_thresh = 0
#     acc_thresh = 0
#     # Sort data
#     sorted_data = list(data)
#     sorted_data = sorted(sorted_data, key=lambda i: i[0], reverse=True)
#     sorted_data = np.array(sorted_data)
#
#     fn_counter = int(sum(sorted_data[:, 1]))
#     tn_counter = len(sorted_data) - fn_counter
#     tp_counter = 0
#     fp_counter = 0
#     metrics = []
#     _small_num = 0.0000000001
#     for score, label in sorted_data:
#         if label == 1:
#             tp_counter += 1
#             fn_counter -= 1
#         else:
#             fp_counter += 1
#             tn_counter -= 1
#
#         # True pos, false pose, true neg, false neg, accuracy, fq
#         # metrics.append((tp_counter, fp_counter, tn_counter, fn_counter))
#         # Compute f1, prec, recall
#         _prec = tp_counter / (tp_counter + fp_counter + _small_num)
#         _rec = tp_counter / (tp_counter + fn_counter + _small_num)
#         _f1 = 2 * (_prec * _rec) / (_prec + _rec + _small_num)
#         _acc = (tp_counter + tn_counter) / (tp_counter + tn_counter + fp_counter + fn_counter)
#
#         if acc <= _acc:
#             acc = _acc
#             acc_thresh = score
#
#         if f1 <= _f1:
#             f1 = _f1
#             f1_thresh = score
#
#     return acc, acc_thresh, f1, f1_thresh
