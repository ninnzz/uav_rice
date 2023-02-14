"""
Contains all graph/display related functions.
"""

import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from uav_utils.postprocess import remove_island
from uav_utils.data_classes import AnalysisSettings, DisplaySettings
from uav_utils.utils import convert_xml


def get_img_annotations(file_path: str) -> dict:
    json_annotations = convert_xml(file_path)
    images = {x["@name"]: x for x in json_annotations["annotations"]["image"]}
    return images


def display_annotations(img: np.ndarray, boxes: list, target_label: str = "damage"):
    """
    Display image annotations.

    Parameters
    ----------
    img :
    boxes :
    target_label :

    Returns
    -------

    """
    for box in boxes:
        if box['@label'] == target_label:
            start = (int(float(box["@xtl"])), int(float(box["@ytl"])))
            end = (int(float(box["@xbr"])), int(float(box["@ybr"])))

            img = cv2.rectangle(img, start, end, (20, 100, 240), 5)

        # Add display for no damage box
        # elif box['@label'] == 'no_damage_true':
        #     start = (int(float(box["@xtl"])), int(float(box["@ytl"])))
        #     end = (int(float(box["@xbr"])), int(float(box["@ybr"])))

        #     img = cv2.rectangle(img, start, end, (240, 50, 10), 5)

    plt.figure(figsize=(70, 40))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), interpolation='nearest')
    plt.show()


def show_result(fname: str, no_island: bool, settings: AnalysisSettings, ds: DisplaySettings):
    """
    Get result for image display.
    
    Parameters
    ----------
    fname :
    no_island :
    settings :
    ds :

    Returns
    -------

    """

    data = pd.read_csv(fname, names=settings.display_columns)

    weighted = (ds.cnn_ratio / 100) * data.cnn + (ds.nn_ratio / 100) * data.nn
    data["weighted"] = weighted
    data["weighted_predicted_label"] = data.apply(lambda row: 0 if row.weighted < ds.ensm_thresh else 1, axis=1)
    data["cnn_predicted_label"] = data.apply(lambda row: 0 if row.cnn < ds.cnn_thresh else 1, axis=1)
    data["nn_predicted_label"] = data.apply(lambda row: 0 if row.nn < ds.nn_thresh else 1, axis=1)

    pos = []

    if no_island:
        for x, y in zip(data["pos_x"], data["pos_y"]):
            pos.append((int(x), int(y)))

        data["weighted_predicted_label"] = remove_island(data.weighted_predicted_label.tolist(),
                                                         data.label.tolist(), pos, settings.dimension)
        data["cnn_predicted_label"] = remove_island(data.cnn_predicted_label.tolist(), data.label.tolist(),
                                                    pos, settings.dimension)
        data["nn_predicted_label"] = remove_island(data.nn_predicted_label.tolist(), data.label.tolist(),
                                                   pos, settings.dimension)

    return data
