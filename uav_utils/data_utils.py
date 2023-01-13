"""
Data related utils.
"""
import cv2
import numpy as np
from scipy.stats import kurtosis, skew
from uav_utils.data_classes import LabeledData


def get_features(data, color_avg=[255, 255, 255]) -> list:
    """
    Extract features for the NN model.

    Parameters
    ----------
    data :
    color_avg :

    Returns
    -------
    list of features
    """
    features = []
    bw = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    hist_bw = cv2.calcHist([bw], [0], None, [256], [0, 256])
    flat = hist_bw.flatten()
    features.append(kurtosis(flat))
    features.append(skew(flat))

    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([data], [i], None, [256], [0, 256])
        flat = histr.flatten()
        features.append(kurtosis(flat))
        features.append(skew(flat))

        # NOTE:
        # Some experimentation on what will happen
        # if we add more derived features
        # This is good when we have larger image patches
        # blue_avg = np.median(data[:,:,0])
        # green_avg = np.median(data[:,:,1])
        # red_avg = np.median(data[:,:,2])
        # features.append(color_avg[0] / blue_avg)
        # features.append(color_avg[1] / green_avg)
        # features.append(color_avg[2] / red_avg)
    return features


def get_test_data(data: LabeledData):
    if len(data.sampled_non_damage_tiles) == 0:
        # Edit
        if len(data.damage_tiles) == 0:
            combined = data.non_damage_tiles
        else:
            combined = np.concatenate((data.damage_tiles,
                                       data.non_damage_tiles), axis=0)
    else:
        if len(data.damage_tiles) == 0:
            combined = np.concatenate((data.sampled_non_damage_tiles,
                                       data.non_damage_tiles), axis=0)
        else:
            combined = np.concatenate((data.damage_tiles,
                                       data.sampled_non_damage_tiles,
                                       data.non_damage_tiles), axis=0)

    x_pos = combined[:, 1]
    x = combined[:, 0]
    y = ([1] * len(data.damage_tiles)) + ([0] * (len(data.sampled_non_damage_tiles) + len(data.non_damage_tiles)))

    return x, y, x_pos


