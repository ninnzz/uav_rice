"""Preprocessing functions."""
import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def apply_median_filter(image: np.ndarray, filter_size: int = 3) -> np.ndarray:
    """
    Median filter function.

    For this research, we opted to just use median filter
    instead of other preprocessing filter. You can always try to
    apply other preprocessing functions if you want to
    experiment more on this part of the study.

    :param image:
    :type image:
    :param filter_size:
    :type filter_size:
    :return:
    :rtype:
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = cv2.medianBlur(image, filter_size)
    return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)


def sobel_preprocess(img, thresh=1000):
    """
    I forgot why I need to use this. lol

    :param img:
    :type img:
    :param thresh:
    :type thresh:
    :return:
    :rtype:
    """
    sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    abs_sx = np.absolute(sx)
    abs_sy = np.absolute(sy)
    abs_all = abs_sx + abs_sy

    shape = abs_all.shape
    abs_all = np.reshape(abs_all, (shape[0], shape[1] * shape[2]))

    scaler = MinMaxScaler(feature_range=(0, 255))
    scaler.fit(abs_all)
    abs_all = scaler.transform(abs_all)
    abs_all = np.reshape(abs_all, shape)
    thresh = np.uint8(abs_all)

    thresh = np.uint8(cv2.threshold(abs_all, 800, 255, cv2.THRESH_BINARY)[1])
    return thresh
