"""Utility functions."""
import json

import numpy as np
import xmltodict
from uav_utils.data_classes import Rectangle


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
