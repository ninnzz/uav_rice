"""
Post processing functions.
"""
import numpy as np
from uav_utils.data_classes import Dimensions


def remove_island(y_pred: list, y_test: list, pos: list, d: Dimensions):
    """Remove predicted damage tiles with no neigbhor."""
    _ypred = []
    map_list = np.zeros((int(d.height / d.split_height) + 2, int(d.width / d.split_width) + 2), dtype=int)

    # Build mapping of tiles
    for item in zip(y_pred, y_test, pos):
        x = int(item[2][0] / d.split_width) + 1
        y = int(item[2][1] / d.split_height) + 1
        map_list[y][x] = item[0]

    for item in zip(y_pred, y_test, pos):
        x = int(item[2][0] / d.split_width) + 1
        y = int(item[2][1] / d.split_height) + 1

        if map_list[y][x] == 0:
            _ypred.append(0)
            continue

        s = 0

        bounds = [
            (0, -1), (0, 1), (-1, 0), (1, 0)
        ]

        for _y, _x in bounds:
            try:
                s += map_list[y + _y][x + _x]
            except:
                pass

        if s > 0:
            _ypred.append(1)
            continue

        _ypred.append(0)

    return _ypred
