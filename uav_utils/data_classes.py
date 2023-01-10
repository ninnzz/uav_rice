"""
Data containers.
"""
import numpy as np
from dataclasses import dataclass


@dataclass(repr=True, eq=True)
class Rectangle:
    xmin: float
    xmax: float
    ymin: float
    ymax: float


@dataclass(repr=True, eq=True)
class LabeledData:
    img_id: str
    img_data: np.ndarray
    damage_tiles: list
    sampled_non_damage_tiles: list
    non_damage_tiles: list
    annotations: list
