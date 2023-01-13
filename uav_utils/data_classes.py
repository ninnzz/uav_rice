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
    damage_tiles: np.ndarray
    sampled_non_damage_tiles: np.ndarray
    non_damage_tiles: np.ndarray
    annotations: list


@dataclass(repr=True, eq=True)
class ExperimentParams:
    target_label: str
    split_width: int
    split_height: int
    intersection_threshold: float
    median_filter_size: int
