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
    model_debug: bool = False
    display_debug: bool = False
    random_state: int = 42
    training_ratio: float = 0.5
    img_width: int = 4000
    img_height: int = 2250


@dataclass(repr=True, eq=True)
class Result:
    image_id: str
    cnn_accuracy: float
    nn_accuracy: float
    max_acc_ratio: str
    max_acc_ratio_score: float
    max_f1_ratio: str
    max_f1_ratio_score: float


@dataclass(repr=True, eq=True)
class MetricScores:
    precision: float = None
    recall: float = None
    accuracy: float = None
    f1: float = None
    target_score: float = None


@dataclass(repr=True, eq=True)
class ModelsMetric:
    weighted: MetricScores
    cnn: MetricScores
    nn: MetricScores


@dataclass(repr=True, eq=True)
class Dimensions:
    width: float
    height: float
    split_width: float
    split_height: float
