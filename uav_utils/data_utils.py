"""
Data related utils.
"""
import os
import cv2
import numpy as np
from typing import List
from scipy.stats import kurtosis, skew
from uav_utils.utils import convert_xml, intersection
from uav_utils.preprocessing import apply_median_filter
from uav_utils.display import display_annotations
from uav_utils.data_classes import LabeledData, ExperimentParams, Rectangle

from IPython.display import display, HTML


def get_features(data: np.ndarray, color_avg: list) -> list:
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


def load_data(annotation_path: str, data_path: str, settings: ExperimentParams) -> List[LabeledData]:
    """
    Loads all data.

    Parameters
    ----------
    annotation_path :
    data_path :
    settings :

    Returns
    -------

    """
    json_annotations = convert_xml(annotation_path)
    images = {x["@name"]: x for x in json_annotations["annotations"]["image"]}
    img_files = images.keys()

    data_final = []

    for img_name in img_files:

        # Load image
        file_path = os.path.join(data_path, img_name)
        img_raw = cv2.imread(file_path)

        img_annotations = images[img_name]

        # Apply median filter to image
        img = apply_median_filter(img_raw, settings.median_filter_size)

        # Get size, copy image just in case want to display
        img_temp = img.copy()
        height = img.shape[0]
        width = img.shape[1]

        damage_tiles = []
        sampled_non_damage_tiles = []
        non_damage_tiles = []

        # BGR median of each image
        color_avg = [np.median(img[:, :, 0]), np.median(img[:, :, 1]), np.median(img[:, :, 2])]
        # Traverse tiles
        for y in range(0, height, settings.split_height):
            for x in range(0, width, settings.split_width):
                y1 = y + settings.split_height
                x1 = x + settings.split_width
                # Get specific tile
                tile = img[y:y1, x:x1]
                tile_rect = Rectangle(xmin=x, xmax=x1, ymin=y, ymax=y1)

                # Get tile feature
                tile_features = get_features(tile, color_avg)

                # Check every annotation
                total_tile_area = 0  # For tiles that overlaps with two damage portions
                label = 0

                for box in img_annotations['box']:

                    # Get box rectangle
                    box_rect = Rectangle(
                        xmin=float(box['@xtl']),
                        xmax=float(box['@xbr']),
                        ymin=float(box['@ytl']),
                        ymax=float(box['@ybr'])
                    )

                    # Get intersectoin
                    int_area = intersection(tile_rect, box_rect)

                    # Get get intersection of damage
                    if box['@label'] == settings.target_label:
                        total_tile_area += int_area / (settings.split_width * settings.split_height)

                    elif box['@label'] == 'no_damage_true':
                        # max coverage only
                        if int_area == (settings.split_width * settings.split_height):
                            label = 2
                            break

                if total_tile_area > settings.intersection_threshold:
                    label = 1

                if label == 1:
                    damage_tiles.append((tile_features, (x, y)))
                elif label == 2:
                    sampled_non_damage_tiles.append((tile_features, (x, y)))
                else:
                    non_damage_tiles.append((tile_features, (x, y)))

                if total_tile_area != 0:
                    b_color = int(total_tile_area * 100)
                    img_temp = cv2.rectangle(img_temp, (x, y), (x1, y1), (255 - b_color * 2, 90, b_color), -1)
                    img_temp = cv2.putText(img_temp, f'{round(total_tile_area, 4)}', (x + 10, y + 30),
                                           cv2.FONT_HERSHEY_SIMPLEX,
                                           1, (20, 100, 240), 2, cv2.LINE_AA)

        data_final.append(LabeledData(
            img_id=img_name,
            img_data=img,
            damage_tiles=np.array(damage_tiles, dtype=object),
            sampled_non_damage_tiles=np.array(sampled_non_damage_tiles, dtype=object),
            non_damage_tiles=np.array(non_damage_tiles, dtype=object),
            annotations=img_annotations['box']
        ))

        if img_name == "1_1.JPG":
            display(HTML(f"<h3>Image tile score overlay<h3>"))
            display_annotations(img_temp, img_annotations['box'])
            display(HTML(f"<h3>Image with annotation<h3>"))
            display_annotations(img, img_annotations['box'])

        return data_final
