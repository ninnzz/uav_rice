"""
Function that trains the model
"""
import cv2
import random
import numpy as np

from IPython.display import display, HTML
from uav_utils.data_classes import ExperimentParams, LabeledData
from uav_utils.data_utils import get_test_data
from uav_utils.utils import convert_to_tiles
from uav_utils.models import create_cnn_model, create_nn_model
from uav_utils.display import display_annotations


def balance_data(data: LabeledData, percent: float = None):
    """
    Returns proper ratio of each data needed for training.

    Parameters
    ----------
    data :
    percent :

    Returns
    -------

    """
    ng = []
    ng_pos = []
    ok = []
    ok_pos = []
    true_ok = []
    true_ok_pos = []

    print(f"Image: {data.img_id}")
    ng_count = len(data.damage_tiles)
    sample_ok_count = len(data.sampled_non_damage_tiles)
    ok_count = len(data.non_damage_tiles)

    total_ok = int(ng_count / percent) - ng_count

    needed_ok = total_ok - sample_ok_count

    print(f"Set Training Ratio: {percent}")

    # No ratio selected, just use everything
    if percent is None:
        if len(data.damage_tiles) != 0:
            ng = data.damage_tiles[:, 0].tolist()
            ng_pos = data.damage_tiles[:, 1].tolist()
        if sample_ok_count != 0:
            true_ok = data.sampled_non_damage_tiles[:, 0].tolist()
            true_ok_pos = data.sampled_non_damage_tiles[:, 1].tolist()
        ok = data.non_damage_tiles[:, 0].tolist()
        ok_pos = data.non_damage_tiles[:, 1].tolist()
        y = ([1] * len(ng)) + ([0] * (len(true_ok) + len(ok)))
        print(
            f"Damaged: {len(ng)}/{ng_count}, "
            f"True No Damage: {len(true_ok)}/{sample_ok_count}, "
            f"No Damage: {len(ok)}/{ok_count}")

        return (ng + true_ok + ok), y, (ng_pos + true_ok_pos + ok_pos)

    # Fixed
    if sample_ok_count != 0:
        true_ok = data.sampled_non_damage_tiles[:, 0].tolist()
        true_ok_pos = data.sampled_non_damage_tiles[:, 1].tolist()

    if needed_ok <= 0:
        ok = []
    else:
        if needed_ok >= ok_count:
            ok = data.non_damage_tiles[:, 0].tolist()
            ok_pos = data.non_damage_tiles[:, 1].tolist()
        else:
            _ok = random.choices(data.non_damage_tiles, k=needed_ok)
            _ok = np.array(_ok)
            ok = _ok[:, 0].tolist()
            ok_pos = _ok[:, 1].tolist()

    # Fixed
    print(data.damage_tiles.shape)
    if len(data.damage_tiles) != 0:
        ng = data.damage_tiles[:, 0].tolist()
        ng_pos = data.damage_tiles[:, 1].tolist()

    print(
        f"Damaged: {len(ng)}/{ng_count}, True No Damage: {len(true_ok)}/{sample_ok_count}, No Damage: {len(ok)}/{ok_count}")

    y = ([1] * len(ng)) + ([0] * (len(true_ok) + len(ok)))

    return (ng + true_ok + ok), y, (ng_pos + true_ok_pos + ok_pos)


def train(data: list, settings: ExperimentParams,
          display_image=False, file_filter: list = None,
          save_path: str = "/tmp/uav_results"):
    """
    Main training function for the models.

    Parameters
    ----------
    save_path :
    data :
    settings :
    display_image :
    file_filter :

    Returns
    -------

    """
    # For image display opacity
    alpha = 0.5

    _true_damage_percent = []
    _pred_damage_percent = []
    total_boxes = int(settings.img_height / settings.split_height) * int(settings.img_width / settings.split_width)

    for i in range(len(data)):

        if file_filter is not None:
            if data[i].img_id not in file_filter:
                continue

        # Get true percentage
        _true_damage_percent.append(data[i].damage_tiles.shape[0] / total_boxes)

        # Get test data
        x_test, y_test, x_pos = get_test_data(data[i])

        x_test = list(x_test)
        y_test = list(y_test)
        x_test_cnn = convert_to_tiles(x_pos, data[i].img_data)

        x_train = []
        y_train = []
        x_train_cnn = []

        display(HTML(f"<h2>================Testing image {data[i].img_id}================</h2>"))
        for j in range(len(data)):
            if j == i:
                # Skip testing image
                continue

            _x, _y, _x_pos = balance_data(data[j], settings.training_ratio)

            x_train += _x
            y_train += _y
            x_train_cnn += convert_to_tiles(_x_pos, data[j].img_data)

        # Cnn
        y_proba_cnn, y_train_proba_cnn = create_cnn_model(x_train_cnn, y_train,
                                                          x_test_cnn, y_test,
                                                          settings)

        # Regular NN
        y_pred, y_proba, train_acc, y_train_proba_nn = create_nn_model(x_train, y_train,
                                                                       x_test, y_test,
                                                                       settings)

        # Write them in CSV
        pos_x = []
        pos_y = []
        for point in x_pos:
            pos_x.append(point[0])
            pos_y.append(point[1])

        _tmp_res = np.vstack((y_proba_cnn.flatten(), y_proba, y_test, pos_x, pos_y)).transpose()
        np.savetxt(f"{save_path}/{data[i].img_id}-test.csv", _tmp_res, delimiter=",")

        _tmp_res = np.vstack((y_train_proba_cnn, y_train_proba_nn, y_train)).transpose()
        np.savetxt(f"{save_path}/{data[i].img_id}-train.csv", _tmp_res, delimiter=",")

    #     # Use only to see images
    #     if display_image:
    #         img_tmp = data[i].img_data.copy()
    #
    #         for item in zip(y_pred, y_test, x_pos):
    #
    #             if item[0] == 1 and item[1] == 1:
    #                 b_color = (171, 140, 209)
    #             elif item[0] == 1 and item[1] == 0:
    #                 b_color = (171, 140, 209)
    #             elif item[0] == 0 and item[1] == 1:
    #                 b_color = (235, 197, 174)
    #             else:
    #                 b_color = None
    #
    #             if b_color is not None:
    #                 x = item[2][0]
    #                 y = item[2][1]
    #                 y1 = y + settings.split_height
    #                 x1 = x + settings.split_width
    #                 img_tmp = cv2.rectangle(img_tmp, (x, y), (x1, y1), b_color, -1)
    #
    #         img_tmp = cv2.addWeighted(img_tmp, alpha, data[i].img_data, 1 - alpha, 0)
    #         display_annotations(img_tmp, data[i].annotations)
    #
    # print(f"True percent: {*_true_damage_percent,}")
    # print(f"Predicted percent: {*_pred_damage_percent,}")
    # # return sorted(res, key = lambda x: x[0])