"""
Post processing functions.
"""
import os
import numpy as np
import pandas as pd
from uav_utils.data_classes import Dimensions, ModelsMetric, MetricScores, AnalysisSettings
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from IPython.display import display, HTML


def remove_island(y_pred: list, y_test: list, pos: list, d: Dimensions):
    """
    Remove predicted damage tiles with no neigbhor.

    Parameters
    ----------
    y_pred :
    y_test :
    pos :
    d :

    Returns
    -------

    """
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


def apply_prediction(fname: str, threshold: float, ratio: tuple,
                     cnn_thresh: float, nn_thresh: float,
                     data_type: str = "testing",
                     has_island: bool = False,
                     metrics: str = "accuracy",
                     settings: AnalysisSettings = None) -> ModelsMetric:
    """
    Computes the accuracy using the ratio and threshold.

    Parameters
    ----------
    fname :
    threshold :
    ratio :
    cnn_thresh :
    nn_thresh :
    data_type :
    has_island :
    metrics :
    settings :

    Returns
    -------

    """

    if data_type == "testing":
        cols = ["cnn", "nn", "label", "pos_x", "pos_y"]
    else:
        cols = ["cnn", "nn", "label"]

    data = pd.read_csv(fname, names=cols)
    cnn_ratio, nn_ratio = ratio

    weighted = (cnn_ratio / 100) * data.cnn + (nn_ratio / 100) * data.nn
    data["weighted"] = weighted
    data["weighted_predicted_label"] = data.apply(lambda row: 0 if row.weighted < threshold else 1, axis=1)
    data["cnn_predicted_label"] = data.apply(lambda row: 0 if row.cnn < cnn_thresh else 1, axis=1)
    data["nn_predicted_label"] = data.apply(lambda row: 0 if row.nn < nn_thresh else 1, axis=1)

    if has_island:
        pos = []
        for x, y in zip(data["pos_x"], data["pos_y"]):
            pos.append((int(x), int(y)))

        data["weighted_predicted_label"] = remove_island(data.weighted_predicted_label.tolist(),
                                                         data.label.tolist(),
                                                         pos, settings.dimension)
        data["cnn_predicted_label"] = remove_island(data.cnn_predicted_label.tolist(), data.label.tolist(), pos,
                                                    settings.dimension)
        data["nn_predicted_label"] = remove_island(data.nn_predicted_label.tolist(), data.label.tolist(), pos,
                                                   settings.dimension)

    weighted_precision = precision_score(data.label, data.weighted_predicted_label)
    cnn_precision = precision_score(data.label, data.cnn_predicted_label)
    nn_precision = precision_score(data.label, data.nn_predicted_label)
    weighted_recall = recall_score(data.label, data.weighted_predicted_label)
    cnn_recall = recall_score(data.label, data.cnn_predicted_label)
    nn_recall = recall_score(data.label, data.nn_predicted_label)

    if metrics == "accuracy":
        # Accuracy
        weighted_accuracy = accuracy_score(data.label, data.weighted_predicted_label)
        cnn_accuracy = accuracy_score(data.label, data.cnn_predicted_label)
        nn_accuracy = accuracy_score(data.label, data.nn_predicted_label)

        ms = ModelsMetric(
            weighted=MetricScores(
                target_score=weighted_accuracy,
                accuracy=weighted_accuracy,
                recall=weighted_recall,
                precision=weighted_precision
            ),
            cnn=MetricScores(
                target_score=cnn_accuracy,
                accuracy=cnn_accuracy,
                recall=cnn_recall,
                precision=cnn_precision
            ),
            nn=MetricScores(
                target_score=nn_accuracy,
                accuracy=nn_accuracy,
                recall=nn_recall,
                precision=nn_precision
            )
        )

        if settings.debug:
            display(data)
            display(HTML(f"<p>Accuracy <i>({cnn_ratio}:{nn_ratio})</i> <b>{round(weighted_accuracy, 6)}</b></p>"))
            display(HTML(f"<p>Accuracy <i>(CNN)</i> <b>{round(cnn_accuracy, 6)}</b></p>"))
            display(HTML(f"<p>Accuracy <i>(NN)</i> <b>{round(nn_accuracy, 6)}</b></p>"))

        return ms
    else:
        # F1
        weighted_f1 = f1_score(data.label, data.weighted_predicted_label)
        cnn_f1 = f1_score(data.label, data.cnn_predicted_label)
        nn_f1 = f1_score(data.label, data.nn_predicted_label)

        ms = ModelsMetric(
            weighted=MetricScores(
                target_score=weighted_f1,
                f1=weighted_f1,
                recall=weighted_recall,
                precision=weighted_precision
            ),
            cnn=MetricScores(
                target_score=cnn_f1,
                f1=cnn_f1,
                recall=cnn_recall,
                precision=cnn_precision
            ),
            nn=MetricScores(
                target_score=nn_f1,
                f1=nn_f1,
                recall=nn_recall,
                precision=nn_precision
            )
        )

        if settings.debug:
            display(data)
            display(HTML(f"<p>F1 <i>({cnn_ratio}:{nn_ratio})</i> <b>{round(weighted_f1, 6)}</b></p>"))
            display(HTML(f"<p>F1 <i>(CNN)</i> <b>{round(cnn_f1, 6)}</b></p>"))
            display(HTML(f"<p>F1 <i>(NN)</i> <b>{round(nn_f1, 6)}</b></p>"))

        return ms


def compute_threshold(data: np.ndarray):
    """
    Computes the optimal threshold base on training results.
    Parameters
    ----------
    data :

    Returns
    -------

    """
    f1 = 0
    acc = 0
    f1_thresh = 0
    acc_thresh = 0
    # Sort data
    sorted_data = list(data)
    sorted_data = sorted(sorted_data, key=lambda i: i[0], reverse=True)
    sorted_data = np.array(sorted_data)

    fn_counter = int(sum(sorted_data[:, 1]))
    tn_counter = len(sorted_data) - fn_counter
    tp_counter = 0
    fp_counter = 0
    metrics = []
    _small_num = 0.0000000001
    for score, label in sorted_data:
        if label == 1:
            tp_counter += 1
            fn_counter -= 1
        else:
            fp_counter += 1
            tn_counter -= 1

        # True pos, false pose, true neg, false neg, accuracy, fq
        # metrics.append((tp_counter, fp_counter, tn_counter, fn_counter))
        # Compute f1, prec, recall
        _prec = tp_counter / (tp_counter + fp_counter + _small_num)
        _rec = tp_counter / (tp_counter + fn_counter + _small_num)
        _f1 = 2 * (_prec * _rec) / (_prec + _rec + _small_num)
        _acc = (tp_counter + tn_counter) / (tp_counter + tn_counter + fp_counter + fn_counter)

        if acc <= _acc:
            acc = _acc
            acc_thresh = score

        if f1 <= _f1:
            f1 = _f1
            f1_thresh = score

    return acc, acc_thresh, f1, f1_thresh


def process_one_image_score(fname: str, img_id: str, metrics: str = "accuracy", debug: bool = False):
    """
    Process one csv result.

    Parameters
    ----------
    fname :
    img_id :
    metrics :
    debug :

    Returns
    -------

    """
    if debug:
        display(HTML(f"<h2>Fold for img: {img_id}</h2>"))
    data = pd.read_csv(fname, names=["cnn", "nn", "label"])

    df = pd.DataFrame(columns=["accuracy", "acc_threshold_score", "f1", "f1_threshold_score"])
    df.index.name = "cnn/nn ratio"
    cnn_ratio = 95

    acc_thresh = 0
    max_acc = 0
    max_acc_ratio = [100, 5]

    f1_thresh = 0
    max_f1 = 0
    max_f1_ratio = [100, 5]

    # CNN only
    cnn_only = np.vstack((data.cnn, data.label)).transpose()
    cnn_acc, cnn_acc_thr, cnn_f1, cnn_f1_thr = compute_threshold(cnn_only)

    # NN Only
    nn_only = np.vstack((data.nn, data.label)).transpose()
    nn_acc, nn_acc_thr, nn_f1, nn_f1_thr = compute_threshold(nn_only)

    cnn_row = f"<tr><td>CNN</td><td>{cnn_acc}</td><td>{cnn_acc_thr}</td><td>{cnn_f1}</td><td>{cnn_f1_thr}</td></tr>"
    nn_row = f"<tr><td>NN</td><td>{nn_acc}</td><td>{nn_acc_thr}</td><td>{nn_f1}</td><td>{nn_f1_thr}</td></tr>"
    table = f"<table><tr><td></td><td>accuracy</td><td>acc_thresh</td><td>f1</td><td>f1_thresh</td></tr>{cnn_row}{nn_row}</table>"

    if debug:
        display(HTML(table))

    # Compute for weights combination and scores
    while cnn_ratio > 0:
        nn_ratio = 100 - cnn_ratio

        weighted = (cnn_ratio / 100) * data.cnn + (nn_ratio / 100) * data.nn

        # Combined
        combined = np.vstack((weighted, data.label)).transpose()
        acc, acc_thr, f1, f1_thr = compute_threshold(combined)

        if acc > max_acc:
            max_acc_ratio = [cnn_ratio, nn_ratio]
            acc_thresh = acc_thr
            max_acc = acc

        if f1 > max_f1:
            max_f1_ratio = [cnn_ratio, nn_ratio]
            f1_thresh = f1_thr
            max_f1 = f1

        df.loc[f"ratio-{cnn_ratio}:{nn_ratio}"] = [acc, acc_thr, f1, f1_thr]
        cnn_ratio -= 5

    if metrics == "accuracy":
        if debug:
            max_df = df[df.accuracy == df.accuracy.max()]
            display(df)
        return acc_thresh, max_acc_ratio, cnn_acc_thr, nn_acc_thr
    else:
        if debug:
            max_df = df[df.f1 == df.f1.max()]
            display(df)
        return f1_thresh, max_f1_ratio, cnn_f1_thr, nn_f1_thr


def process_images(file_loc: str, settings: AnalysisSettings, metric: str = "accuracy"):
    """
    Starts analysis.

    Parameters
    ----------
    file_loc :
    settings :
    metric :

    Returns
    -------

    """
    weighted_scores = []
    cnn_scores = []
    nn_scores = []

    weighted_scores_no_island = []
    cnn_scores_no_island = []
    nn_scores_no_island = []

    tr_weighted_scores = []
    tr_cnn_scores = []
    tr_nn_scores = []

    fold_df = pd.DataFrame(columns=["ratio"] + settings.display_columns)
    fold_df.index.name = "Image Fold"

    for filename in settings.file_names:
        df_item = []
        # Training data
        training_fname = os.path.join(file_loc, f"{filename}-train.csv")
        thresh, ratio, cnn_thresh, nn_thresh = process_one_image_score(training_fname, filename, metric, settings.debug)

        if settings.debug:
            print(f"Applying threshold {thresh} and ratio (cnn/nn) {ratio} to test data.")
            print(f"CNN Only thres: {cnn_thresh}")
            print(f"NN Only thres: {nn_thresh}")

        # Testing data
        testing_fname = os.path.join(file_loc, f"{filename}-test.csv")
        ms = apply_prediction(testing_fname, thresh, ratio, cnn_thresh, nn_thresh, "testing", False, metric, settings)
        weighted_scores.append(ms.weighted.target_score)
        cnn_scores.append(ms.cnn.target_score)
        nn_scores.append(ms.nn.target_score)

        # Testing remove island
        ms2 = apply_prediction(testing_fname, thresh, ratio, cnn_thresh, nn_thresh, "testing", True, metric, settings)
        weighted_scores_no_island.append(ms2.weighted.target_score)
        cnn_scores_no_island.append(ms2.cnn.target_score)
        nn_scores_no_island.append(ms2.nn.target_score)

        # Training data prediction
        ms3 = apply_prediction(training_fname, thresh, ratio, cnn_thresh, nn_thresh, "training", False, metric, settings)
        tr_weighted_scores.append(ms3.weighted.target_score)
        tr_cnn_scores.append(ms3.cnn.target_score)
        tr_nn_scores.append(ms3.nn.target_score)

        # Add entries to df item
        df_item.append(ratio)

        for sc in [ms, ms2, ms3]:
            df_item.append(sc.weighted.target_score)
            df_item.append(sc.cnn.target_score)
            df_item.append(sc.nn.target_score)
            df_item.append(sc.weighted.recall)
            df_item.append(sc.cnn.recall)
            df_item.append(sc.nn.recall)
            df_item.append(sc.weighted.precision)
            df_item.append(sc.cnn.precision)
            df_item.append(sc.nn.precision)

        fold_df.loc[filename] = df_item

    #         tb += f"<tr><td>{filename}</td><td>{ratio}</td><td>{round(w, 5)}</td><td>{round(cnn, 5)}</td><td>{round(nn, 5)}</td><td>{round(w_ni, 5)}</td><td>{round(cnn_ni, 5)}</td><td>{round(nn_ni, 5)}</td><td>{round(tr_w, 5)}</td><td>{round(tr_cnn, 5)}</td><td>{round(tr_nn, 5)}</td></tr>"

    #     tb += f"<tr><td>Mean</td><td></td><td>{round(np.mean(weighted_scores), 5)}</td><td>{round(np.mean(cnn_scores), 5)}</td><td>{round(np.mean(nn_scores), 5)}</td><td>{round(np.mean(weighted_scores_no_island), 5)}</td><td>{round(np.mean(cnn_scores_no_island), 5)}</td><td>{round(np.mean(nn_scores_no_island), 5)}</td><td>{round(np.mean(tr_weighted_scores), 5)}</td><td>{round(np.mean(tr_cnn_scores), 5)}</td><td>{round(np.mean(tr_nn_scores), 5)}</td></tr>"

    # Return the final mean scores for accuracy and f1
    mean_cols = []
    for num_cols in settings.display_columns:
        mean_cols.append(np.mean(fold_df[num_cols]))

    fold_df.loc["mean"] = [""] + mean_cols
    #     final_mean_scores = [
    #         np.mean(ms.weighted.target_score),
    #         np.mean(ms.cnn.target_score),
    #         np.mean(nn_scores),
    #         np.mean(ms2.weighted.target_score),
    #         np.mean(ms2.cnn.target_score),
    #         np.mean(ms2.nn.target_score),
    #         np.mean(ms3.weighted.target_score),
    #         np.mean(ms3.cnn.target_score),
    #         np.mean(ms3.nn.target_score)
    #     ]
    #     return HTML(f"<table>{tb}</table>"), final_mean_scores
    return fold_df, mean_cols
