"""Metrics related functions."""
import numpy as np
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score


def custom_f1(y: np.ndarray, ypred: np.ndarray) -> float:
    """
    Custom f1 function for cnn model.

    Parameters
    ----------
    y :
    ypred :

    Returns
    -------

    """
    def recall_m(y_true, y_pred):
        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        return tp / (positives + K.epsilon())

    def precision_m(y_true, y_pred):
        tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        pred_positive = K.sum(K.round(K.clip(y_pred, 0, 1)))

        return tp / (pred_positive + K.epsilon())

    precision, recall = precision_m(y, ypred), recall_m(y, ypred)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def get_stats(y_pred, y_test) -> tuple:
    """
    Prints and returns the metric scores for model.

    Parameters
    ----------
    y_pred :
    y_test :

    Returns
    -------

    """
    f1 = round(f1_score(y_test, y_pred), 5)
    prec = round(precision_score(y_test, y_pred), 5)
    rec = round(recall_score(y_test, y_pred), 5)
    acc = round(accuracy_score(y_test, y_pred), 5)
    print(f"Accuracy - {acc} F1 Score - {f1}")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print("\t  Actual")
    print("\t 1   \t|   0")
    print(f"Pred 1 \t {tp}\t|   {fp}")
    print(f"     0 \t {fn}\t|   {tn}")
    fpr = round((fp / (fp + tn)), 5)
    fnr = round((fn / (fn + tp)), 5)

    return acc, f1, prec, rec, fpr, fnr, fp, fn
