"""
Main model files.
"""
import numpy as np

# CNN
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers

# NN
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from uav_utils.metrics import custom_f1
from uav_utils.utils import compute_threshold
from uav_utils.data_utils import ExperimentParams


def create_cnn_model(x_train: np.ndarray, y_train: np.ndarray,
                     x_test: np.ndarray, y_test: np.ndarray,
                     params: ExperimentParams) -> tuple:
    """
    Creates the CNN based model.

    Returns the probabilty score and the labels.
    Parameters
    ----------
    x_train :
    y_train :
    x_test :
    y_test :
    params :

    Returns
    -------

    """
    x_train, x_test = np.array(x_train) / 255.0, np.array(x_test) / 255.0

    print("X train length", len(x_train))
    print("X test length", len(x_test))
    y_train = np.array(y_train).astype(float)
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3),
                            activation='relu',
                            input_shape=(params.split_width, params.split_height, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy', custom_f1])

    if params.model_debug:
        # Run basic training if you are just debugging the code
        model.fit(x_train, y_train, epochs=1, batch_size=10)
    else:
        model.fit(x_train, y_train, epochs=20, batch_size=256)

    y_test_pred = model.predict(x_train)

    combined = np.vstack((y_test_pred.flatten(), y_train)).transpose()
    print(combined[:10])
    acc, acc_thr, f1, f1_thr = compute_threshold(combined)
    print(f"CNN Threshold for highest accuracy ({acc}): {acc_thr}")
    print(f"CNN Threshold for highest f1 ({f1}): {f1_thr}")

    y_pred = model.predict(x_test)

    return y_pred, y_test_pred.flatten()


def create_nn_model(x_train: np.ndarray, y_train: np.ndarray,
                    x_test: np.ndarray, y_test: np.ndarray,
                    params: ExperimentParams) -> tuple:
    """
    Creates the NN model.

    Parameters
    ----------
    x_train :
    y_train :
    x_test :
    y_test :
    params :

    Returns
    -------

    """
    clf = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            solver='lbfgs',
            alpha=0.00001,
            hidden_layer_sizes=(9, 9,),
            max_iter=10000, random_state=params.random_state))

    print("Finish training")
    clf.fit(x_train, y_train)
    y_proba = clf.predict_proba(x_test)[:, 1]
    y_pred = clf.predict(x_test)
    test_acc = round(clf.score(x_train, y_train), 5)

    # Get probabilty scores
    y_test_pred = clf.predict_proba(x_train)[:, 1]

    combined = np.vstack((y_test_pred, y_train)).transpose()
    print(combined[:10])
    acc, acc_thr, f1, f1_thr = compute_threshold(combined)
    print(f"NN Threshold for highest accuracy ({acc}): {acc_thr}")
    print(f"NN Threshold for highest f1 ({f1}): {f1_thr}")

    return y_pred, y_proba, test_acc, y_test_pred
