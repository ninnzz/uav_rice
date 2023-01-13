"""
Contains all graph/display related functions.
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt


def display_annotations(img: np.ndarray, boxes: list, target_label: str = "damage"):
    """
    Display image annotations.

    Parameters
    ----------
    img :
    boxes :
    target_label :

    Returns
    -------

    """
    for box in boxes:
        if box['@label'] == target_label:
            start = (int(float(box["@xtl"])), int(float(box["@ytl"])))
            end = (int(float(box["@xbr"])), int(float(box["@ybr"])))

            img = cv2.rectangle(img, start, end, (20, 100, 240), 5)

        # Add display for no damage box
        # elif box['@label'] == 'no_damage_true':
        #     start = (int(float(box["@xtl"])), int(float(box["@ytl"])))
        #     end = (int(float(box["@xbr"])), int(float(box["@ybr"])))

        #     img = cv2.rectangle(img, start, end, (240, 50, 10), 5)

    plt.figure(figsize=(70, 40))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), interpolation='nearest')
    plt.show()
