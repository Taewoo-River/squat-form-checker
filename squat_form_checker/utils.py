import os

import cv2
import numpy as np


def open_webcam(index: int = 0):
    """Open a webcam by index.

    On Windows, the default MSMF backend often yields a black image in browsers
    (Streamlit) or OpenCV windows; DirectShow usually fixes it.
    """
    if os.name == "nt":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(index)


def calculate_angle(a, b, c):
    """Angle in degrees at vertex b formed by points a-b-c.

    Each point is an (x, y) or (x, y, z) sequence; only x and y are used.
    """
    a = np.array(a[:2], dtype=float)
    b = np.array(b[:2], dtype=float)
    c = np.array(c[:2], dtype=float)

    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def midpoint(a, b):
    """Midpoint between two (x, y, ...) points."""
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
