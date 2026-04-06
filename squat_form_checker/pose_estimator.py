"""MediaPipe Pose landmark extraction using the Tasks API (mediapipe >= 0.10.21).

On first run, the lite pose-landmarker model (~4 MB) is downloaded
automatically to the package directory.
"""

import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    PoseLandmarksConnections,
    RunningMode,
)

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_full.task")


class PoseEstimator:
    """Thin wrapper around the MediaPipe PoseLandmarker (Tasks API)."""

    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30

    _CONNECTIONS = list(PoseLandmarksConnections.POSE_LANDMARKS)

    def __init__(self):
        self._ensure_model()
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)
        self._ts_ms = 0

    @staticmethod
    def _ensure_model():
        if os.path.exists(_MODEL_PATH):
            return
        print(f"Downloading pose model to {_MODEL_PATH} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("Download complete.")

    def process_frame(self, frame, timestamp_ms: int | None = None):
        """Return (landmarks_dict | None, raw_result) for a BGR frame.

        *timestamp_ms* must increase monotonically between calls.  When
        ``None`` (default) an internal counter increments by 33 ms (~30 fps).
        For video files, pass the actual frame timestamp for best accuracy.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if timestamp_ms is None:
            self._ts_ms += 33
            timestamp_ms = self._ts_ms
        else:
            self._ts_ms = timestamp_ms
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return None, result

        h, w = frame.shape[:2]
        raw_lms = result.pose_landmarks[0]
        landmarks = {}
        for idx, lm in enumerate(raw_lms):
            landmarks[idx] = {
                "x": lm.x * w,
                "y": lm.y * h,
                "z": lm.z,
                "visibility": lm.visibility if hasattr(lm, "visibility") else lm.presence,
            }
        return landmarks, result

    def draw_landmarks(self, frame, results):
        """Draw skeleton overlay on *frame* (mutates in-place)."""
        if not results.pose_landmarks:
            return frame

        h, w = frame.shape[:2]
        lms = results.pose_landmarks[0]

        for conn in self._CONNECTIONS:
            a, b = lms[conn.start], lms[conn.end]
            pt1 = (int(a.x * w), int(a.y * h))
            pt2 = (int(b.x * w), int(b.y * h))
            cv2.line(frame, pt1, pt2, (0, 220, 0), 2, cv2.LINE_AA)

        for lm in lms:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1, cv2.LINE_AA)

        return frame

    def get_point(self, landmarks, idx):
        """Return (x, y) pixel coords for landmark *idx*, or None."""
        lm = landmarks.get(idx)
        return (lm["x"], lm["y"]) if lm else None

    def get_visibility(self, landmarks, idx):
        lm = landmarks.get(idx)
        return lm["visibility"] if lm else 0.0

    def close(self):
        self._landmarker.close()
