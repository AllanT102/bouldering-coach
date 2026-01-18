from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class PoseResult:
    landmarks: Optional[Dict[str, np.ndarray]]
    visibility: Optional[Dict[str, float]]


class PoseEstimator:
    def __init__(self, min_confidence: float = 0.5, model_path: Optional[Path] = None) -> None:
        self.min_confidence = min_confidence
        self.model_path = model_path
        self._pose = None

    def __enter__(self) -> "PoseEstimator":
        model_path = self._ensure_model_path()
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            min_pose_detection_confidence=self.min_confidence,
            min_pose_presence_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence,
        )
        self._pose = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._pose:
            self._pose.close()

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> PoseResult:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self._pose.detect_for_video(mp_image, timestamp_ms)

        if not results.pose_landmarks:
            return PoseResult(landmarks=None, visibility=None)

        landmarks = {}
        visibility = {}
        first_pose = results.pose_landmarks[0]
        for idx, landmark in enumerate(first_pose):
            name = POSE_LANDMARK_NAMES[idx]
            landmarks[name] = np.array([landmark.x, landmark.y])
            visibility[name] = float(getattr(landmark, "visibility", getattr(landmark, "presence", 0.0)))

        return PoseResult(landmarks=landmarks, visibility=visibility)

    def _ensure_model_path(self) -> Path:
        if self.model_path:
            return self.model_path.expanduser().resolve()

        cache_dir = Path.home() / ".cache" / "boulder_coach"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = cache_dir / "pose_landmarker_lite.task"
        if not model_path.exists():
            urlretrieve(MODEL_URL, model_path)
        return model_path


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

POSE_LANDMARK_NAMES = [
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
]
