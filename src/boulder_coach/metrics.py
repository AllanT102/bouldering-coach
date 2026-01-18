from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Optional

import numpy as np

from boulder_coach.pose import PoseResult


LANDMARKS = {
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
}


@dataclass
class FrameMetrics:
    frame: int
    detected: bool
    left_elbow_angle: Optional[float]
    right_elbow_angle: Optional[float]
    left_knee_angle: Optional[float]
    right_knee_angle: Optional[float]
    left_hip_angle: Optional[float]
    right_hip_angle: Optional[float]
    shoulder_line_tilt: Optional[float]
    hip_line_tilt: Optional[float]
    hip_to_hand_ratio: Optional[float]

    @staticmethod
    def from_pose(frame: int, pose: PoseResult) -> "FrameMetrics":
        if not pose.landmarks:
            return FrameMetrics(
                frame=frame,
                detected=False,
                left_elbow_angle=None,
                right_elbow_angle=None,
                left_knee_angle=None,
                right_knee_angle=None,
                left_hip_angle=None,
                right_hip_angle=None,
                shoulder_line_tilt=None,
                hip_line_tilt=None,
                hip_to_hand_ratio=None,
            )

        pts = pose.landmarks
        left_elbow = angle(pts["LEFT_SHOULDER"], pts["LEFT_ELBOW"], pts["LEFT_WRIST"])
        right_elbow = angle(pts["RIGHT_SHOULDER"], pts["RIGHT_ELBOW"], pts["RIGHT_WRIST"])
        left_knee = angle(pts["LEFT_HIP"], pts["LEFT_KNEE"], pts["LEFT_ANKLE"])
        right_knee = angle(pts["RIGHT_HIP"], pts["RIGHT_KNEE"], pts["RIGHT_ANKLE"])
        left_hip = angle(pts["LEFT_SHOULDER"], pts["LEFT_HIP"], pts["LEFT_KNEE"])
        right_hip = angle(pts["RIGHT_SHOULDER"], pts["RIGHT_HIP"], pts["RIGHT_KNEE"])

        shoulder_line_tilt = line_tilt(pts["LEFT_SHOULDER"], pts["RIGHT_SHOULDER"])
        hip_line_tilt = line_tilt(pts["LEFT_HIP"], pts["RIGHT_HIP"])

        hip_center = (pts["LEFT_HIP"] + pts["RIGHT_HIP"]) / 2.0
        hand_center = (pts["LEFT_WRIST"] + pts["RIGHT_WRIST"]) / 2.0
        shoulder_center = (pts["LEFT_SHOULDER"] + pts["RIGHT_SHOULDER"]) / 2.0
        torso_length = np.linalg.norm(hip_center - shoulder_center) or 1.0
        hip_to_hand_ratio = np.linalg.norm(hand_center - hip_center) / torso_length

        return FrameMetrics(
            frame=frame,
            detected=True,
            left_elbow_angle=left_elbow,
            right_elbow_angle=right_elbow,
            left_knee_angle=left_knee,
            right_knee_angle=right_knee,
            left_hip_angle=left_hip,
            right_hip_angle=right_hip,
            shoulder_line_tilt=shoulder_line_tilt,
            hip_line_tilt=hip_line_tilt,
            hip_to_hand_ratio=hip_to_hand_ratio,
        )

    @staticmethod
    def csv_fields() -> Iterable[str]:
        return (
            "frame",
            "detected",
            "left_elbow_angle",
            "right_elbow_angle",
            "left_knee_angle",
            "right_knee_angle",
            "left_hip_angle",
            "right_hip_angle",
            "shoulder_line_tilt",
            "hip_line_tilt",
            "hip_to_hand_ratio",
        )

    def to_csv_row(self) -> dict:
        return {
            "frame": self.frame,
            "detected": self.detected,
            "left_elbow_angle": self.left_elbow_angle,
            "right_elbow_angle": self.right_elbow_angle,
            "left_knee_angle": self.left_knee_angle,
            "right_knee_angle": self.right_knee_angle,
            "left_hip_angle": self.left_hip_angle,
            "right_hip_angle": self.right_hip_angle,
            "shoulder_line_tilt": self.shoulder_line_tilt,
            "hip_line_tilt": self.hip_line_tilt,
            "hip_to_hand_ratio": self.hip_to_hand_ratio,
        }


def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = a - b
    cb = c - b
    ab_norm = ab / (np.linalg.norm(ab) + 1e-6)
    cb_norm = cb / (np.linalg.norm(cb) + 1e-6)
    cosine = float(np.clip(np.dot(ab_norm, cb_norm), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def line_tilt(a: np.ndarray, b: np.ndarray) -> float:
    delta = b - a
    angle_rad = np.arctan2(delta[1], delta[0])
    return float(np.degrees(angle_rad))


def summarize_metrics(rows: Iterable[FrameMetrics], fps: float, every_n: int) -> dict:
    detected_rows = [row for row in rows if row.detected]

    def avg(values: Iterable[Optional[float]]) -> Optional[float]:
        values_list = [v for v in values if v is not None]
        return mean(values_list) if values_list else None

    total_frames = len(rows)
    duration_seconds = (total_frames * every_n) / fps if fps else None

    summary = {
        "total_frames": total_frames,
        "detected_frames": len(detected_rows),
        "duration_seconds": duration_seconds,
        "avg_left_elbow_angle": avg(row.left_elbow_angle for row in detected_rows),
        "avg_right_elbow_angle": avg(row.right_elbow_angle for row in detected_rows),
        "avg_left_knee_angle": avg(row.left_knee_angle for row in detected_rows),
        "avg_right_knee_angle": avg(row.right_knee_angle for row in detected_rows),
        "avg_left_hip_angle": avg(row.left_hip_angle for row in detected_rows),
        "avg_right_hip_angle": avg(row.right_hip_angle for row in detected_rows),
        "avg_shoulder_line_tilt": avg(row.shoulder_line_tilt for row in detected_rows),
        "avg_hip_line_tilt": avg(row.hip_line_tilt for row in detected_rows),
        "avg_hip_to_hand_ratio": avg(row.hip_to_hand_ratio for row in detected_rows),
    }

    return summary
