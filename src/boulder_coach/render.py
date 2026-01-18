from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from boulder_coach.metrics import FrameMetrics
from boulder_coach.pose import POSE_LANDMARK_NAMES, PoseResult


def annotate_frame(
    frame_bgr: np.ndarray,
    pose: PoseResult,
    metrics: FrameMetrics,
) -> np.ndarray:
    annotated = frame_bgr.copy()

    if pose.landmarks:
        _draw_pose(annotated, pose)

    _draw_metrics(annotated, metrics)
    return annotated


def _draw_pose(frame: np.ndarray, pose: PoseResult) -> None:
    height, width = frame.shape[:2]
    points = []
    for name in POSE_LANDMARK_NAMES:
        if not pose.landmarks or name not in pose.landmarks:
            points.append(None)
            continue
        x_norm, y_norm = pose.landmarks[name]
        points.append((int(x_norm * width), int(y_norm * height)))

    for connection in mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS:
        start = points[connection.start]
        end = points[connection.end]
        if start is None or end is None:
            continue
        cv2.line(frame, start, end, (255, 255, 255), 2)

    for point in points:
        if point is None:
            continue
        cv2.circle(frame, point, 3, (0, 255, 0), -1)


def _draw_metrics(frame: np.ndarray, metrics: FrameMetrics) -> None:
    x = 16
    y = 24
    line_height = 20

    if not metrics.detected:
        cv2.putText(
            frame,
            "Pose not detected",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        return

    lines = [
        f"Left elbow: {metrics.left_elbow_angle:.1f}" if metrics.left_elbow_angle is not None else "Left elbow: --",
        f"Right elbow: {metrics.right_elbow_angle:.1f}" if metrics.right_elbow_angle is not None else "Right elbow: --",
        f"Left knee: {metrics.left_knee_angle:.1f}" if metrics.left_knee_angle is not None else "Left knee: --",
        f"Right knee: {metrics.right_knee_angle:.1f}" if metrics.right_knee_angle is not None else "Right knee: --",
        f"Hip tilt: {metrics.hip_line_tilt:.1f}" if metrics.hip_line_tilt is not None else "Hip tilt: --",
        f"Shoulder tilt: {metrics.shoulder_line_tilt:.1f}" if metrics.shoulder_line_tilt is not None else "Shoulder tilt: --",
        f"Hip-to-hand ratio: {metrics.hip_to_hand_ratio:.2f}" if metrics.hip_to_hand_ratio is not None else "Hip-to-hand ratio: --",
    ]

    for line in lines:
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        y += line_height
