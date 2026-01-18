from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from boulder_coach.pose import PoseResult


@dataclass
class RouteMask:
    mask: np.ndarray
    threshold: int = 10

    def contains(self, x: int, y: int) -> bool:
        height, width = self.mask.shape[:2]
        if x < 0 or y < 0 or x >= width or y >= height:
            return False
        return int(self.mask[y, x]) > self.threshold


@dataclass
class HandROI:
    frame: int
    hand: str
    bbox: tuple[int, int, int, int]
    image_bgr: np.ndarray


def load_route_mask(path: Path, frame_size: tuple[int, int], threshold: int = 10) -> RouteMask:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Unable to read route mask: {path}")
    width, height = frame_size
    if mask.shape[1] != width or mask.shape[0] != height:
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return RouteMask(mask=mask, threshold=threshold)


def build_route_mask_from_points(
    points: Iterable[tuple[int, int]],
    frame_size: tuple[int, int],
    radius: int = 30,
    threshold: int = 10,
) -> RouteMask:
    width, height = frame_size
    mask = np.zeros((height, width), dtype=np.uint8)
    for x, y in points:
        cv2.circle(mask, (x, y), radius, 255, -1)
    return RouteMask(mask=mask, threshold=threshold)


def collect_wrist_points(
    pose: PoseResult,
    frame_bgr: np.ndarray,
    min_visibility: float = 0.4,
) -> list[tuple[int, int]]:
    if not pose.landmarks:
        return []
    height, width = frame_bgr.shape[:2]
    points = []
    for landmark_name in ("LEFT_WRIST", "RIGHT_WRIST"):
        if landmark_name not in pose.landmarks:
            continue
        if pose.visibility and pose.visibility.get(landmark_name, 1.0) < min_visibility:
            continue
        x_norm, y_norm = pose.landmarks[landmark_name]
        points.append((int(x_norm * width), int(y_norm * height)))
    return points


def extract_hand_rois(
    frame_bgr: np.ndarray,
    pose: PoseResult,
    frame_index: int,
    route_mask: Optional[RouteMask] = None,
    min_visibility: float = 0.4,
    min_size: int = 64,
    crop_scale: float = 0.7,
) -> list[HandROI]:
    if not pose.landmarks:
        return []

    height, width = frame_bgr.shape[:2]

    def to_pixel(point: np.ndarray) -> tuple[int, int]:
        return int(point[0] * width), int(point[1] * height)

    shoulder_center = _safe_center(pose, "LEFT_SHOULDER", "RIGHT_SHOULDER", width, height)
    hip_center = _safe_center(pose, "LEFT_HIP", "RIGHT_HIP", width, height)
    torso_length = None
    if shoulder_center and hip_center:
        torso_length = int(np.linalg.norm(np.array(shoulder_center) - np.array(hip_center)))

    crop_size = max(min_size, int((torso_length or min_size) * crop_scale))

    rois: list[HandROI] = []
    for hand_label, landmark_name in ("left", "LEFT_WRIST"), ("right", "RIGHT_WRIST"):
        if landmark_name not in pose.landmarks:
            continue
        if pose.visibility and pose.visibility.get(landmark_name, 1.0) < min_visibility:
            continue
        x, y = to_pixel(pose.landmarks[landmark_name])
        if route_mask and not route_mask.contains(x, y):
            continue
        roi = _crop_square(frame_bgr, x, y, crop_size)
        if roi is None:
            continue
        rois.append(
            HandROI(
                frame=frame_index,
                hand=hand_label,
                bbox=roi[0],
                image_bgr=roi[1],
            )
        )

    return rois


def _safe_center(
    pose: PoseResult,
    left_name: str,
    right_name: str,
    width: int,
    height: int,
) -> Optional[tuple[int, int]]:
    if not pose.landmarks or left_name not in pose.landmarks or right_name not in pose.landmarks:
        return None
    left = pose.landmarks[left_name]
    right = pose.landmarks[right_name]
    x = int(((left[0] + right[0]) / 2.0) * width)
    y = int(((left[1] + right[1]) / 2.0) * height)
    return x, y


def _crop_square(
    frame_bgr: np.ndarray,
    x: int,
    y: int,
    size: int,
) -> Optional[tuple[tuple[int, int, int, int], np.ndarray]]:
    height, width = frame_bgr.shape[:2]
    half = size // 2
    x1 = max(0, x - half)
    y1 = max(0, y - half)
    x2 = min(width, x + half)
    y2 = min(height, y + half)
    if x2 <= x1 or y2 <= y1:
        return None
    cropped = frame_bgr[y1:y2, x1:x2]
    return (x1, y1, x2, y2), cropped
