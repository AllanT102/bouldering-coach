from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from boulder_coach.gemini_client import GeminiGripClient
from boulder_coach.grip import GripObservation, summarize_grip_observations
from boulder_coach.holds import (
    build_route_mask_from_points,
    collect_wrist_points,
    extract_hand_rois,
    load_route_mask,
)
from boulder_coach.pose import PoseEstimator, PoseResult
from boulder_coach.render import annotate_frame
from boulder_coach.metrics import FrameMetrics, summarize_metrics


@dataclass
class AnalysisOutput:
    annotated_video: Path
    metrics_csv: Path
    summary_json: Path
    grip_json: Optional[Path] = None


def analyze_video(
    input_path: Path,
    output_dir: Path,
    every_n: int = 1,
    min_confidence: float = 0.5,
    route_mask_path: Optional[Path] = None,
    route_mask_threshold: int = 10,
    gemini_enabled: bool = False,
    gemini_model: Optional[str] = None,
    gemini_every_n: int = 10,
    expected_grip_map: Optional[dict] = None,
    gemini_output: Optional[Path] = None,
) -> AnalysisOutput:
    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    annotated_path = output_dir / "annotated.mp4"
    metrics_path = output_dir / "metrics.csv"
    summary_path = output_dir / "summary.json"
    grip_path = gemini_output or (output_dir / "grip_analysis.json")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(annotated_path),
        fourcc,
        fps,
        (width, height),
    )

    frame_index = 0
    metrics_rows: list[FrameMetrics] = []
    grip_observations: list[GripObservation] = []

    route_mask = None
    if route_mask_path:
        route_mask = load_route_mask(route_mask_path, (width, height), threshold=route_mask_threshold)

    gemini_client = None
    route_detection = None
    if gemini_enabled:
        if route_mask is None:
            route_mask = _build_auto_route_mask(
                input_path,
                every_n=every_n,
                min_confidence=min_confidence,
                threshold=route_mask_threshold,
            )
            route_detection = "auto"
        else:
            route_detection = "mask"
        gemini_client = GeminiGripClient(model=gemini_model)

    with PoseEstimator(min_confidence=min_confidence) as estimator:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_index += 1
            if every_n > 1 and frame_index % every_n != 0:
                continue

            timestamp_ms = int((frame_index / fps) * 1000)
            pose_result = estimator.process(frame, timestamp_ms)
            metrics = FrameMetrics.from_pose(frame_index, pose_result)
            metrics_rows.append(metrics)

            if gemini_client and frame_index % gemini_every_n == 0:
                hand_rois = extract_hand_rois(
                    frame,
                    pose_result,
                    frame_index,
                    route_mask=route_mask,
                )
                for roi in hand_rois:
                    result = gemini_client.analyze_hold(roi.image_bgr, roi.hand)
                    expected_grip = None
                    mismatch = None
                    if expected_grip_map and result.hold_type:
                        expected_grip = expected_grip_map.get(result.hold_type.lower())
                        if expected_grip and result.grip_type:
                            mismatch = expected_grip != result.grip_type

                    grip_observations.append(
                        GripObservation(
                            frame=roi.frame,
                            hand=roi.hand,
                            hold_type=result.hold_type,
                            grip_type=result.grip_type,
                            confidence=result.confidence,
                            expected_grip=expected_grip,
                            mismatch=mismatch,
                            notes=result.notes,
                        )
                    )

            annotated = annotate_frame(frame, pose_result, metrics)
            writer.write(annotated)

    cap.release()
    writer.release()

    _write_metrics_csv(metrics_path, metrics_rows)
    summary = summarize_metrics(metrics_rows, fps, every_n)
    if grip_observations:
        summary["grip_analysis"] = summarize_grip_observations(grip_observations)
        summary["route_detection"] = route_detection
    _write_summary_json(summary_path, summary)

    if grip_observations:
        _write_grip_json(grip_path, grip_observations)

    return AnalysisOutput(
        annotated_video=annotated_path,
        metrics_csv=metrics_path,
        summary_json=summary_path,
        grip_json=grip_path if grip_observations else None,
    )


def _write_metrics_csv(path: Path, rows: Iterable[FrameMetrics]) -> None:
    fieldnames = list(FrameMetrics.csv_fields())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.to_csv_row())


def _write_summary_json(path: Path, summary: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def _write_grip_json(path: Path, observations: Iterable[GripObservation]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump([obs.to_dict() for obs in observations], handle, indent=2)


def _build_auto_route_mask(
    input_path: Path,
    every_n: int,
    min_confidence: float,
    threshold: int,
    radius: int = 30,
    min_visibility: float = 0.4,
) -> "RouteMask":
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    points: list[tuple[int, int]] = []
    frame_index = 0

    with PoseEstimator(min_confidence=min_confidence) as estimator:
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_index += 1
            if every_n > 1 and frame_index % every_n != 0:
                continue

            timestamp_ms = int((frame_index / fps) * 1000)
            pose_result = estimator.process(frame, timestamp_ms)
            points.extend(collect_wrist_points(pose_result, frame, min_visibility=min_visibility))

    cap.release()
    return build_route_mask_from_points(points, (width, height), radius=radius, threshold=threshold)
