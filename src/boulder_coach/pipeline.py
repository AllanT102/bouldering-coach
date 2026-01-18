from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from boulder_coach.pose import PoseEstimator, PoseResult
from boulder_coach.render import annotate_frame
from boulder_coach.metrics import FrameMetrics, summarize_metrics


@dataclass
class AnalysisOutput:
    annotated_video: Path
    metrics_csv: Path
    summary_json: Path


def analyze_video(
    input_path: Path,
    output_dir: Path,
    every_n: int = 1,
    min_confidence: float = 0.5,
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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(annotated_path),
        fourcc,
        fps,
        (width, height),
    )

    frame_index = 0
    metrics_rows: list[FrameMetrics] = []

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

            annotated = annotate_frame(frame, pose_result, metrics)
            writer.write(annotated)

    cap.release()
    writer.release()

    _write_metrics_csv(metrics_path, metrics_rows)
    summary = summarize_metrics(metrics_rows, fps, every_n)
    _write_summary_json(summary_path, summary)

    return AnalysisOutput(
        annotated_video=annotated_path,
        metrics_csv=metrics_path,
        summary_json=summary_path,
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
