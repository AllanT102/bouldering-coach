import argparse
import json
from pathlib import Path

from boulder_coach.pipeline import analyze_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze bouldering form from an MP4 video."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the input MP4 video.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory where outputs will be written.",
    )
    parser.add_argument(
        "--every-n",
        type=int,
        default=1,
        help="Process every Nth frame for speed (default: 1).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Pose detection confidence threshold (default: 0.5).",
    )
    parser.add_argument(
        "--route-mask",
        type=Path,
        help="Path to a route mask image (white = route holds).",
    )
    parser.add_argument(
        "--route-mask-threshold",
        type=int,
        default=10,
        help="Pixel threshold for the route mask (default: 10).",
    )
    parser.add_argument(
        "--gemini",
        action="store_true",
        help="Enable Gemini grip and hold analysis.",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default=None,
        help="Gemini model name (default: gemini-1.5-flash).",
    )
    parser.add_argument(
        "--gemini-every-n",
        type=int,
        default=10,
        help="Run Gemini every Nth processed frame (default: 10).",
    )
    parser.add_argument(
        "--expected-grip-map",
        type=Path,
        help="JSON file mapping hold_type to expected grip_type.",
    )
    parser.add_argument(
        "--gemini-output",
        type=Path,
        help="Optional path to write Gemini grip observations JSON.",
    )
    return parser


def _load_json_map(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object mapping hold_type to grip_type.")
    normalized = {}
    for key, value in data.items():
        if key is None:
            continue
        normalized_key = str(key).strip().lower()
        normalized_value = str(value).strip().lower() if value is not None else None
        normalized[normalized_key] = normalized_value
    return normalized


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    expected_grip_map = None
    if args.expected_grip_map:
        expected_grip_map = _load_json_map(args.expected_grip_map)

    analyze_video(
        input_path=args.input,
        output_dir=args.output,
        every_n=args.every_n,
        min_confidence=args.confidence,
        route_mask_path=args.route_mask,
        route_mask_threshold=args.route_mask_threshold,
        gemini_enabled=args.gemini,
        gemini_model=args.gemini_model,
        gemini_every_n=args.gemini_every_n,
        expected_grip_map=expected_grip_map,
        gemini_output=args.gemini_output,
    )


if __name__ == "__main__":
    main()
