import argparse
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    analyze_video(
        input_path=args.input,
        output_dir=args.output,
        every_n=args.every_n,
        min_confidence=args.confidence,
    )


if __name__ == "__main__":
    main()
