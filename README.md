# Boulder Coach

Analyze bouldering form from an MP4 video using pose estimation. The tool generates an annotated video, per-frame metrics, and a summary report.

## What it does
- Detects body pose per frame
- Computes basic climbing form metrics (joint angles, shoulder/hip tilt, reach ratio)
- Writes annotated MP4 and metrics CSV/JSON

## Quick start
1. Install dependencies
2. Run the analyzer

### Install
Use your preferred environment manager, then install:
- mediapipe
- opencv-python
- numpy

Or install the project in editable mode:
```
python -m pip install -e .
```

### Run
```
python -m boulder_coach.cli --input /path/to/video.mp4 --output /path/to/output
```

## Outputs
- annotated.mp4
- metrics.csv
- summary.json
