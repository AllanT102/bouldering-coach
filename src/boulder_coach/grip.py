from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class GripObservation:
    frame: int
    hand: str
    hold_type: Optional[str]
    grip_type: Optional[str]
    confidence: Optional[float]
    expected_grip: Optional[str]
    mismatch: Optional[bool]
    notes: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


def summarize_grip_observations(observations: list[GripObservation]) -> dict:
    if not observations:
        return {
            "total_observations": 0,
            "mismatch_count": 0,
            "by_hold_type": {},
            "by_grip_type": {},
            "mismatches": [],
        }

    by_hold_type: dict[str, int] = {}
    by_grip_type: dict[str, int] = {}
    mismatches: list[dict] = []

    for obs in observations:
        if obs.hold_type:
            by_hold_type[obs.hold_type] = by_hold_type.get(obs.hold_type, 0) + 1
        if obs.grip_type:
            by_grip_type[obs.grip_type] = by_grip_type.get(obs.grip_type, 0) + 1
        if obs.mismatch:
            mismatches.append(
                {
                    "frame": obs.frame,
                    "hand": obs.hand,
                    "hold_type": obs.hold_type,
                    "grip_type": obs.grip_type,
                    "expected_grip": obs.expected_grip,
                }
            )

    return {
        "total_observations": len(observations),
        "mismatch_count": len(mismatches),
        "by_hold_type": by_hold_type,
        "by_grip_type": by_grip_type,
        "mismatches": mismatches,
    }
