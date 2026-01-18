from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import cv2
import google.generativeai as genai
import numpy as np
from PIL import Image

DEFAULT_MODEL = "gemini-1.5-flash"


@dataclass
class GeminiGripResult:
    hold_type: Optional[str]
    grip_type: Optional[str]
    confidence: Optional[float]
    notes: Optional[str]
    raw_text: str


class GeminiGripClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError("GEMINI_API_KEY is required to use Gemini grip analysis.")
        genai.configure(api_key=resolved_key)
        self.model = genai.GenerativeModel(model or DEFAULT_MODEL)

    def analyze_hold(self, image_bgr: np.ndarray, hand: str) -> GeminiGripResult:
        prompt = (
            "You are a climbing coach. Analyze the hand and hold in the image. "
            "Return ONLY JSON with keys: hold_type, grip_type, confidence, notes. "
            "hold_type options: jug, edge, sloper, pinch, pocket, volume, undercling, sidepull, gaston, arete, other. "
            "grip_type options: open_hand, half_crimp, full_crimp, pinch, pocket, undercling, sidepull, gaston, other. "
            "If uncertain use null and low confidence (0-1). "
            f"Hand: {hand}."
        )
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_rgb)
        response = self.model.generate_content([prompt, image])
        raw_text = (response.text or "").strip()
        parsed = _safe_parse_json(raw_text)

        hold_type = _safe_get_str(parsed, "hold_type")
        grip_type = _safe_get_str(parsed, "grip_type")
        return GeminiGripResult(
            hold_type=hold_type.lower() if hold_type else None,
            grip_type=grip_type.lower() if grip_type else None,
            confidence=_safe_get_float(parsed, "confidence"),
            notes=_safe_get_str(parsed, "notes"),
            raw_text=raw_text,
        )


def _safe_parse_json(text: str) -> dict:
    if not text:
        return {}

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def _safe_get_str(data: dict, key: str) -> Optional[str]:
    value = data.get(key)
    if value is None:
        return None
    return str(value)


def _safe_get_float(data: dict, key: str) -> Optional[float]:
    value = data.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
