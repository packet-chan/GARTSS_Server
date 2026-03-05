"""
Gemini API を使った画像解析モジュール

改善点 v3:
- 正規化座標 (0-1000) を採用 → Geminiのネイティブフォーマット
- 物理クロップ廃止 → 全体画像 + 論理的空間ヒントで2段階検出
- [ymin, xmin, ymax, xmax] フォーマット (Gemini標準)
"""

import base64
import json
import os
import re
from dataclasses import dataclass

import cv2
import httpx
import numpy as np


@dataclass
class BBoxResult:
    name: str
    bbox: list[float]  # [x_min, y_min, x_max, y_max] in pixel coordinates
    center: list[float]  # [cx, cy] in pixel coordinates
    confidence: str  # "high", "medium", "low"


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# タスクごとの親オブジェクトと検出ヒント
TASK_CONTEXT = {
    "drip_tray": {
        "parent": "coffee machine",
        "description": "the flat removable drip tray at the bottom/base of the coffee machine where liquid drips collect",
        "hints": "It is a wide, shallow, horizontal platform that sticks out from the front of the machine at the very bottom. Do NOT include the machine body above the tray.",
    },
    "water_tank": {
        "parent": "coffee machine",
        "description": "the removable water tank/reservoir of the coffee machine",
        "hints": "Usually located at the back or side of the machine. It is a transparent or semi-transparent container.",
    },
    "power_button": {
        "parent": "coffee machine",
        "description": "the power button or on/off switch on the coffee machine",
        "hints": "Usually a circular button or rocker switch on the front or top of the machine.",
    },
}

DEFAULT_CONTEXT = {
    "parent": None,
    "description": "the target component",
    "hints": "",
}


def _encode_image(image_bytes: bytes, image_width: int, image_height: int) -> tuple[str, str]:
    """画像バイト列をbase64エンコード。PNG/JPG/YUV対応。"""
    try:
        img_array = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if img_array is not None:
            _, png_bytes = cv2.imencode(".png", img_array)
            return base64.b64encode(png_bytes.tobytes()).decode("utf-8"), "image/png"
        raise ValueError("Not a valid image format")
    except Exception:
        y_size = image_width * image_height
        if len(image_bytes) >= y_size * 3 // 2:
            yuv_data = np.frombuffer(image_bytes[:y_size * 3 // 2], dtype=np.uint8)
            nv21 = np.zeros(y_size * 3 // 2, dtype=np.uint8)
            nv21[:y_size] = yuv_data[:y_size]
            nv21[y_size:] = yuv_data[y_size:y_size * 3 // 2]
            nv21 = nv21.reshape((image_height * 3 // 2, image_width))
            rgb_img = cv2.cvtColor(nv21, cv2.COLOR_YUV2BGR_NV12)
            _, png_bytes = cv2.imencode(".png", rgb_img)
            print(f"  [Gemini] Converted YUV to color PNG")
            return base64.b64encode(png_bytes.tobytes()).decode("utf-8"), "image/png"
        elif len(image_bytes) >= y_size:
            y_plane = np.frombuffer(image_bytes[:y_size], dtype=np.uint8).reshape(
                (image_height, image_width)
            )
            _, png_bytes = cv2.imencode(".png", y_plane)
            return base64.b64encode(png_bytes.tobytes()).decode("utf-8"), "image/png"
        raise ValueError(f"Cannot process image: size={len(image_bytes)}")


def _normalized_to_pixel(bbox_norm: list[float], image_width: int, image_height: int) -> list[float]:
    """
    Gemini正規化座標 [ymin, xmin, ymax, xmax] (0-1000)
    → ピクセル座標 [x_min, y_min, x_max, y_max] に変換
    """
    ymin, xmin, ymax, xmax = bbox_norm
    x_min = xmin / 1000.0 * image_width
    y_min = ymin / 1000.0 * image_height
    x_max = xmax / 1000.0 * image_width
    y_max = ymax / 1000.0 * image_height
    return [x_min, y_min, x_max, y_max]


async def _call_gemini(prompt: str, b64_image: str, mime_type: str) -> dict | None:
    """Gemini API を呼び出して JSON レスポンスを返す"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": b64_image,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 1024,
        },
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()

    result = response.json()

    try:
        text = result["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        print(f"  [Gemini] Unexpected response structure: {e}")
        return None

    print(f"  [Gemini] Raw response: {text[:500]}")

    text = text.strip()
    json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  [Gemini] Failed to parse JSON: {e}")
        return None


def _parse_normalized_bbox(data: dict) -> list[float] | None:
    """JSON レスポンスから正規化 BBox [ymin, xmin, ymax, xmax] を抽出"""
    bbox = data.get("bbox")
    if bbox is None or len(bbox) != 4:
        return None

    ymin, xmin, ymax, xmax = [float(v) for v in bbox]

    # 範囲チェック (0-1000)
    ymin = max(0, min(ymin, 1000))
    xmin = max(0, min(xmin, 1000))
    ymax = max(0, min(ymax, 1000))
    xmax = max(0, min(xmax, 1000))

    if xmin >= xmax or ymin >= ymax:
        print(f"  [Gemini] Invalid bbox: [{ymin},{xmin},{ymax},{xmax}]")
        return None

    return [ymin, xmin, ymax, xmax]


async def detect_objects(
    image_bytes: bytes,
    task: str,
    image_width: int = 1280,
    image_height: int = 1280,
    is_grayscale: bool = False,
) -> list[BBoxResult]:
    """
    Gemini API で画像内のオブジェクトを検出する。

    2段階検出（物理クロップなし）:
      Stage 1: 親オブジェクトの位置を正規化座標で取得
      Stage 2: 全体画像 + 親の位置情報をヒントにターゲットを検出

    正規化座標 [ymin, xmin, ymax, xmax] (0-1000) を使用。
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

    b64_image, mime_type = _encode_image(image_bytes, image_width, image_height)

    context = TASK_CONTEXT.get(task, DEFAULT_CONTEXT)
    parent_name = context.get("parent")

    # =========================================================
    #  Stage 1: 親オブジェクトの検出（正規化座標）
    # =========================================================
    parent_bbox_norm = None
    if parent_name:
        stage1_prompt = f"""You are a precise object detection system.

Locate the {parent_name} in this image.
Return the bounding box in NORMALIZED coordinates scaled from 0 to 1000.

Use the format [ymin, xmin, ymax, xmax] where:
- ymin: top edge (0 = top of image, 1000 = bottom)
- xmin: left edge (0 = left of image, 1000 = right)
- ymax: bottom edge
- xmax: right edge

The box must enclose the ENTIRE {parent_name} including ALL of its parts (body, base, tray, buttons, everything).

Respond ONLY with JSON:
{{"name": "{parent_name}", "bbox": [ymin, xmin, ymax, xmax]}}"""

        print(f"  [Gemini] Stage 1: Detecting parent '{parent_name}'...")
        data = await _call_gemini(stage1_prompt, b64_image, mime_type)
        if data:
            parent_bbox_norm = _parse_normalized_bbox(data)
            if parent_bbox_norm:
                print(f"  [Gemini] Stage 1: Parent norm bbox = {parent_bbox_norm}")

    # =========================================================
    #  Stage 2: ターゲット検出（全体画像 + 空間ヒント）
    # =========================================================
    description = context.get("description", task)
    hints = context.get("hints", "")

    # 親の位置情報を論理ヒントとして構築
    spatial_hint = ""
    if parent_bbox_norm:
        ymin_p, xmin_p, ymax_p, xmax_p = parent_bbox_norm
        spatial_hint = (
            f"\nSPATIAL CONTEXT: The {parent_name} is located at normalized coordinates "
            f"[ymin={ymin_p:.0f}, xmin={xmin_p:.0f}, ymax={ymax_p:.0f}, xmax={xmax_p:.0f}] in this image. "
            f"Use this location as reference to find the {task}, which is part of the {parent_name}."
        )

    stage2_prompt = f"""You are a precise object detection system for AR work assistance.

Find the {task} in this image.
Description: {description}
{f'Hints: {hints}' if hints else ''}{spatial_hint}

Return the bounding box in NORMALIZED coordinates scaled from 0 to 1000.

Use the format [ymin, xmin, ymax, xmax] where:
- ymin: top edge (0 = top of image, 1000 = bottom)
- xmin: left edge (0 = left of image, 1000 = right)
- ymax: bottom edge
- xmax: right edge

The box must TIGHTLY enclose the ENTIRE {task} with a small margin (~3%).
Be very precise about the boundaries.

Respond ONLY with JSON:
{{"name": "{task}", "bbox": [ymin, xmin, ymax, xmax]}}

If not found:
{{"name": "not_found", "bbox": [0, 0, 0, 0]}}"""

    stage_label = "Stage 2" if parent_bbox_norm else "Single-stage"
    print(f"  [Gemini] {stage_label}: Detecting '{task}'...")
    data = await _call_gemini(stage2_prompt, b64_image, mime_type)

    if data is None:
        print(f"  [Gemini] Detection failed for '{task}'")
        return []

    target_bbox_norm = _parse_normalized_bbox(data)
    if target_bbox_norm is None:
        name = data.get("name", "")
        if name == "not_found":
            print(f"  [Gemini] Target '{task}' not found")
        return []

    # 正規化座標 → ピクセル座標に変換
    pixel_bbox = _normalized_to_pixel(target_bbox_norm, image_width, image_height)

    cx = (pixel_bbox[0] + pixel_bbox[2]) / 2
    cy = (pixel_bbox[1] + pixel_bbox[3]) / 2

    confidence = "high" if parent_bbox_norm else "medium"

    print(f"  [Gemini] {stage_label}: norm bbox = {target_bbox_norm}")
    print(f"  [Gemini] Final: pixel bbox = [{pixel_bbox[0]:.0f},{pixel_bbox[1]:.0f},{pixel_bbox[2]:.0f},{pixel_bbox[3]:.0f}], center=({cx:.0f},{cy:.0f})")

    return [BBoxResult(
        name=task,
        bbox=pixel_bbox,
        center=[cx, cy],
        confidence=confidence,
    )]