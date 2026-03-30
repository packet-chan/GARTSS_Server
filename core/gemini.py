"""
Gemini API を使った画像解析モジュール

改善点 v3:
- 正規化座標 (0-1000) を採用 → Geminiのネイティブフォーマット
- 物理クロップ廃止 → 全体画像 + 論理的空間ヒントで2段階検出
- [ymin, xmin, ymax, xmax] フォーマット (Gemini標準)

改善点 v4:
- arrow_origin (矢印起点の2Dピクセル座標) をGeminiに返させる
- TASK_CONTEXT に arrow_origin_hint を追加
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
    arrow_origin: list[float] | None = None  # [x, y] 矢印起点のピクセル座標 (Gemini指定)


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# タスクごとの親オブジェクトと検出ヒント
TASK_CONTEXT = {
    "drip_tray": {
        "parent": "coffee machine",
        "description": "the drip tray",
        "hints": "",
        "arrow_origin_hint": "the center of the FRONT FACE (vertical surface) of the drip tray, NOT the top surface",
    },
    "rotary_knob": {
        "parent": "coffee machine",
        "description": "the left rotary knob",
        "hints": "Target only the LEFT knob.",
        "arrow_origin_hint": "the center of the knob surface",
    },
    "water_tank": {
        "parent": "coffee machine",
        "description": "the water tank",
        "hints": "",
        "arrow_origin_hint": "the top center of the water tank",
    },
    "power_button": {
        "parent": "coffee machine",
        "description": "the power button",
        "hints": "",
        "arrow_origin_hint": "the center of the button surface",
    },
}

DEFAULT_CONTEXT = {
    "parent": None,
    "description": "the target component",
    "hints": "",
    "arrow_origin_hint": "",
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
    """Gemini API を呼び出して JSON レスポンスを返す。429 なら Lite にフォールバック。"""

    models = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-3.1-flash-lite",
    ]

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
            "maxOutputTokens": 2048,
        },
    }

    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload)
                if response.status_code == 429:
                    print(f"  [Gemini] {model} → 429 Rate Limited, trying next model...")
                    continue
                response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                print(f"  [Gemini] {model} → 429 Rate Limited, trying next model...")
                continue
            raise

        print(f"  [Gemini] Using model: {model}")
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

    # 全モデルが 429 だった場合
    print(f"  [Gemini] All models rate limited!")
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

    v4: arrow_origin (矢印起点) も同時に返す。
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
        stage1_prompt = f"""Locate the {parent_name} in this image.
Return bounding box in normalized coordinates (0-1000), format [ymin, xmin, ymax, xmax].
Enclose the ENTIRE {parent_name} including all parts.

Respond with single-line compact JSON only:
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
    arrow_origin_hint = context.get("arrow_origin_hint", "")

    # 親の位置情報を論理ヒントとして構築
    spatial_hint = ""
    if parent_bbox_norm:
        ymin_p, xmin_p, ymax_p, xmax_p = parent_bbox_norm
        spatial_hint = (
            f"\nSPATIAL CONTEXT: The {parent_name} is located at normalized coordinates "
            f"[ymin={ymin_p:.0f}, xmin={xmin_p:.0f}, ymax={ymax_p:.0f}, xmax={xmax_p:.0f}] in this image. "
            f"Use this location as reference to find the {task}, which is part of the {parent_name}."
        )

    stage2_prompt = f"""Find {description} of the {parent_name if parent_name else 'device'} in this image.
{f'{hints}' if hints else ''}{spatial_hint}

Return bounding box in normalized coordinates (0-1000), format [ymin, xmin, ymax, xmax].
The box must enclose the ENTIRE component including all visible surfaces.

Also provide "arrow_origin": [y, x] — the point where a user would interact with this component.
{f'Specifically: {arrow_origin_hint}.' if arrow_origin_hint else ''}

Respond with single-line compact JSON only:
{{"name": "{task}", "bbox": [ymin, xmin, ymax, xmax], "arrow_origin": [y, x]}}

If not found: {{"name": "not_found", "bbox": [0,0,0,0], "arrow_origin": [0,0]}}"""

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

    # arrow_origin の解析 (正規化 [y, x] → ピクセル [x, y])
    arrow_origin_pixel = None
    raw_origin = data.get("arrow_origin")
    if raw_origin and len(raw_origin) == 2:
        oy_norm, ox_norm = float(raw_origin[0]), float(raw_origin[1])
        if oy_norm > 0 or ox_norm > 0:  # [0,0] は未検出扱い
            oy_norm = max(0, min(oy_norm, 1000))
            ox_norm = max(0, min(ox_norm, 1000))
            arrow_origin_pixel = [
                ox_norm / 1000.0 * image_width,
                oy_norm / 1000.0 * image_height,
            ]
            print(f"  [Gemini] Arrow origin (norm): [y={oy_norm:.0f}, x={ox_norm:.0f}] → pixel: ({arrow_origin_pixel[0]:.0f}, {arrow_origin_pixel[1]:.0f})")

    confidence = "high" if parent_bbox_norm else "medium"

    print(f"  [Gemini] {stage_label}: norm bbox = {target_bbox_norm}")
    print(f"  [Gemini] Final: pixel bbox = [{pixel_bbox[0]:.0f},{pixel_bbox[1]:.0f},{pixel_bbox[2]:.0f},{pixel_bbox[3]:.0f}], center=({cx:.0f},{cy:.0f})")

    return [BBoxResult(
        name=task,
        bbox=pixel_bbox,
        center=[cx, cy],
        confidence=confidence,
        arrow_origin=arrow_origin_pixel,
    )]