"""
Gemini API を使った画像解析モジュール
RGB画像を送信し、指定タスクに応じたBounding Boxを取得する
ピクセル座標で直接返させる方式
"""

import base64
import json
import os
import re
from dataclasses import dataclass

import httpx


@dataclass
class BBoxResult:
    name: str
    bbox: list[float]  # [x_min, y_min, x_max, y_max] in pixel coordinates
    center: list[float]  # [cx, cy] in pixel coordinates
    confidence: str  # "high", "medium", "low"


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


async def detect_objects(
    image_bytes: bytes,
    task: str,
    image_width: int = 1280,
    image_height: int = 1280,
    is_grayscale: bool = False,
) -> list[BBoxResult]:
    """
    Gemini API で画像内のオブジェクトを検出する
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

    import cv2
    import numpy as np

    # 画像をbase64エンコード
    try:
        img_array = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if img_array is not None:
            _, png_bytes = cv2.imencode(".png", img_array)
            b64_image = base64.b64encode(png_bytes.tobytes()).decode("utf-8")
            mime_type = "image/png"
        else:
            raise ValueError("Not a valid image format")
    except Exception:
        y_size = image_width * image_height
        if len(image_bytes) >= y_size * 3 // 2:
            yuv_data = np.frombuffer(image_bytes[:y_size * 3 // 2], dtype=np.uint8)
            nv21 = np.zeros(y_size * 3 // 2, dtype=np.uint8)
            nv21[:y_size] = yuv_data[:y_size]
            nv21[y_size:] = yuv_data[y_size:y_size * 3 // 2]
            nv21 = nv21.reshape((image_height * 3 // 2, image_width))
            rgb_img = cv2.cvtColor(nv21, cv2.COLOR_YUV2BGR_NV21)
            _, png_bytes = cv2.imencode(".png", rgb_img)
            b64_image = base64.b64encode(png_bytes.tobytes()).decode("utf-8")
            mime_type = "image/png"
            print(f"  [Gemini] Converted YUV to color PNG")
        elif len(image_bytes) >= y_size:
            y_plane = np.frombuffer(image_bytes[:y_size], dtype=np.uint8).reshape(
                (image_height, image_width)
            )
            _, png_bytes = cv2.imencode(".png", y_plane)
            b64_image = base64.b64encode(png_bytes.tobytes()).decode("utf-8")
            mime_type = "image/png"
        else:
            raise ValueError(f"Cannot process image: size={len(image_bytes)}")

    # プロンプト: ピクセル座標で直接返させる
    system_prompt = f"""You are an object detection system for AR work assistance.

This image is {image_width}x{image_height} pixels.

Given the image and a target component name, locate the component and return its bounding box in PIXEL coordinates.

RULES:
- Coordinates are in PIXELS, not normalized
- Format: [x_min, y_min, x_max, y_max]
- x ranges from 0 (left) to {image_width} (right)
- y ranges from 0 (top) to {image_height} (bottom)
- x_min < x_max and y_min < y_max
- The box must tightly enclose the ENTIRE target object with ~3% margin
- Look carefully at the image and identify the exact object boundaries

Respond ONLY with JSON, no markdown, no explanation:
{{"name": "component_name", "bbox": [x_min, y_min, x_max, y_max]}}

If not found:
{{"name": "not_found", "bbox": [0, 0, 0, 0]}}"""

    user_prompt = (
        f"I want to identify where the {task} is in this {image_width}x{image_height} pixel image.\n"
        f"The drip_tray is the flat removable tray at the very bottom/base of the coffee machine where drips collect. "
        f"It is a wide, shallow, horizontal platform that sticks out from the front of the machine. "
        f"Do NOT include the machine body above the tray.\n"
        f"The bounding box must enclose the ENTIRE tray from its left edge to right edge, top surface to bottom. "
        f"Make the box slightly larger than the tray (5% margin).\n"
        f"Output the bounding box in pixel coordinates [x_min, y_min, x_max, y_max]."
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": system_prompt + "\n\n" + user_prompt},
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
        print(f"  [Gemini] Response: {json.dumps(result, indent=2)[:500]}")
        return []

    print(f"  [Gemini] Raw response: {text[:500]}")

    # JSONを抽出
    text = text.strip()
    json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  [Gemini] Failed to parse JSON: {e}")
        print(f"  [Gemini] Raw text: {text[:300]}")
        return []

    # BBoxResult に変換（ピクセル座標をそのまま使用）
    results = []

    if "bbox" in data:
        objects_list = [data]
    elif "targets" in data:
        objects_list = data["targets"]
    elif "objects" in data:
        objects_list = data["objects"]
    else:
        objects_list = [data]

    for obj in objects_list:
        bbox = obj.get("bbox")
        name = obj.get("name") or obj.get("label") or "unknown"

        if bbox is None or name == "not_found":
            continue
        if len(bbox) != 4:
            continue

        x_min, y_min, x_max, y_max = bbox

        # クリップ
        x_min = max(0, min(x_min, image_width))
        y_min = max(0, min(y_min, image_height))
        x_max = max(0, min(x_max, image_width))
        y_max = max(0, min(y_max, image_height))

        if x_min >= x_max or y_min >= y_max:
            print(f"  [Gemini] Invalid bbox: x=[{x_min},{x_max}], y=[{y_min},{y_max}]")
            continue

        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

        results.append(
            BBoxResult(
                name=name,
                bbox=[x_min, y_min, x_max, y_max],
                center=[cx, cy],
                confidence="high",
            )
        )

    print(f"  [Gemini] Detected {len(results)} objects")
    for r in results:
        print(f"    - {r.name}: bbox=[{r.bbox[0]:.0f},{r.bbox[1]:.0f},{r.bbox[2]:.0f},{r.bbox[3]:.0f}], center=({r.center[0]:.0f},{r.center[1]:.0f})")

    return results