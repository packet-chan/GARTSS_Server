"""
Gemini API を使った画像解析モジュール
RGB画像を送信し、指定タスクに応じたBounding Boxを取得する
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

SYSTEM_PROMPT = """You are a vision AI that detects objects in images for AR content placement.

Given an image and a task description, identify the target object/region and return its bounding box.

IMPORTANT: Return coordinates in PIXEL coordinates based on the image dimensions provided.

Respond ONLY with a JSON object in this exact format, no markdown, no explanation:
{
  "objects": [
    {
      "name": "descriptive name of the object/region",
      "bbox": [x_min, y_min, x_max, y_max],
      "confidence": "high"
    }
  ]
}

bbox coordinates must be in pixels (integers) within the image dimensions.
If you cannot find the target, return {"objects": []}.
"""


async def detect_objects(
    image_bytes: bytes,
    task: str,
    image_width: int = 1280,
    image_height: int = 1280,
    is_grayscale: bool = False,
) -> list[BBoxResult]:
    """
    Gemini API で画像内のオブジェクトを検出する

    Args:
        image_bytes: RGB画像のバイナリ（PNG or YUV raw）
        task: 検出タスクの説明（例: "コーヒーマシンのトレーを検出して"）
        image_width: 画像の幅
        image_height: 画像の高さ
        is_grayscale: グレースケール画像かどうか

    Returns:
        BBoxResult のリスト
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

    # 画像をbase64エンコード
    # PNG/JPGならそのまま、rawならPNGに変換
    import cv2
    import numpy as np

    try:
        # PNGとして読めるか試す
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
        # YUV raw data → グレースケールPNGに変換
        y_size = image_width * image_height
        if len(image_bytes) >= y_size:
            y_plane = np.frombuffer(image_bytes[:y_size], dtype=np.uint8).reshape(
                (image_height, image_width)
            )
            _, png_bytes = cv2.imencode(".png", y_plane)
            b64_image = base64.b64encode(png_bytes.tobytes()).decode("utf-8")
            mime_type = "image/png"
        else:
            raise ValueError(f"Cannot process image: size={len(image_bytes)}")

    # Gemini API リクエスト
    user_prompt = (
        f"Image dimensions: {image_width}x{image_height} pixels.\n"
        f"Task: {task}\n"
        f"Return bounding boxes in pixel coordinates."
    )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": SYSTEM_PROMPT + "\n\n" + user_prompt},
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

    # レスポンスからテキストを抽出
    try:
        text = result["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        print(f"  [Gemini] Unexpected response structure: {e}")
        print(f"  [Gemini] Response: {json.dumps(result, indent=2)[:500]}")
        return []

    # JSONを抽出（マークダウンコードブロックに入っている場合も対応）
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

    # BBoxResult に変換
    results = []
    for obj in data.get("objects", []):
        bbox = obj.get("bbox", [0, 0, 0, 0])
        # 座標をクリップ
        x_min = max(0, min(bbox[0], image_width))
        y_min = max(0, min(bbox[1], image_height))
        x_max = max(0, min(bbox[2], image_width))
        y_max = max(0, min(bbox[3], image_height))

        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

        results.append(
            BBoxResult(
                name=obj.get("name", "unknown"),
                bbox=[x_min, y_min, x_max, y_max],
                center=[cx, cy],
                confidence=obj.get("confidence", "medium"),
            )
        )

    print(f"  [Gemini] Detected {len(results)} objects")
    for r in results:
        print(f"    - {r.name}: bbox={r.bbox}, center={r.center}")

    return results
