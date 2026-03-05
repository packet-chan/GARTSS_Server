"""
GARTSS Server
Quest 3からのRGB+Depthデータを受け取り、3D再投影アライメントを行う。
キャプチャごとに可視化画像を保存。
"""

import io
import json
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from core.alignment import AlignmentEngine
from core.gemini import detect_objects
from core.sam_segment import segment_with_bbox, mask_contour_to_3d, save_mask_visualization, mask_to_point_cloud, save_ply
from models.schemas import (
    SessionInitRequest, SessionInitResponse,
    DepthDescriptor, HMDPose,
    CaptureResponse, DepthQueryResponse,
    AnalyzeRequest, AnalyzeResponse, DetectedObject,
)

# 保存先ディレクトリ
OUTPUT_DIR = Path("captures")
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="GARTSS Server",
    description="Zero-shot AR Authoring - Quest 3 RGB-Depth Alignment",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict[str, dict] = {}


def decode_rgb(rgb_bytes: bytes, w: int = 1280, h: int = 1280) -> np.ndarray | None:
    """RGB画像バイト列をデコード。PNG/JPG or YUV_420_888 に対応。戻り値はBGR"""
    # まずPNG/JPGとしてデコード
    img = cv2.imdecode(np.frombuffer(rgb_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is not None:
        return img

    # YUV_420_888 (NV21) として変換
    y_size = w * h
    if len(rgb_bytes) >= y_size * 3 // 2:
        yuv_data = np.frombuffer(rgb_bytes[:y_size * 3 // 2], dtype=np.uint8)
        nv21 = np.zeros(y_size * 3 // 2, dtype=np.uint8)
        nv21[:y_size] = yuv_data[:y_size]
        nv21[y_size:] = yuv_data[y_size:y_size * 3 // 2]
        nv21 = nv21.reshape((h * 3 // 2, w))
        bgr = cv2.cvtColor(nv21, cv2.COLOR_YUV2BGR_NV12)
        print(f"  RGB: YUV_420_888 -> color ({w}x{h})")
        return bgr

    # Y plane のみ（フォールバック）
    if len(rgb_bytes) >= y_size:
        y_plane = np.frombuffer(rgb_bytes[:y_size], dtype=np.uint8).reshape((h, w))
        bgr = cv2.cvtColor(y_plane, cv2.COLOR_GRAY2BGR)
        print(f"  RGB: Y plane only -> grayscale ({w}x{h})")
        return bgr

    print(f"  RGB: Unknown format, size={len(rgb_bytes)}")
    return None


def save_visualization(session_dir: Path, capture_idx: int, engine: AlignmentEngine,
                       rgb_bytes: bytes | None, depth_raw: np.ndarray, aligned_depth: np.ndarray):
    """RGB, Depth(raw), Aligned Depth, Overlay を保存"""
    prefix = f"cap{capture_idx:03d}"

    # 1. Depth raw (NDC) → カラーマップ
    depth_vis = cv2.normalize(depth_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
    cv2.imwrite(str(session_dir / f"{prefix}_depth_raw.png"), depth_colored)
    print(f"  Saved: {prefix}_depth_raw.png ({depth_raw.shape})")

    # 2. Aligned Depth → カラーマップ
    valid = aligned_depth[aligned_depth > 0]
    if len(valid) > 0:
        vmin, vmax = np.percentile(valid, [2, 98])
    else:
        vmin, vmax = 0, 5

    aligned_norm = np.clip((aligned_depth - vmin) / max(vmax - vmin, 0.01), 0, 1)
    aligned_norm[aligned_depth <= 0] = 0
    aligned_u8 = (aligned_norm * 255).astype(np.uint8)
    aligned_colored = cv2.applyColorMap(aligned_u8, cv2.COLORMAP_TURBO)
    aligned_colored[aligned_depth <= 0] = [0, 0, 0]
    cv2.imwrite(str(session_dir / f"{prefix}_aligned_depth.png"), aligned_colored)
    print(f"  Saved: {prefix}_aligned_depth.png ({aligned_depth.shape})")

    # 3. RGB
    if rgb_bytes is not None and len(rgb_bytes) > 0:
        rgb_bgr = decode_rgb(rgb_bytes)
        if rgb_bgr is not None:
            cv2.imwrite(str(session_dir / f"{prefix}_rgb.png"), rgb_bgr)
            print(f"  Saved: {prefix}_rgb.png ({rgb_bgr.shape})")

            # 4. Overlay (RGB + Aligned Depth)
            h, w = rgb_bgr.shape[:2]
            ah, aw = aligned_colored.shape[:2]
            if (h, w) != (ah, aw):
                aligned_resized = cv2.resize(aligned_colored, (w, h), interpolation=cv2.INTER_NEAREST)
                mask = cv2.resize((aligned_depth > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                aligned_resized = aligned_colored
                mask = (aligned_depth > 0).astype(np.uint8)

            overlay = rgb_bgr.copy()
            mask_3ch = np.stack([mask] * 3, axis=-1)
            overlay = np.where(mask_3ch > 0,
                               cv2.addWeighted(rgb_bgr, 0.5, aligned_resized, 0.5, 0),
                               overlay)
            cv2.imwrite(str(session_dir / f"{prefix}_overlay.png"), overlay)
            print(f"  Saved: {prefix}_overlay.png")
    else:
        print(f"  No RGB image, skipping overlay")


@app.post("/session/init", response_model=SessionInitResponse)
async def session_init(request: SessionInitRequest):
    session_id = str(uuid.uuid4())[:8]
    engine = AlignmentEngine()
    engine.init_session(
        camera_characteristics=request.camera_characteristics.model_dump(),
        image_width=request.image_format.width,
        image_height=request.image_format.height,
    )

    # セッション用ディレクトリ作成
    session_dir = OUTPUT_DIR / session_id
    session_dir.mkdir(exist_ok=True)

    sessions[session_id] = {
        "engine": engine,
        "captures": [],
        "dir": session_dir,
        "capture_count": 0,
        "current_task": "drip_tray",
    }
    print(f"\n=== Session {session_id} initialized ===")
    print(f"  Output dir: {session_dir}")
    return SessionInitResponse(session_id=session_id)


@app.post("/session/{session_id}/capture", response_model=CaptureResponse)
async def capture(
    session_id: str,
    depth_raw: UploadFile = File(...),
    depth_descriptor: str = Form(...),
    hmd_poses: str = Form(...),
    rgb_image: UploadFile = File(None),
):
    """キャプチャデータを受け取り、RGB-Depthアライメントを実行し、可視化を保存"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    engine: AlignmentEngine = session["engine"]
    capture_idx = session["capture_count"]

    print(f"\n--- Capture {capture_idx} (session {session_id}) ---")

    try:
        desc = DepthDescriptor(**json.loads(depth_descriptor))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid depth_descriptor: {e}")

    try:
        poses = [HMDPose(**p) for p in json.loads(hmd_poses)]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid hmd_poses: {e}")

    print(f"  Depth timestamp: {desc.timestamp_ms}")
    print(f"  HMD poses: {len(poses)}")
    print(f"  Depth size: {desc.width}x{desc.height}")

    depth_bytes = await depth_raw.read()
    h, w = desc.height, desc.width
    expected_size = h * w * 4
    if len(depth_bytes) != expected_size:
        raise HTTPException(
            status_code=400,
            detail=f"Depth raw size mismatch: {len(depth_bytes)} vs {expected_size}",
        )
    depth_array = np.frombuffer(depth_bytes, dtype=np.float32).reshape((h, w))

    print(f"  Depth range: [{depth_array.min():.4f}, {depth_array.max():.4f}]")

    # HMDポーズ設定
    engine.set_hmd_poses([p.model_dump() for p in poses])

    # Depthタイムスタンプに最も近いHMDポーズを補間で取得
    pose = engine.interpolator.interpolate_pose(desc.timestamp_ms)
    if pose is None:
        raise HTTPException(status_code=400, detail="Cannot interpolate HMD pose")

    hmd_pos, hmd_rot = pose
    print(f"  HMD pos: [{hmd_pos[0]:.4f}, {hmd_pos[1]:.4f}, {hmd_pos[2]:.4f}]")

    result = engine.align(
        depth_raw=depth_array,
        depth_descriptor=desc.model_dump(),
        hmd_pos=hmd_pos,
        hmd_rot=hmd_rot,
    )

    filled = engine.fill_holes()

    coverage = result["coverage"]
    filled_coverage = np.count_nonzero(filled) / (engine.img_w * engine.img_h) * 100
    print(f"  Coverage: {coverage:.1f}% (after fill: {filled_coverage:.1f}%)")

    # RGB読み込み
    rgb_bytes_data = None
    if rgb_image is not None:
        rgb_bytes_data = await rgb_image.read()
        if len(rgb_bytes_data) > 0:
            print(f"  RGB: {len(rgb_bytes_data)} bytes")
            session["latest_rgb"] = rgb_bytes_data
        else:
            rgb_bytes_data = None

    # 可視化保存
    save_visualization(
        session_dir=session["dir"],
        capture_idx=capture_idx,
        engine=engine,
        rgb_bytes=rgb_bytes_data,
        depth_raw=depth_array,
        aligned_depth=filled,
    )

    session["capture_count"] += 1
    session["captures"].append({
        "timestamp_ms": desc.timestamp_ms,
        "coverage": coverage,
        "filled_coverage": filled_coverage,
    })

    return CaptureResponse(
        aligned=True,
        coverage=round(coverage, 1),
        message=f"Coverage: {coverage:.1f}% (filled: {filled_coverage:.1f}%)",
    )


@app.get("/session/{session_id}/depth", response_model=DepthQueryResponse)
async def depth_query(session_id: str, u: float, v: float):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    engine: AlignmentEngine = sessions[session_id]["engine"]
    depth_m = engine.get_depth_at_pixel(u, v)
    point_3d = engine.get_3d_point_unity(u, v)

    if depth_m is None:
        return DepthQueryResponse(message=f"No depth at ({u}, {v})")

    return DepthQueryResponse(
        depth_m=round(depth_m, 4),
        point_3d_unity=point_3d.tolist() if point_3d is not None else None,
        message="OK",
    )


@app.post("/session/{session_id}/analyze", response_model=AnalyzeResponse)
async def analyze(session_id: str):
    """Gemini APIでRGB画像を解析し、検出物体の3D座標を返す"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    engine: AlignmentEngine = session["engine"]

    # 最新のRGB画像を取得
    rgb_data = session.get("latest_rgb")
    if rgb_data is None:
        raise HTTPException(status_code=400, detail="No RGB image captured yet. Capture first.")

    # タスクはサーバー側で管理（PDFから自動生成予定）
    task = session.get("current_task", "drip_tray")

    print(f"\n=== Analyze (session {session_id}) ===")
    print(f"  Task: {task}")
    print(f"  RGB data: {len(rgb_data)} bytes")

    # Gemini APIで検出
    try:
        detections = await detect_objects(
            image_bytes=rgb_data,
            task=task,
            image_width=engine.img_w,
            image_height=engine.img_h,
        )
    except Exception as e:
        print(f"  [Analyze] Gemini API error: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini API error: {e}")

    # RGB画像をデコード (SAM用)
    session_dir = session["dir"]
    rgb_bgr = decode_rgb(rgb_data)

    # 検出結果に3D座標を付与 + SAMセグメンテーション
    objects = []
    for det in detections:
        u, v = det.center[0], det.center[1]

        # Depthから3D座標を取得
        point_3d = engine.get_3d_point_unity(u, v)
        depth_m = engine.get_depth_at_pixel(u, v)

        # SAMでBBox内をセグメンテーション
        contour_3d = []
        if rgb_bgr is not None:
            try:
                sam_result = segment_with_bbox(rgb_bgr, det.bbox)
                contour_3d = mask_contour_to_3d(
                    sam_result["contours"], engine,
                    simplify_epsilon=5.0,
                )

                # マスク可視化を保存
                save_mask_visualization(
                    rgb_bgr, sam_result["mask"], sam_result["contours"],
                    det.bbox,
                    str(session_dir / f"sam_result_{det.name}.png"),
                    label=det.name,
                )

                # === 点群生成 & PLY 保存 ===
                pc_result = mask_to_point_cloud(
                    mask=sam_result["mask"],
                    engine=engine,
                    image_bgr=rgb_bgr,
                    sample_step=2,  # 1/4 サンプリングで速度と密度のバランス
                )
                if pc_result["count"] > 0:
                    ply_path = str(session_dir / f"pointcloud_{det.name}.ply")
                    save_ply(
                        output_path=ply_path,
                        points=pc_result["points"],
                        colors=pc_result["colors"],
                    )
                    print(f"    Point cloud: {pc_result['count']} points → {ply_path}")

            except Exception as e:
                print(f"  [SAM2] Error: {e}")

        obj = DetectedObject(
            name=det.name,
            center_2d=det.center,
            center_3d=point_3d.tolist() if point_3d is not None else None,
            depth_m=round(depth_m, 4) if depth_m is not None else None,
            ar_placement={
                "bbox": det.bbox,
                "confidence": det.confidence,
                "contour_3d": contour_3d,
            },
        )
        objects.append(obj)

        print(f"  Object: {det.name}")
        print(f"    2D center: ({u:.0f}, {v:.0f})")
        print(f"    Depth: {depth_m:.4f}m" if depth_m else "    Depth: N/A")
        print(f"    3D: {point_3d}" if point_3d is not None else "    3D: N/A")
        print(f"    Contour 3D vertices: {len(contour_3d)}")

    # 可視化: BBoxをRGB画像に描画して保存
    _save_analyze_visualization(session, detections)

    return AnalyzeResponse(
        objects=objects,
        message=f"Detected {len(objects)} objects",
    )


def _save_analyze_visualization(session, detections):
    """検出結果をRGB画像に描画して保存"""
    rgb_data = session.get("latest_rgb")
    if rgb_data is None:
        return

    session_dir = session["dir"]

    try:
        img_array = decode_rgb(rgb_data)
        if img_array is None:
            return

        # BBoxを描画
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            cx, cy = int(det.center[0]), int(det.center[1])

            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(img_array, (cx, cy), 8, (0, 0, 255), -1)
            cv2.putText(img_array, det.name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        output_path = session_dir / "analyze_result.png"
        cv2.imwrite(str(output_path), img_array)
        print(f"  Saved: analyze_result.png")

    except Exception as e:
        print(f"  Failed to save analyze visualization: {e}")


@app.get("/health")
async def health():
    return {"status": "ok", "active_sessions": len(sessions)}


@app.put("/session/{session_id}/task")
async def set_task(session_id: str, task: str):
    """現在の検出タスクを設定（将来的にPDFパイプラインから自動設定）"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    sessions[session_id]["current_task"] = task
    print(f"  Task updated: {task}")
    return {"session_id": session_id, "task": task}


@app.get("/session/{session_id}/info")
async def session_info(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]
    engine: AlignmentEngine = session["engine"]
    return {
        "session_id": session_id,
        "initialized": engine._initialized,
        "has_aligned_depth": engine._aligned_depth is not None,
        "captures_count": session["capture_count"],
        "image_size": f"{engine.img_w}x{engine.img_h}",
        "output_dir": str(session["dir"]),
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del sessions[session_id]
    return {"deleted": session_id}