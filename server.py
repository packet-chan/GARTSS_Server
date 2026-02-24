"""
AR Work Assist Server
Quest 3からのRGB+Depthデータを受け取り、3D再投影アライメントを行う。
"""

import io
import json
import uuid
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from core.alignment import AlignmentEngine
from models.schemas import (
    SessionInitRequest, SessionInitResponse,
    DepthDescriptor, HMDPose,
    CaptureResponse, DepthQueryResponse,
    AnalyzeRequest, AnalyzeResponse,
)

app = FastAPI(
    title="AR Work Assist Server",
    description="Zero-shot AR Authoring - Quest 3 RGB-Depth Alignment",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions: dict[str, dict] = {}


@app.post("/session/init", response_model=SessionInitResponse)
async def session_init(request: SessionInitRequest):
    session_id = str(uuid.uuid4())[:8]
    engine = AlignmentEngine()
    engine.init_session(
        camera_characteristics=request.camera_characteristics.model_dump(),
        image_width=request.image_format.width,
        image_height=request.image_format.height,
    )
    sessions[session_id] = {"engine": engine, "captures": []}
    return SessionInitResponse(session_id=session_id)


@app.post("/session/{session_id}/capture", response_model=CaptureResponse)
async def capture(
    session_id: str,
    depth_raw: UploadFile = File(...),
    depth_descriptor: str = Form(...),
    hmd_poses: str = Form(...),
    rgb_image: UploadFile = File(None),
):
    """キャプチャデータを受け取り、RGB-Depthアライメントを実行"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    engine: AlignmentEngine = session["engine"]

    try:
        desc = DepthDescriptor(**json.loads(depth_descriptor))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid depth_descriptor: {e}")

    try:
        poses = [HMDPose(**p) for p in json.loads(hmd_poses)]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid hmd_poses: {e}")

    depth_bytes = await depth_raw.read()
    h, w = desc.height, desc.width
    expected_size = h * w * 4
    if len(depth_bytes) != expected_size:
        raise HTTPException(
            status_code=400,
            detail=f"Depth raw size mismatch: {len(depth_bytes)} vs {expected_size}",
        )
    depth_array = np.frombuffer(depth_bytes, dtype=np.float32).reshape((h, w))

    # HMDポーズ設定
    engine.set_hmd_poses([p.model_dump() for p in poses])

    # Depthタイムスタンプに最も近いHMDポーズを補間で取得
    pose = engine.interpolator.interpolate_pose(desc.timestamp_ms)
    if pose is None:
        raise HTTPException(status_code=400, detail="Cannot interpolate HMD pose")

    hmd_pos, hmd_rot = pose

    result = engine.align(
        depth_raw=depth_array,
        depth_descriptor=desc.model_dump(),
        hmd_pos=hmd_pos,
        hmd_rot=hmd_rot,
    )

    engine.fill_holes()

    if rgb_image is not None:
        rgb_bytes = await rgb_image.read()
        session["captures"].append({
            "rgb_bytes": rgb_bytes,
            "depth_descriptor": desc.model_dump(),
            "coverage": result["coverage"],
        })

    return CaptureResponse(
        aligned=True,
        coverage=round(result["coverage"], 1),
        message=f"Coverage: {result['coverage']:.1f}%",
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
async def analyze(session_id: str, request: AnalyzeRequest):
    """[Phase 3 stub] LLM + SAMによる画像解析"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return AnalyzeResponse(message="Phase 3: Not yet implemented")


@app.get("/health")
async def health():
    return {"status": "ok", "active_sessions": len(sessions)}


@app.get("/session/{session_id}/info")
async def session_info(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    engine: AlignmentEngine = sessions[session_id]["engine"]
    return {
        "session_id": session_id,
        "initialized": engine._initialized,
        "has_aligned_depth": engine._aligned_depth is not None,
        "captures_count": len(sessions[session_id]["captures"]),
        "image_size": f"{engine.img_w}x{engine.img_h}",
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del sessions[session_id]
    return {"deleted": session_id}
