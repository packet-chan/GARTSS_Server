"""Pydantic スキーマ"""

from pydantic import BaseModel, Field
from typing import Optional


class CameraPose(BaseModel):
    translation: list[float] = Field(..., min_length=3, max_length=3)
    rotation: list[float] = Field(..., min_length=4, max_length=4)


class CameraIntrinsics(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float


class CameraCharacteristics(BaseModel):
    pose: CameraPose
    intrinsics: CameraIntrinsics


class ImageFormat(BaseModel):
    width: int = 1280
    height: int = 1280


class SessionInitRequest(BaseModel):
    camera_characteristics: CameraCharacteristics
    image_format: ImageFormat = ImageFormat()


class SessionInitResponse(BaseModel):
    session_id: str


class DepthDescriptor(BaseModel):
    timestamp_ms: int
    create_pose_location_x: float
    create_pose_location_y: float
    create_pose_location_z: float
    create_pose_rotation_x: float
    create_pose_rotation_y: float
    create_pose_rotation_z: float
    create_pose_rotation_w: float
    fov_left_angle_tangent: float
    fov_right_angle_tangent: float
    fov_top_angle_tangent: float
    fov_down_angle_tangent: float
    near_z: float = 0.1
    far_z: str = "Infinity"
    width: int = 320
    height: int = 320


class HMDPose(BaseModel):
    """PoseInterpolatorと同じフィールド名を使用"""
    timestamp_ms: int
    pos_x: float
    pos_y: float
    pos_z: float
    rot_x: float
    rot_y: float
    rot_z: float
    rot_w: float


class CaptureResponse(BaseModel):
    aligned: bool
    coverage: float
    message: str = ""


class DepthQueryResponse(BaseModel):
    depth_m: Optional[float] = None
    point_3d_unity: Optional[list[float]] = None
    message: str = ""


class AnalyzeRequest(BaseModel):
    task: str


class DetectedObject(BaseModel):
    name: str
    center_2d: list[float]
    center_3d: Optional[list[float]] = None
    depth_m: Optional[float] = None
    ar_placement: Optional[dict] = None


class AnalyzeResponse(BaseModel):
    objects: list[DetectedObject] = []
    message: str = ""
