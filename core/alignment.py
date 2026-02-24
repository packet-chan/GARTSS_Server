"""
RGB-Depth アライメントエンジン

Quest 3のDepthマップを3D再投影でRGB画像座標系にアライメントする。
reproject_align_v2.py の DepthToRGBProjector をサーバー向けにリファクタ。
"""

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from typing import Optional

from .coordinate_utils import (
    convert_camera2_pose,
    apply_local_transform,
    convert_unity_to_open3d_camera,
    convert_open3d_to_unity_point,
    compute_extrinsic_wc,
    ndc_to_linear,
    compute_depth_intrinsics,
)
from .pose_interpolator import PoseInterpolator


class AlignmentEngine:
    """
    Quest 3 RGB-Depth 3D再投影アライメントエンジン

    使い方:
        1. init_session() でカメラパラメータを設定
        2. set_hmd_poses() でHMDポーズを設定
        3. align() でDepthフレームをRGB座標系にアライメント
        4. get_depth_at_pixel() / get_3d_point_unity() でクエリ
    """

    def __init__(self):
        self.rgb_local_t = None
        self.rgb_local_r = None
        self.r_fx = 0.0
        self.r_fy = 0.0
        self.r_cx = 0.0
        self.r_cy = 0.0
        self.r_cx_o3d = 0.0
        self.img_w = 0
        self.img_h = 0

        self.interpolator = PoseInterpolator()

        self._aligned_depth = None
        self._rgb_ext_cw = None  # Camera-to-World (4x4)
        self._initialized = False

    def init_session(self, camera_characteristics, image_width, image_height):
        """セッション初期化: カメラパラメータを設定"""
        raw_t = camera_characteristics["pose"]["translation"]
        raw_r = camera_characteristics["pose"]["rotation"]
        self.rgb_local_t, self.rgb_local_r = convert_camera2_pose(raw_t, raw_r)

        intrinsics = camera_characteristics["intrinsics"]
        self.r_fx = intrinsics["fx"]
        self.r_fy = intrinsics["fy"]
        self.r_cx = intrinsics["cx"]
        self.r_cy = intrinsics["cy"]

        self.img_w = image_width
        self.img_h = image_height
        self.r_cx_o3d = self.img_w - self.r_cx

        self._initialized = True

    def set_hmd_poses(self, poses):
        """HMDポーズデータを設定"""
        self.interpolator.load_poses(poses)

    def _get_rgb_camera_extrinsic(self, hmd_pos, hmd_rot):
        """HMD姿勢からRGBカメラのOpen3D座標系での外部行列を計算"""
        hmd_pos = hmd_pos.reshape(1, 3)
        hmd_rot = hmd_rot.reshape(1, 4)

        cam_pos_unity, cam_rot_unity = apply_local_transform(
            hmd_pos, hmd_rot, self.rgb_local_t, self.rgb_local_r
        )
        cam_pos_o3d, cam_rot_o3d = convert_unity_to_open3d_camera(
            cam_pos_unity, cam_rot_unity
        )

        ext_wc = compute_extrinsic_wc(cam_pos_o3d, cam_rot_o3d)[0]
        ext_cw = np.linalg.inv(ext_wc)
        return ext_wc, ext_cw

    def _get_depth_camera_extrinsic(self, depth_descriptor):
        """Depthカメラの World-to-Camera外部行列(4x4)をOpen3D座標系で取得"""
        pos = np.array([[
            float(depth_descriptor["create_pose_location_x"]),
            float(depth_descriptor["create_pose_location_y"]),
            float(depth_descriptor["create_pose_location_z"]),
        ]])
        rot = np.array([[
            float(depth_descriptor["create_pose_rotation_x"]),
            float(depth_descriptor["create_pose_rotation_y"]),
            float(depth_descriptor["create_pose_rotation_z"]),
            float(depth_descriptor["create_pose_rotation_w"]),
        ]])

        pos_o3d, rot_o3d = convert_unity_to_open3d_camera(pos, rot)
        ext_wc = compute_extrinsic_wc(pos_o3d, rot_o3d)
        return ext_wc[0]

    def align(self, depth_raw, depth_descriptor, hmd_pos, hmd_rot):
        """
        Depthマップを3D経由でRGB画像に正確に再投影。

        Args:
            depth_raw: (H, W) float32 NDCバッファ
            depth_descriptor: dict (FoV, near/far, pose等)
            hmd_pos: (3,) HMD位置 (Unity座標系)
            hmd_rot: (4,) HMD回転 [x, y, z, w] (Unity座標系)

        Returns:
            dict with aligned_depth and coverage
        """
        if not self._initialized:
            raise RuntimeError("init_session() を先に実行してください")

        h, w = depth_raw.shape

        # 1. NDC -> リニア変換
        near = float(depth_descriptor["near_z"])
        far_str = depth_descriptor.get("far_z", "Infinity")
        far = float("inf") if far_str == "Infinity" else float(far_str)
        depth_linear = ndc_to_linear(depth_raw, near, far)

        # 2. Depth内部パラメータ
        d_intr = compute_depth_intrinsics(
            width=w, height=h,
            fov_left=float(depth_descriptor["fov_left_angle_tangent"]),
            fov_right=float(depth_descriptor["fov_right_angle_tangent"]),
            fov_top=float(depth_descriptor["fov_top_angle_tangent"]),
            fov_bottom=float(depth_descriptor["fov_down_angle_tangent"]),
        )
        d_fx, d_fy = d_intr["fx"], d_intr["fy"]
        d_cx_o3d, d_cy = d_intr["cx_o3d"], d_intr["cy"]

        # 3. 外部行列の取得 (Open3D座標系)
        depth_ext_wc = self._get_depth_camera_extrinsic(depth_descriptor)
        rgb_ext_wc, rgb_ext_cw = self._get_rgb_camera_extrinsic(hmd_pos, hmd_rot)

        # 4. Depthピクセル -> Depthカメラ3D
        v_grid, u_grid = np.mgrid[0:h, 0:w]
        valid = (depth_linear > 0) & np.isfinite(depth_linear)
        d_vals = depth_linear[valid]
        u_vals = u_grid[valid].astype(np.float64)
        v_vals = v_grid[valid].astype(np.float64)

        x_cam = (u_vals - d_cx_o3d) / d_fx * d_vals
        y_cam = (v_vals - d_cy) / d_fy * d_vals
        z_cam = d_vals

        # 5. Depthカメラ3D -> ワールド3D
        pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=1)
        depth_ext_cw = np.linalg.inv(depth_ext_wc)
        pts_world = (depth_ext_cw @ pts_cam.T).T[:, :3]

        # 6. ワールド3D -> RGBカメラ3D
        pts_world_h = np.hstack([pts_world, np.ones((len(pts_world), 1))])
        pts_rgb_cam = (rgb_ext_wc @ pts_world_h.T).T[:, :3]

        # 7. RGBカメラ3D -> RGBピクセル
        in_front = pts_rgb_cam[:, 2] > 0.01
        pts_rgb_cam = pts_rgb_cam[in_front]
        d_rgb = d_vals[in_front]

        u_rgb = (pts_rgb_cam[:, 0] / pts_rgb_cam[:, 2]) * self.r_fx + self.r_cx_o3d
        v_rgb = (pts_rgb_cam[:, 1] / pts_rgb_cam[:, 2]) * self.r_fy + self.r_cy

        # 8. RGB画像にDepth書き込み (Z-buffer)
        aligned_depth = np.zeros((self.img_h, self.img_w), dtype=np.float32)

        u_int = np.round(u_rgb).astype(np.int32)
        v_int = np.round(v_rgb).astype(np.int32)
        in_bounds = (
            (u_int >= 0) & (u_int < self.img_w) &
            (v_int >= 0) & (v_int < self.img_h)
        )

        u_valid = u_int[in_bounds]
        v_valid = v_int[in_bounds]
        d_valid = d_rgb[in_bounds]

        sort_idx = np.argsort(-d_valid)
        u_valid = u_valid[sort_idx]
        v_valid = v_valid[sort_idx]
        d_valid = d_valid[sort_idx]

        aligned_depth[v_valid, u_valid] = d_valid

        self._aligned_depth = aligned_depth
        self._rgb_ext_cw = rgb_ext_cw

        coverage = np.count_nonzero(aligned_depth) / (self.img_h * self.img_w) * 100

        return {"aligned_depth": aligned_depth, "coverage": coverage}

    # =========================================================================
    #  クエリ API
    # =========================================================================

    def get_depth_at_pixel(self, u, v, aligned_depth=None):
        """RGB画像のピクセル (u, v) のDepth [m] を返す"""
        if aligned_depth is None:
            aligned_depth = self._aligned_depth
        if aligned_depth is None:
            return None

        u_int, v_int = int(round(u)), int(round(v))
        if not (0 <= u_int < self.img_w and 0 <= v_int < self.img_h):
            return None
        d = float(aligned_depth[v_int, u_int])
        return d if d > 0 else None

    def get_depth_patch(self, u, v, radius=2, aligned_depth=None):
        """周辺パッチのメディアンDepth [m]"""
        if aligned_depth is None:
            aligned_depth = self._aligned_depth
        if aligned_depth is None:
            return None

        ui, vi = int(round(u)), int(round(v))
        patch = aligned_depth[
            max(0, vi - radius) : min(self.img_h, vi + radius + 1),
            max(0, ui - radius) : min(self.img_w, ui + radius + 1),
        ]
        valid = patch[patch > 0]
        return float(np.median(valid)) if len(valid) > 0 else None

    def get_3d_point_unity(self, u, v, aligned_depth=None):
        """RGB画像のピクセル (u, v) -> Unity ワールド座標系の3D点 [m]"""
        depth_m = self.get_depth_patch(u, v, radius=3, aligned_depth=aligned_depth)
        if depth_m is None:
            return None
        if self._rgb_ext_cw is None:
            return None

        x_cam = (u - self.r_cx_o3d) / self.r_fx * depth_m
        y_cam = (v - self.r_cy) / self.r_fy * depth_m
        z_cam = depth_m

        point_cam = np.array([x_cam, y_cam, z_cam, 1.0])
        point_world_o3d = (self._rgb_ext_cw @ point_cam)[:3]

        return convert_open3d_to_unity_point(point_world_o3d)

    def fill_holes(self, aligned_depth=None, kernel_size=5):
        """スパースなDepthマップの穴埋め"""
        if aligned_depth is None:
            aligned_depth = self._aligned_depth
        if aligned_depth is None:
            raise RuntimeError("align() を先に実行してください")

        valid_mask = (aligned_depth > 0).astype(np.float32)
        depth_sum = cv2.blur(aligned_depth, (kernel_size, kernel_size))
        count_sum = cv2.blur(valid_mask, (kernel_size, kernel_size))
        filled = np.divide(depth_sum, count_sum, out=np.zeros_like(depth_sum), where=count_sum > 0).astype(np.float32)
        result = np.where(aligned_depth > 0, aligned_depth, filled)

        self._aligned_depth = result
        return result
