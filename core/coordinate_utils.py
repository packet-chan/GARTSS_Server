"""
座標変換ユーティリティ

Quest 3のCamera2 API, Unity, Open3D間の座標変換を行う。
metaquest-3d-reconstruction リポジトリの座標変換ロジックを完全に再現。
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def convert_camera2_pose(raw_translation, raw_rotation):
    """Camera2 APIのraw姿勢をUnity座標系のローカル変換に変換"""
    transl = list(raw_translation)
    transl[2] *= -1

    qx = -raw_rotation[0]
    qy = -raw_rotation[1]
    qz = raw_rotation[2]
    qw = raw_rotation[3]

    rot = R.from_quat([qx, qy, qz, qw]).inv()
    rot = rot * R.from_euler("x", np.pi)

    return np.array(transl), rot.as_quat()


def apply_local_transform(parent_positions, parent_rotations, local_position, local_rotation):
    """親の姿勢にローカル変換を適用して子の姿勢を得る"""
    parent_rot = R.from_quat(parent_rotations)
    rotated_pos = parent_rot.apply(local_position)
    child_positions = parent_positions + rotated_pos

    local_rot = R.from_quat(local_rotation)
    child_rot = parent_rot * local_rot

    return child_positions, child_rot.as_quat()


def convert_unity_to_open3d_camera(positions, rotations):
    """Unity座標系のカメラ姿勢をOpen3D座標系に変換"""
    R_conv = np.diag([1.0, 1.0, -1.0])
    converted_positions = (R_conv @ positions.T).T

    rotation_matrices = R.from_quat(rotations).as_matrix()
    source_basis = np.eye(3)
    target_basis = np.diag([1.0, -1.0, -1.0])

    rotation_matrices = rotation_matrices @ source_basis.T
    converted_rotations = R_conv @ rotation_matrices @ R_conv.T
    converted_rotations = converted_rotations @ target_basis

    return converted_positions, R.from_matrix(converted_rotations).as_quat()


def convert_open3d_to_unity_point(point_o3d):
    """Open3Dワールド座標 -> Unityワールド座標（Z反転）"""
    result = np.array(point_o3d, dtype=np.float64)
    if result.ndim == 1:
        result[2] *= -1
    else:
        result[:, 2] *= -1
    return result


def compute_extrinsic_wc(positions, rotations):
    """Camera-to-World行列の逆行列（World-to-Camera）を計算"""
    N = len(positions)
    R_cw = R.from_quat(rotations).as_matrix()

    extrinsic_cw = np.zeros((N, 4, 4), dtype=np.float64)
    extrinsic_cw[:, :3, :3] = R_cw
    extrinsic_cw[:, :3, 3] = positions
    extrinsic_cw[:, 3, 3] = 1.0

    return np.linalg.inv(extrinsic_cw)


def ndc_to_linear(depth_buffer, near, far):
    """NDCバッファ (0~1) をリニアDepth [m] に変換"""
    if np.isinf(far) or far < near:
        x_param = -2.0 * near
        y_param = -1.0
    else:
        x_param = -2.0 * far * near / (far - near)
        y_param = -(far + near) / (far - near)

    ndc = depth_buffer * 2.0 - 1.0
    denom = ndc + y_param
    return np.divide(
        x_param, denom, out=np.zeros_like(depth_buffer), where=denom != 0
    ).astype(np.float32)


def compute_depth_intrinsics(width, height, fov_left, fov_right, fov_top, fov_bottom):
    """Depth FoV tangent値から内部パラメータを計算"""
    fx = width / (fov_right + fov_left)
    fy = height / (fov_top + fov_bottom)
    cx = width * fov_right / (fov_right + fov_left)
    cy = height * fov_top / (fov_top + fov_bottom)
    cx_o3d = width - cx

    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "cx_o3d": cx_o3d}


def compute_rgb_intrinsics_o3d(fx, fy, cx, cy, image_width):
    """RGB内部パラメータのOpen3D用変換"""
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "cx_o3d": image_width - cx}
