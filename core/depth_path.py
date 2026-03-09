"""
Depth Map に沿ったガイドパス生成

2D 画像上で方向をサンプリングし、各ピクセルの Depth を使って
マシンの凹凸に沿った 3D パス（ガイドライン）を生成する。
"""

import numpy as np
from typing import Optional


def compute_guide_path(
    engine,
    start_2d: tuple[float, float],
    direction_2d: tuple[float, float],
    num_samples: int = 30,
    step_px: float = 10.0,
) -> list[list[float]]:
    """
    2D 画像上で direction_2d 方向にサンプリングし、
    各点の Depth → 3D 変換で Depth Map に沿った 3D パスを生成する。

    マシンの凹み等がある場合、この 3D パスは自然なカーブになる。

    Args:
        engine: AlignmentEngine (get_3d_point_unity メソッド必須)
        start_2d: (u, v) 開始ピクセル座標
        direction_2d: (du, dv) 正規化された 2D 方向ベクトル
        num_samples: サンプル数
        step_px: サンプリング間隔 (ピクセル)

    Returns:
        3D パス座標リスト [[x, y, z], ...] (Unity ワールド座標)
    """
    path_3d = []
    for i in range(num_samples):
        u = start_2d[0] + direction_2d[0] * step_px * i
        v = start_2d[1] + direction_2d[1] * step_px * i

        # 画像範囲チェック
        if u < 0 or u >= engine.img_w or v < 0 or v >= engine.img_h:
            break

        point = engine.get_3d_point_unity(float(u), float(v))
        if point is not None:
            path_3d.append(point.tolist())

    return path_3d


def compute_pull_guide_path(
    engine,
    centroid_2d: tuple[float, float],
    action_direction_2d: tuple[float, float],
    arrow_length_m: float = 0.15,
    num_samples: int = 20,
) -> list[list[float]]:
    """
    引き出し操作用のガイドパスを生成する。
    centroid から action_direction_2d 方向に伸びる Depth 沿いのパス。

    Args:
        engine: AlignmentEngine
        centroid_2d: (u, v) セグメント重心のピクセル座標
        action_direction_2d: (du, dv) 引き出し方向の 2D ベクトル (正規化済み)
        arrow_length_m: ガイドラインの目標長さ [m]
        num_samples: サンプル数

    Returns:
        3D パス座標リスト (Unity ワールド座標)
    """
    # ステップサイズを推定
    # 大まかに: 1px ≈ depth_m / focal_length なので
    # arrow_length_m に対応するピクセル数を概算
    depth_at_center = engine.get_depth_at_pixel(centroid_2d[0], centroid_2d[1])
    if depth_at_center is None or depth_at_center <= 0:
        depth_at_center = 1.0  # フォールバック

    # 1m あたりのピクセル数 ≈ fx / depth
    # → arrow_length_m のピクセル数 ≈ fx * arrow_length_m / depth
    px_per_meter = engine.r_fx / depth_at_center
    total_px = px_per_meter * arrow_length_m
    step_px = total_px / max(num_samples - 1, 1)

    return compute_guide_path(
        engine=engine,
        start_2d=centroid_2d,
        direction_2d=action_direction_2d,
        num_samples=num_samples,
        step_px=step_px,
    )


def project_3d_direction_to_2d(
    engine,
    centroid_2d: tuple[float, float],
    direction_3d: np.ndarray,
    step_m: float = 0.01,
) -> Optional[tuple[float, float]]:
    """
    3D 方向ベクトルを 2D 画像上の方向に射影する。

    centroid の 3D 位置から direction_3d 方向に少し進んだ点を
    再度 2D に投影し、2D 方向を推定する。

    Args:
        engine: AlignmentEngine
        centroid_2d: (u, v) 重心のピクセル座標
        direction_3d: (3,) 操作方向ベクトル (Unity座標系)
        step_m: 微小移動量 [m]

    Returns:
        (du, dv) 正規化された 2D 方向、または None
    """
    # centroid の 3D 位置
    point_3d = engine.get_3d_point_unity(centroid_2d[0], centroid_2d[1])
    if point_3d is None:
        return None

    # direction_3d 方向に step_m 進んだ点
    target_3d = point_3d + direction_3d * step_m

    # Unity → Open3D 変換 (Z反転)
    target_o3d = target_3d.copy()
    target_o3d[2] *= -1

    # Open3D ワールド → カメラ座標
    if engine._rgb_ext_cw is None:
        return None
    ext_wc = np.linalg.inv(engine._rgb_ext_cw)
    target_h = np.append(target_o3d, 1.0)
    target_cam = (ext_wc @ target_h)[:3]

    if target_cam[2] <= 0.01:
        return None

    # カメラ → ピクセル
    u_target = (target_cam[0] / target_cam[2]) * engine.r_fx + engine.r_cx_o3d
    v_target = (target_cam[1] / target_cam[2]) * engine.r_fy + engine.r_cy

    # 2D 方向
    du = u_target - centroid_2d[0]
    dv = v_target - centroid_2d[1]
    length = np.sqrt(du * du + dv * dv)
    if length < 1e-6:
        return None

    return (du / length, dv / length)
