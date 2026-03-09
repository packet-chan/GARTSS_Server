"""
Surface Normal 推定モジュール

点群データから PCA (主成分分析) を使ってサーフェスの法線方向を推定し、
タスクに応じた操作方向（引き出し・回転・押下）を計算する。
"""

import numpy as np
from typing import Optional


def compute_surface_normal(
    points: np.ndarray,
    camera_position: Optional[np.ndarray] = None,
) -> dict:
    """
    点群から PCA でサーフェス法線を推定する。

    Args:
        points: (N, 3) float64 — Unity ワールド座標系の点群
        camera_position: (3,) カメラ位置 (Unity座標系)。
                         Normal の向きをカメラ側に統一するために使用。
                         None の場合は原点を仮定。

    Returns:
        {
            "centroid": np.ndarray (3,),        — 点群の重心
            "normal": np.ndarray (3,),          — 単位法線ベクトル (カメラ側向き)
            "principal_axes": np.ndarray (3, 3), — 主成分軸 [PC1, PC2, PC3] (行ベクトル)
            "eigenvalues": np.ndarray (3,),     — 固有値 (昇順)
            "planarity": float,                 — 面の平面度 (0~1, 1に近いほど平面)
            "valid": bool,                      — 推定が信頼できるか
        }
    """
    if points is None or len(points) < 10:
        return {
            "centroid": np.zeros(3),
            "normal": np.array([0.0, 1.0, 0.0]),  # デフォルト: 上向き
            "principal_axes": np.eye(3),
            "eigenvalues": np.zeros(3),
            "planarity": 0.0,
            "valid": False,
        }

    # 1. 重心
    centroid = points.mean(axis=0)

    # 2. 共分散行列 → 固有値分解
    centered = points - centroid
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh は固有値を昇順で返す
    # eigenvectors[:, 0] = 最小固有値方向 = 法線

    normal = eigenvectors[:, 0].copy()
    pc_medium = eigenvectors[:, 1].copy()
    pc_major = eigenvectors[:, 2].copy()

    # 3. Normal の向きをカメラ側に統一
    if camera_position is None:
        camera_position = np.zeros(3)
    to_camera = camera_position - centroid
    if np.linalg.norm(to_camera) > 1e-6:
        if np.dot(normal, to_camera) < 0:
            normal = -normal

    # 4. 平面度 (planarity) の計算
    #    完全な平面なら最小固有値 ≈ 0 → planarity ≈ 1
    total_var = eigenvalues.sum()
    if total_var > 0:
        planarity = 1.0 - (eigenvalues[0] / total_var)
    else:
        planarity = 0.0

    # 5. 信頼性判定
    #    - 点が十分にある (>= 50)
    #    - 面がある程度平面 (planarity >= 0.8)
    valid = len(points) >= 50 and planarity >= 0.8

    return {
        "centroid": centroid,
        "normal": normal / np.linalg.norm(normal),  # 念のため正規化
        "principal_axes": np.stack([pc_major, pc_medium, normal]),
        "eigenvalues": eigenvalues,
        "planarity": planarity,
        "valid": valid,
    }


def compute_action_direction(
    normal: np.ndarray,
    action_type: str,
    camera_position: Optional[np.ndarray] = None,
    centroid: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Surface Normal とアクションタイプから操作方向ベクトルを計算する。

    Args:
        normal: (3,) 単位法線ベクトル (Unity座標系、カメラ側向き)
        action_type: "pull" | "push" | "press" | "rotate_cw" | "rotate_ccw"
        camera_position: (3,) カメラ位置
        centroid: (3,) 面の重心

    Returns:
        (3,) 操作方向の単位ベクトル (Unity座標系)
    """
    if camera_position is None:
        camera_position = np.zeros(3)
    if centroid is None:
        centroid = np.zeros(3)

    if action_type == "pull":
        # 引き出す = カメラ方向を水平面に射影
        # トレーやタンクは水平に引き出すものなので、Y成分を0にして水平に限定
        to_camera = camera_position - centroid
        to_camera[1] = 0  # Y成分を除去 → 水平方向のみ
        norm = np.linalg.norm(to_camera)
        if norm < 1e-6:
            return np.array([0.0, 0.0, 1.0])
        return to_camera / norm

    elif action_type == "push" or action_type == "press":
        # 押す = Normal の逆方向（面に向かって押し込む）
        return -normal / max(np.linalg.norm(normal), 1e-8)

    elif action_type == "rotate_cw":
        # 時計回り = Normal 方向そのまま（右ねじの法則: Normal方向から見てCW）
        return normal / max(np.linalg.norm(normal), 1e-8)

    elif action_type == "rotate_ccw":
        # 反時計回り = Normal の逆
        return -normal / max(np.linalg.norm(normal), 1e-8)

    else:
        # フォールバック: Normal 方向
        return normal / max(np.linalg.norm(normal), 1e-8)


def compute_arrow_start(
    points: np.ndarray,
    centroid: np.ndarray,
    action_direction: np.ndarray,
    normal: np.ndarray,
    offset_m: float = 0.03,
    percentile: float = 95,
) -> np.ndarray:
    """
    centroid を基準に、action_direction 方向にだけ前端まで移動した矢印開始位置を返す。

    左右方向は centroid のまま（＝オブジェクトの真ん中）を維持し、
    前後方向だけ前端に寄せることで、「ど真ん中から手前に飛び出す矢印」になる。

    Args:
        points: (N, 3) 点群
        centroid: (3,) 重心
        action_direction: (3,) 操作方向の単位ベクトル
        normal: (3,) 法線ベクトル
        offset_m: 面からの浮き距離 [m]
        percentile: 前方何%の点を使うか (95 = 上位5%の平均)

    Returns:
        (3,) 矢印の開始位置 (Unity座標系)
    """
    if points is None or len(points) < 10:
        return centroid + normal * offset_m

    # 各点を action_direction に射影して、前方スコアを計算
    centered = points - centroid
    projections = centered @ action_direction

    # 上位 percentile の前方距離を取得
    threshold = np.percentile(projections, percentile)

    # centroid から action_direction 方向に threshold 分だけ前進
    # (左右方向は centroid のまま！)
    arrow_start = centroid + action_direction * threshold + normal * offset_m

    return arrow_start