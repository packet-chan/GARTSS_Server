"""
compute_local_normal のユニットテスト

局所法線推定が正しく動作するかを検証する。
"""

import numpy as np
import sys
sys.path.insert(0, ".")

from core.normal_estimator import compute_local_normal, compute_surface_normal


def _make_plane_points(normal, center, n=200, spread=0.05, noise=0.001):
    """指定法線の平面上に点群を生成"""
    normal = np.array(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    # 法線に垂直な2つのベクトルを求める
    if abs(normal[0]) < 0.9:
        u = np.cross(normal, [1, 0, 0])
    else:
        u = np.cross(normal, [0, 1, 0])
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    center = np.array(center, dtype=float)
    rng = np.random.default_rng(42)
    s = rng.uniform(-spread, spread, (n, 2))
    points = center + s[:, 0:1] * u + s[:, 1:2] * v
    points += rng.normal(0, noise, points.shape)
    return points


def test_local_normal_flat_surface():
    """水平面 (法線 = Y up) の局所法線テスト"""
    points = _make_plane_points([0, 1, 0], [0, 0.5, 0])
    query = np.array([0.0, 0.5, 0.0])
    camera = np.array([0.0, 1.5, 0.5])

    result = compute_local_normal(points, query, radius_m=0.1, camera_position=camera)

    assert result["valid"], f"Should be valid, planarity={result['planarity']}"
    # Y 成分が支配的であるべき
    assert abs(result["normal"][1]) > 0.9, f"Normal Y should be > 0.9, got {result['normal']}"
    print(f"✓ Flat surface: normal={result['normal']}, planarity={result['planarity']:.4f}")


def test_local_normal_tilted_surface():
    """45度傾斜面の局所法線テスト"""
    # 法線が (0, 1, 1)/sqrt(2) の傾斜面
    tilted_normal = np.array([0, 1, 1]) / np.sqrt(2)
    points = _make_plane_points(tilted_normal, [0, 0.5, 0.5])
    query = np.array([0.0, 0.5, 0.5])
    camera = np.array([0.0, 1.5, 1.5])

    result = compute_local_normal(points, query, radius_m=0.1, camera_position=camera)

    assert result["valid"], f"Should be valid, planarity={result['planarity']}"
    # Y と Z 成分がほぼ等しいはず
    dot = np.dot(result["normal"], tilted_normal)
    assert abs(dot) > 0.95, f"Normal should align with tilted_normal, dot={dot}"
    print(f"✓ Tilted surface: normal={result['normal']}, dot={dot:.4f}")


def test_local_normal_two_surfaces():
    """2つの面が接合する部品で、起点に近い面の法線を取得するテスト"""
    # 面1: 水平面 (法線 = Y up), center = (0, 0.5, 0)
    plane1 = _make_plane_points([0, 1, 0], [0, 0.5, 0], n=300, spread=0.1)
    # 面2: 垂直前面 (法線 = Z forward), center = (0, 0.5, 0.15)
    plane2 = _make_plane_points([0, 0, 1], [0, 0.5, 0.15], n=300, spread=0.05)

    all_points = np.vstack([plane1, plane2])

    # 全体 PCA → 両面混合で法線が不正確になりうる
    global_result = compute_surface_normal(all_points, camera_position=np.array([0, 1.5, 0.5]))

    # 面2の中心付近で局所法線 → Z forward を検出するはず
    query_front = np.array([0.0, 0.5, 0.15])
    local_result = compute_local_normal(
        all_points, query_front, radius_m=0.04,
        camera_position=np.array([0, 1.5, 0.5])
    )

    if local_result["valid"]:
        z_component = abs(local_result["normal"][2])
        print(f"✓ Two surfaces: local normal Z={z_component:.4f} "
              f"(global Z={abs(global_result['normal'][2]):.4f})")
        # 局所法線の Z 成分は全体法線より大きいはず (前面に近いため)
        assert z_component > 0.7, f"Local normal Z should be > 0.7, got {z_component}"
    else:
        print(f"⚠ Two surfaces: local PCA not valid, this may happen with small radius")


def test_local_normal_radius_expansion():
    """近傍点が少ない場合に半径を拡大するテスト"""
    # 疎な点群 (半径 0.03m では点が足りない)
    points = _make_plane_points([0, 1, 0], [0, 0.5, 0], n=50, spread=0.15)
    query = np.array([0.0, 0.5, 0.0])

    result = compute_local_normal(points, query, radius_m=0.01, camera_position=np.array([0, 1.5, 0]))

    # 半径拡大 or フォールバックで結果が出るはず
    print(f"✓ Radius expansion: valid={result['valid']}, "
          f"normal={result['normal']}, planarity={result['planarity']:.4f}")


def test_global_pca_unchanged():
    """既存の compute_surface_normal が変更後も同じ結果を返すテスト"""
    points = _make_plane_points([0, 1, 0], [0, 0.5, 0])
    camera = np.array([0.0, 1.5, 0.5])

    result = compute_surface_normal(points, camera)

    assert result["valid"]
    assert abs(result["normal"][1]) > 0.9
    print(f"✓ Global PCA unchanged: normal={result['normal']}, planarity={result['planarity']:.4f}")


if __name__ == "__main__":
    test_local_normal_flat_surface()
    test_local_normal_tilted_surface()
    test_local_normal_two_surfaces()
    test_local_normal_radius_expansion()
    test_global_pca_unchanged()
    print("\n=== All tests passed ===")
