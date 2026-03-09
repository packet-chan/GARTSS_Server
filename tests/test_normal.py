"""
テスト: Surface Normal 推定

実際の drip_tray 点群データを使って法線推定の正しさを検証する。

使い方:
  cd GARTSS_Server
  python -m pytest tests/test_normal.py -v
  
  または直接実行:
  python tests/test_normal.py
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from core.normal_estimator import compute_surface_normal, compute_action_direction


def load_ply(ply_path: str) -> tuple[np.ndarray, np.ndarray]:
    """PLY ファイルから点群と色を読み込む"""
    points = []
    colors = []
    header_done = False
    with open(ply_path, "r") as f:
        for line in f:
            line = line.strip()
            if not header_done:
                if line == "end_header":
                    header_done = True
                continue
            parts = line.split()
            if len(parts) >= 3:
                points.append([float(parts[0]), float(parts[1]), float(parts[2])])
                if len(parts) >= 6:
                    colors.append([int(parts[3]), int(parts[4]), int(parts[5])])
    return np.array(points), np.array(colors) if colors else None


# =========================================================================
#  PLY パスの設定
# =========================================================================
# GitHub Actions / CI では PLY が無い場合はスキップ
PLY_PATH = Path(__file__).parent.parent / "captures" / "14116286" / "pointcloud_drip_tray.ply"
# ローカルテスト用: アップロードされたファイル
ALT_PLY_PATH = Path("/mnt/user-data/uploads/pointcloud_drip_tray.ply")

def get_ply_path():
    if PLY_PATH.exists():
        return PLY_PATH
    if ALT_PLY_PATH.exists():
        return ALT_PLY_PATH
    return None


# =========================================================================
#  テストケース
# =========================================================================

def test_basic_normal_estimation():
    """基本的な平面の法線推定テスト (合成データ)"""
    # XZ 平面上の点群 (Y = 0, ノイズ少々)
    rng = np.random.default_rng(42)
    n = 1000
    points = np.zeros((n, 3))
    points[:, 0] = rng.uniform(-1, 1, n)   # X
    points[:, 1] = rng.normal(0, 0.01, n)  # Y (ほぼ0)
    points[:, 2] = rng.uniform(-1, 1, n)   # Z

    result = compute_surface_normal(points, camera_position=np.array([0, 5, 0]))

    assert result["valid"]
    normal = result["normal"]
    # Normal は Y 方向に近いはず
    assert abs(abs(normal[1]) - 1.0) < 0.05, f"Normal should be ~Y axis, got {normal}"
    # カメラは Y+ にあるので Normal も Y+ を向くはず
    assert normal[1] > 0, f"Normal should face camera (Y+), got {normal}"
    assert result["planarity"] > 0.95


def test_tilted_plane():
    """傾いた平面のテスト"""
    rng = np.random.default_rng(123)
    n = 500
    # Y = 0.5*X の平面
    x = rng.uniform(-1, 1, n)
    z = rng.uniform(-1, 1, n)
    y = 0.5 * x + rng.normal(0, 0.005, n)
    points = np.stack([x, y, z], axis=1)

    result = compute_surface_normal(points)
    assert result["valid"]
    # 法線は [-0.5, 1, 0] を正規化した方向に近いはず
    expected = np.array([-0.5, 1.0, 0.0])
    expected /= np.linalg.norm(expected)
    dot = abs(np.dot(result["normal"], expected))
    assert dot > 0.95, f"Normal {result['normal']} not aligned with expected {expected}, dot={dot}"


def test_insufficient_points():
    """点が少なすぎる場合"""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    result = compute_surface_normal(points)
    assert not result["valid"]


def test_none_input():
    """None 入力"""
    result = compute_surface_normal(None)
    assert not result["valid"]


def test_pull_direction():
    """Pull 方向の計算テスト"""
    normal = np.array([0.0, 1.0, 0.0])  # 上向き
    camera_pos = np.array([0.0, 2.0, 3.0])  # 斜め上前方
    centroid = np.array([0.0, 0.0, 0.0])

    direction = compute_action_direction(normal, "pull", camera_pos, centroid)
    # 面に射影されるので Y 成分 ≈ 0
    assert abs(direction[1]) < 0.01, f"Pull Y should be ~0, got {direction[1]}"
    # Z+ 方向 (カメラに向かう)
    assert direction[2] > 0.5, f"Pull should be toward camera (Z+), got {direction}"


def test_press_direction():
    """Press 方向 = Normal の逆"""
    normal = np.array([0.0, 1.0, 0.0])
    direction = compute_action_direction(normal, "press")
    np.testing.assert_array_almost_equal(direction, [0.0, -1.0, 0.0])


def test_drip_tray_ply():
    """実際の drip_tray PLY データでのテスト"""
    ply_path = get_ply_path()
    if ply_path is None:
        print("SKIP: PLY file not found")
        return

    points, colors = load_ply(str(ply_path))
    print(f"Loaded {len(points)} points from {ply_path}")

    # カメラはだいたい原点付近 (Quest 3 HMD)
    camera_pos = np.array([0.0, 0.0, 0.0])

    result = compute_surface_normal(points, camera_position=camera_pos)

    print(f"  Centroid: {result['centroid']}")
    print(f"  Normal:   {result['normal']}")
    print(f"  Planarity: {result['planarity']:.4f}")
    print(f"  Valid: {result['valid']}")

    assert result["valid"], "drip_tray should be a valid planar surface"
    assert result["planarity"] > 0.95, f"drip_tray should be very flat, got {result['planarity']}"

    # Drip Tray は水平面なので Normal の Y 成分が支配的
    normal = result["normal"]
    assert abs(normal[1]) > 0.8, f"Normal Y should be dominant for horizontal tray, got {normal}"

    # Pull 方向を計算
    action_dir = compute_action_direction(
        normal=result["normal"],
        action_type="pull",
        camera_position=camera_pos,
        centroid=result["centroid"],
    )
    print(f"  Pull direction: {action_dir}")

    # 引き出し方向は Z+ (手前) が支配的、Y ≈ 0 (水平)
    assert action_dir[2] > 0.5, f"Pull Z should be positive (toward camera), got {action_dir[2]}"
    assert abs(action_dir[1]) < 0.1, f"Pull Y should be ~0 (horizontal), got {action_dir[1]}"

    print("  ✓ All drip_tray assertions passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("Surface Normal Estimation Tests")
    print("=" * 60)

    tests = [
        ("Basic normal estimation", test_basic_normal_estimation),
        ("Tilted plane", test_tilted_plane),
        ("Insufficient points", test_insufficient_points),
        ("None input", test_none_input),
        ("Pull direction", test_pull_direction),
        ("Press direction", test_press_direction),
        ("Drip tray PLY", test_drip_tray_ply),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            print(f"\n--- {name} ---")
            test_fn()
            print(f"  ✓ PASSED")
            passed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
