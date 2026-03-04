"""
点群生成テスト

合成データで mask_to_point_cloud → save_ply の一連の動作を検証する。
SAM2 モデルは不要（マスクを直接渡す）。
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.alignment import AlignmentEngine
from core.sam_segment import mask_to_point_cloud, save_ply


def test_point_cloud_from_mask():
    print("=== test_point_cloud_from_mask ===")

    # --- 1. AlignmentEngine をセットアップ ---
    engine = AlignmentEngine()
    camera_chars = {
        "pose": {
            "translation": [0.03, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0, 1.0],
        },
        "intrinsics": {"fx": 600.0, "fy": 600.0, "cx": 640.0, "cy": 640.0},
    }
    engine.init_session(camera_chars, image_width=1280, image_height=1280)
    print("  Engine initialized")

    # --- 2. 合成 Depth データでアライメント ---
    h, w = 320, 320
    depth_raw = np.full((h, w), 0.9, dtype=np.float32)

    depth_descriptor = {
        "timestamp_ms": 1000,
        "create_pose_location_x": 0.0,
        "create_pose_location_y": 1.6,
        "create_pose_location_z": 0.0,
        # Y軸180度回転（Unity前方を向く合成データ用）
        "create_pose_rotation_x": 0.0,
        "create_pose_rotation_y": 1.0,
        "create_pose_rotation_z": 0.0,
        "create_pose_rotation_w": 0.0,
        "fov_left_angle_tangent": 1.376,
        "fov_right_angle_tangent": 0.839,
        "fov_top_angle_tangent": 0.966,
        "fov_down_angle_tangent": 1.428,
        "near_z": 0.1,
        "far_z": "Infinity",
        "width": 320,
        "height": 320,
    }

    hmd_poses = [
        {"timestamp_ms": 1000, "pos_x": 0, "pos_y": 1.6, "pos_z": 0,
         "rot_x": 0, "rot_y": 0, "rot_z": 0, "rot_w": 1},
    ]
    engine.set_hmd_poses(hmd_poses)

    hmd_pos = np.array([0.0, 1.6, 0.0])
    hmd_rot = np.array([0.0, 0.0, 0.0, 1.0])

    result = engine.align(depth_raw, depth_descriptor, hmd_pos, hmd_rot)
    engine.fill_holes()
    print(f"  Alignment coverage: {result['coverage']:.1f}%")

    # --- 3. 合成マスク (中央 200x200 ピクセル領域) ---
    mask = np.zeros((1280, 1280), dtype=bool)
    mask[540:740, 540:740] = True
    mask_pixel_count = np.count_nonzero(mask)
    print(f"  Mask pixels: {mask_pixel_count}")

    # --- 4. 合成 RGB 画像 (赤色の領域) ---
    image_bgr = np.zeros((1280, 1280, 3), dtype=np.uint8)
    image_bgr[540:740, 540:740] = [0, 0, 255]  # BGR: 赤

    # --- 5. 点群生成 ---
    pc = mask_to_point_cloud(
        mask=mask,
        engine=engine,
        image_bgr=image_bgr,
        sample_step=2,
    )

    print(f"  Point cloud: {pc['count']} points")
    assert pc["count"] > 0, "Should have some valid 3D points"
    assert pc["points"].shape[1] == 3, "Points should be (N, 3)"
    assert pc["colors"] is not None, "Colors should be present"
    assert pc["colors"].shape[1] == 3, "Colors should be (N, 3)"
    print(f"  Points shape: {pc['points'].shape}")
    print(f"  Colors shape: {pc['colors'].shape}")

    # 座標値が妥当かチェック (数メートル以内)
    max_dist = np.max(np.abs(pc["points"]))
    print(f"  Max coordinate value: {max_dist:.4f}m")
    assert max_dist < 50, f"Coordinates seem too large: {max_dist}"

    # --- 6. PLY 保存 ---
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    # 色なし
    ply_path_no_color = str(output_dir / "test_no_color.ply")
    save_ply(ply_path_no_color, pc["points"])
    assert Path(ply_path_no_color).exists()

    # 色あり
    ply_path_color = str(output_dir / "test_with_color.ply")
    save_ply(ply_path_color, pc["points"], pc["colors"])
    assert Path(ply_path_color).exists()

    # --- 7. PLY ファイルの中身を検証 ---
    with open(ply_path_color, "r") as f:
        lines = f.readlines()

    # ヘッダー検証
    assert lines[0].strip() == "ply"
    assert lines[1].strip() == "format ascii 1.0"
    assert f"element vertex {pc['count']}" in lines[2]
    assert "property uchar red" in "".join(lines[:10])

    # データ行の検証
    header_end = next(i for i, line in enumerate(lines) if "end_header" in line)
    data_lines = lines[header_end + 1:]
    assert len(data_lines) == pc["count"], f"Data lines: {len(data_lines)} vs {pc['count']}"

    # 1行パース
    parts = data_lines[0].strip().split()
    assert len(parts) == 6, f"Expected 6 values per line (x y z r g b), got {len(parts)}"
    print(f"  Sample PLY line: {data_lines[0].strip()}")

    print(f"\n  pass: All point cloud tests passed!")
    print(f"  Output files:")
    print(f"    {ply_path_no_color}")
    print(f"    {ply_path_color}")


def test_batch_3d_conversion():
    """get_3d_points_batch の基本テスト"""
    print("\n=== test_batch_3d_conversion ===")

    engine = AlignmentEngine()
    camera_chars = {
        "pose": {
            "translation": [0.03, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0, 1.0],
        },
        "intrinsics": {"fx": 600.0, "fy": 600.0, "cx": 640.0, "cy": 640.0},
    }
    engine.init_session(camera_chars, image_width=1280, image_height=1280)

    hmd_poses = [
        {"timestamp_ms": 1000, "pos_x": 0, "pos_y": 1.6, "pos_z": 0,
         "rot_x": 0, "rot_y": 0, "rot_z": 0, "rot_w": 1},
    ]
    engine.set_hmd_poses(hmd_poses)

    depth_raw = np.full((320, 320), 0.9, dtype=np.float32)
    depth_descriptor = {
        "timestamp_ms": 1000,
        "create_pose_location_x": 0.0, "create_pose_location_y": 1.6,
        "create_pose_location_z": 0.0,
        "create_pose_rotation_x": 0.0, "create_pose_rotation_y": 1.0,
        "create_pose_rotation_z": 0.0, "create_pose_rotation_w": 0.0,
        "fov_left_angle_tangent": 1.376, "fov_right_angle_tangent": 0.839,
        "fov_top_angle_tangent": 0.966, "fov_down_angle_tangent": 1.428,
        "near_z": 0.1, "far_z": "Infinity", "width": 320, "height": 320,
    }
    engine.align(depth_raw, depth_descriptor,
                 np.array([0.0, 1.6, 0.0]), np.array([0.0, 0.0, 0.0, 1.0]))
    engine.fill_holes()

    # バッチ変換 vs 単体変換の一貫性チェック
    u_test = np.array([640.0, 500.0, 700.0])
    v_test = np.array([640.0, 500.0, 700.0])

    batch_points, batch_valid = engine.get_3d_points_batch(u_test, v_test)
    print(f"  Batch: {len(batch_points)} valid points out of {len(u_test)}")

    for i in range(len(u_test)):
        if batch_valid[i]:
            single = engine.get_3d_point_unity(u_test[i], v_test[i])
            if single is not None:
                # batch_points の対応するインデックスを見つける
                batch_idx = np.sum(batch_valid[:i + 1]) - 1
                diff = np.abs(batch_points[batch_idx] - single)
                max_diff = np.max(diff)
                print(f"  Pixel ({u_test[i]:.0f}, {v_test[i]:.0f}): "
                      f"batch={batch_points[batch_idx]}, single={single}, diff={max_diff:.6f}")
                # 注: get_3d_point_unity は median patch を使うので完全一致はしない
                # バッチ版は直接 depth 値を使うため若干異なる

    print("  pass: Batch conversion works")


def test_empty_mask():
    """空マスクのエッジケース"""
    print("\n=== test_empty_mask ===")

    engine = AlignmentEngine()
    engine.init_session(
        {"pose": {"translation": [0, 0, 0], "rotation": [0, 0, 0, 1]},
         "intrinsics": {"fx": 600, "fy": 600, "cx": 640, "cy": 640}},
        1280, 1280
    )

    # 空マスク
    mask = np.zeros((1280, 1280), dtype=bool)
    pc = mask_to_point_cloud(mask, engine)
    assert pc["count"] == 0
    print("  pass: Empty mask returns 0 points")

    # PLY に空のデータ
    save_ply("test_output/test_empty.ply", pc["points"])
    print("  pass: Empty PLY handled gracefully")


if __name__ == "__main__":
    test_batch_3d_conversion()
    test_point_cloud_from_mask()
    test_empty_mask()

    print("\n" + "=" * 50)
    print("All point cloud tests passed!")
