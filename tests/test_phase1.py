"""
Phase 1 テスト

コアモジュールとAPIエンドポイントの動作確認。
既存のQuestDataがなくても合成データで検証できる。
"""

import json
import io
import sys
import numpy as np
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.coordinate_utils import (
    convert_camera2_pose,
    apply_local_transform,
    convert_unity_to_open3d_camera,
    convert_open3d_to_unity_point,
    ndc_to_linear,
    compute_depth_intrinsics,
)
from core.pose_interpolator import PoseInterpolator
from core.alignment import AlignmentEngine


def test_coordinate_utils():
    print("=== test_coordinate_utils ===")

    raw_t = [0.01, 0.02, 0.03]
    raw_r = [0.0, 0.0, 0.0, 1.0]
    t, r = convert_camera2_pose(raw_t, raw_r)
    assert t[2] == -0.03, f"Z should be flipped: {t}"
    assert len(r) == 4
    print("  pass: convert_camera2_pose")

    parent_pos = np.array([[0, 1, 0]])
    parent_rot = np.array([[0, 0, 0, 1]])
    local_pos = np.array([1, 0, 0])
    local_rot = np.array([0, 0, 0, 1])
    child_pos, child_rot = apply_local_transform(
        parent_pos, parent_rot, local_pos, local_rot
    )
    np.testing.assert_allclose(child_pos[0], [1, 1, 0], atol=1e-10)
    print("  pass: apply_local_transform")

    pos = np.array([[0, 1, 2]])
    rot = np.array([[0, 0, 0, 1]])
    pos_o3d, rot_o3d = convert_unity_to_open3d_camera(pos, rot)
    assert pos_o3d[0, 2] == -2.0, f"Z should be flipped: {pos_o3d}"
    print("  pass: convert_unity_to_open3d_camera")

    p = np.array([1, 2, 3])
    p_unity = convert_open3d_to_unity_point(p)
    assert p_unity[2] == -3.0
    print("  pass: convert_open3d_to_unity_point")

    ndc_buf = np.array([[0.5, 0.9, 0.1]], dtype=np.float32)
    linear = ndc_to_linear(ndc_buf, near=0.1, far=float("inf"))
    assert linear.shape == ndc_buf.shape
    assert np.all(linear >= 0)
    print(f"  pass: ndc_to_linear: {linear}")

    intr = compute_depth_intrinsics(
        320, 320,
        fov_left=1.376, fov_right=0.839,
        fov_top=0.966, fov_bottom=1.428,
    )
    assert "fx" in intr and "cx_o3d" in intr
    print(f"  pass: compute_depth_intrinsics: fx={intr['fx']:.1f}, cx_o3d={intr['cx_o3d']:.1f}")


def test_pose_interpolator():
    print("\n=== test_pose_interpolator ===")

    poses = [
        {"timestamp_ms": 1000, "pos_x": 0, "pos_y": 1, "pos_z": 0,
         "rot_x": 0, "rot_y": 0, "rot_z": 0, "rot_w": 1},
        {"timestamp_ms": 2000, "pos_x": 1, "pos_y": 1, "pos_z": 0,
         "rot_x": 0, "rot_y": 0, "rot_z": 0, "rot_w": 1},
        {"timestamp_ms": 3000, "pos_x": 2, "pos_y": 1, "pos_z": 0,
         "rot_x": 0, "rot_y": 0, "rot_z": 0, "rot_w": 1},
    ]

    interp = PoseInterpolator(poses)
    assert len(interp) == 3
    print("  pass: load_poses")

    pos, rot = interp.interpolate_pose(1500)
    np.testing.assert_allclose(pos, [0.5, 1, 0], atol=1e-10)
    print(f"  pass: interpolate(1500): pos={pos}")

    pos, rot = interp.interpolate_pose(1000)
    np.testing.assert_allclose(pos, [0, 1, 0], atol=1e-10)
    print(f"  pass: interpolate(1000): pos={pos}")

    pos, rot = interp.interpolate_pose(500)
    np.testing.assert_allclose(pos, [0, 1, 0], atol=1e-10)
    print(f"  pass: interpolate(500) [clamp]: pos={pos}")


def test_alignment_engine():
    print("\n=== test_alignment_engine ===")

    engine = AlignmentEngine()

    camera_chars = {
        "pose": {
            "translation": [0.03, 0.0, 0.0],
            "rotation": [0.0, 0.0, 0.0, 1.0],
        },
        "intrinsics": {"fx": 600.0, "fy": 600.0, "cx": 640.0, "cy": 640.0},
    }
    engine.init_session(camera_chars, image_width=1280, image_height=1280)
    assert engine._initialized
    print("  pass: init_session")

    hmd_poses = [
        {"timestamp_ms": 1000, "pos_x": 0, "pos_y": 1.6, "pos_z": 0,
         "rot_x": 0, "rot_y": 0, "rot_z": 0, "rot_w": 1},
    ]
    engine.set_hmd_poses(hmd_poses)
    print("  pass: set_hmd_poses")

    h, w = 320, 320
    depth_raw = np.full((h, w), 0.9, dtype=np.float32)

    depth_descriptor = {
        "timestamp_ms": 1000,
        "create_pose_location_x": 0.0,
        "create_pose_location_y": 1.6,
        "create_pose_location_z": 0.0,
        "create_pose_rotation_x": 0.0,
        "create_pose_rotation_y": 0.0,
        "create_pose_rotation_z": 0.0,
        "create_pose_rotation_w": 1.0,
        "fov_left_angle_tangent": 1.376,
        "fov_right_angle_tangent": 0.839,
        "fov_top_angle_tangent": 0.966,
        "fov_down_angle_tangent": 1.428,
        "near_z": 0.1,
        "far_z": "Infinity",
        "width": 320,
        "height": 320,
    }

    hmd_pos = np.array([0.0, 1.6, 0.0])
    hmd_rot = np.array([0.0, 0.0, 0.0, 1.0])

    result = engine.align(depth_raw, depth_descriptor, hmd_pos, hmd_rot)
    print(f"  pass: align: coverage={result['coverage']:.1f}%")

    filled = engine.fill_holes()
    coverage_filled = np.count_nonzero(filled) / (1280 * 1280) * 100
    print(f"  pass: fill_holes: coverage={coverage_filled:.1f}%")

    d = engine.get_depth_at_pixel(640, 640)
    print(f"  pass: get_depth_at_pixel(640,640): {d}")

    point = engine.get_3d_point_unity(640, 640)
    if point is not None:
        print(f"  pass: get_3d_point_unity(640,640): [{point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f}]")
    else:
        print("  pass: get_3d_point_unity(640,640): None")


async def test_api_endpoints():
    print("\n=== test_api_endpoints ===")
    from httpx import AsyncClient, ASGITransport
    from server import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        print(f"  pass: GET /health: {resp.json()}")

        resp = await client.post("/session/init", json={
            "camera_characteristics": {
                "pose": {
                    "translation": [0.03, 0.0, 0.0],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                },
                "intrinsics": {"fx": 600.0, "fy": 600.0, "cx": 640.0, "cy": 640.0},
            },
            "image_format": {"width": 1280, "height": 1280},
        })
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]
        print(f"  pass: POST /session/init: session_id={session_id}")

        h, w = 320, 320
        depth_raw = np.full((h, w), 0.9, dtype=np.float32)
        depth_bytes = depth_raw.tobytes()

        depth_descriptor = {
            "timestamp_ms": 1000,
            "create_pose_location_x": 0.0,
            "create_pose_location_y": 1.6,
            "create_pose_location_z": 0.0,
            "create_pose_rotation_x": 0.0,
            "create_pose_rotation_y": 0.0,
            "create_pose_rotation_z": 0.0,
            "create_pose_rotation_w": 1.0,
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

        resp = await client.post(
            f"/session/{session_id}/capture",
            data={
                "depth_descriptor": json.dumps(depth_descriptor),
                "hmd_poses": json.dumps(hmd_poses),
            },
            files={
                "depth_raw": ("depth.raw", io.BytesIO(depth_bytes), "application/octet-stream"),
            },
        )
        assert resp.status_code == 200
        print(f"  pass: POST /capture: {resp.json()}")

        resp = await client.get(
            f"/session/{session_id}/depth", params={"u": 640, "v": 640}
        )
        assert resp.status_code == 200
        print(f"  pass: GET /depth: {resp.json()}")

        resp = await client.get(f"/session/{session_id}/info")
        assert resp.status_code == 200
        print(f"  pass: GET /info: {resp.json()}")

        resp = await client.post(
            f"/session/{session_id}/analyze",
            json={"task": "test"},
        )
        assert resp.status_code == 200
        print(f"  pass: POST /analyze (stub): {resp.json()}")

        resp = await client.delete(f"/session/{session_id}")
        assert resp.status_code == 200
        print(f"  pass: DELETE /session")


if __name__ == "__main__":
    import asyncio

    test_coordinate_utils()
    test_pose_interpolator()
    test_alignment_engine()

    print("\n--- Running API tests ---")
    asyncio.run(test_api_endpoints())

    print("\n" + "=" * 50)
    print("All Phase 1 tests passed!")
