"""
点群 → テクスチャ付きメッシュ生成モジュール

SAM マスクから得た点群を Poisson Surface Reconstruction でメッシュ化し、
RGB 画像からテクスチャ投影によりリアルな3Dモデルを生成する。

v2: UV 座標計算 + テクスチャ crop + base64 エンコードを追加。
    頂点カラーはフォールバックとして維持。
"""

import base64
import io
import numpy as np
from typing import Optional

import cv2

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False


def pointcloud_to_mesh(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    target_triangles: int = 5000,
    poisson_depth: int = 8,
    density_quantile: float = 0.15,
    normal_radius: float = 0.02,
    normal_max_nn: int = 30,
) -> Optional[dict]:
    """
    色付き点群からトライアングルメッシュを生成する。

    パイプライン:
      1. Open3D で法線推定
      2. Poisson Surface Reconstruction
      3. 低密度頂点の除去 (穴や外れ面を削減)
      4. 元の点群から頂点カラーを KNN 転送
      5. Quadric Decimation で軽量化

    Returns:
        {
            "vertex_count": int,
            "triangle_count": int,
            "vertices": list[float],
            "triangles": list[int],
            "normals": list[float],
            "colors": list[float],     # (0-1) fallback vertex colors
        }
        or None if failed
    """
    if not HAS_OPEN3D:
        print("[Mesh] Open3D not available, skipping mesh generation")
        return None

    if points is None or len(points) < 100:
        print(f"[Mesh] Not enough points ({len(points) if points is not None else 0})")
        return None

    print(f"[Mesh] Input: {len(points)} points")

    # 色の正規化 (0-1)
    colors_normalized = None
    if colors is not None and len(colors) == len(points):
        colors_f = colors.astype(np.float64)
        if colors_f.max() > 1.0:
            colors_f = colors_f / 255.0
        colors_normalized = colors_f

    # Open3D PointCloud 構築
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors_normalized is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors_normalized)

    # 法線推定
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=normal_max_nn
        )
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)
    print(f"[Mesh] Normals estimated")

    # Poisson Surface Reconstruction
    try:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=poisson_depth
        )
    except Exception as e:
        print(f"[Mesh] Poisson reconstruction failed: {e}")
        return None

    print(f"[Mesh] Poisson: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

    # 低密度部分を除去
    densities_np = np.asarray(densities)
    threshold = np.quantile(densities_np, density_quantile)
    mesh.remove_vertices_by_mask(densities_np < threshold)
    print(f"[Mesh] After density filter: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

    # 色を元の点群から KNN で転送
    if colors_normalized is not None:
        tree = o3d.geometry.KDTreeFlann(pcd)
        mesh_verts = np.asarray(mesh.vertices)
        vertex_colors = np.zeros((len(mesh_verts), 3))
        for i, v in enumerate(mesh_verts):
            [_, idx, _] = tree.search_knn_vector_3d(v, 3)
            vertex_colors[i] = np.mean(colors_normalized[idx], axis=0)
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
        print(f"[Mesh] Colors transferred from point cloud")

    # 簡略化
    if len(mesh.triangles) > target_triangles:
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_triangles
        )
        print(f"[Mesh] Simplified: {len(mesh.vertices)} verts, {len(mesh.triangles)} tris")

    # 法線再計算
    mesh.compute_vertex_normals()

    # フラット配列に変換
    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    normals = np.asarray(mesh.vertex_normals)
    vert_colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else np.ones((len(verts), 3)) * 0.5

    result = {
        "vertex_count": len(verts),
        "triangle_count": len(tris),
        "vertices": verts.flatten().tolist(),
        "triangles": tris.flatten().astype(int).tolist(),
        "normals": normals.flatten().tolist(),
        "colors": vert_colors.flatten().tolist(),
    }

    print(f"[Mesh] Final: {result['vertex_count']} verts, {result['triangle_count']} tris")
    return result


def compute_texture_projection(
    mesh_vertices: np.ndarray,
    engine,
    image_bgr: np.ndarray,
    bbox: list[float],
    sam_mask: np.ndarray = None,
    texture_padding: int = 4,
    png_compress: int = 6,
) -> Optional[dict]:
    """
    メッシュ頂点を RGB カメラに逆投影し、UV 座標とテクスチャ画像を生成する。

    Args:
        mesh_vertices: (N, 3) メッシュ頂点 (Unity ワールド座標系)
        engine: AlignmentEngine
        image_bgr: BGR 画像 (1280x1280)
        bbox: [x_min, y_min, x_max, y_max] SAM BBox ピクセル座標
        sam_mask: (H, W) bool SAM マスク。テクスチャの背景を透明にするために使用。
        texture_padding: テクスチャ crop のパディング [px]
        png_compress: PNG 圧縮レベル (0-9)

    Returns:
        {
            "uvs": list[float],
            "texture_base64": str,         # base64 PNG (RGBA, マスク外透明)
            "texture_width": int,
            "texture_height": int,
            "uv_valid_ratio": float,
        }
        or None if projection failed
    """
    if engine._rgb_ext_cw is None:
        print("[Texture] No camera extrinsics available")
        return None

    if image_bgr is None:
        print("[Texture] No RGB image available")
        return None

    n_verts = len(mesh_vertices)
    if n_verts == 0:
        return None

    # === 1. Unity → Open3D 座標変換 (Z反転) ===
    verts_o3d = mesh_vertices.copy()
    verts_o3d[:, 2] *= -1

    # === 2. World → Camera 変換 ===
    ext_wc = np.linalg.inv(engine._rgb_ext_cw)
    verts_h = np.hstack([verts_o3d, np.ones((n_verts, 1))])
    verts_cam = (ext_wc @ verts_h.T).T[:, :3]

    in_front = verts_cam[:, 2] > 0.01

    # === 3. カメラ → ピクセル座標 ===
    pixel_u = np.zeros(n_verts)
    pixel_v = np.zeros(n_verts)
    pixel_u[in_front] = (verts_cam[in_front, 0] / verts_cam[in_front, 2]) * engine.r_fx + engine.r_cx_o3d
    pixel_v[in_front] = (verts_cam[in_front, 1] / verts_cam[in_front, 2]) * engine.r_fy + engine.r_cy

    in_image = (
        in_front &
        (pixel_u >= 0) & (pixel_u < engine.img_w) &
        (pixel_v >= 0) & (pixel_v < engine.img_h)
    )

    uv_valid_ratio = np.count_nonzero(in_image) / n_verts if n_verts > 0 else 0
    print(f"[Texture] UV valid: {np.count_nonzero(in_image)}/{n_verts} ({uv_valid_ratio*100:.1f}%)")

    # === 4. BBox 領域でクロップ ===
    x_min_b, y_min_b, x_max_b, y_max_b = bbox
    h_img, w_img = image_bgr.shape[:2]
    crop_x1 = max(0, int(x_min_b) - texture_padding)
    crop_y1 = max(0, int(y_min_b) - texture_padding)
    crop_x2 = min(w_img, int(x_max_b) + texture_padding)
    crop_y2 = min(h_img, int(y_max_b) + texture_padding)

    texture_crop = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    tex_h, tex_w = texture_crop.shape[:2]

    if tex_w < 2 or tex_h < 2:
        print(f"[Texture] Crop too small: {tex_w}x{tex_h}")
        return None

    print(f"[Texture] Crop: [{crop_x1},{crop_y1},{crop_x2},{crop_y2}] → {tex_w}x{tex_h}")

    # === 5. SAM マスクを適用してテクスチャのマスク外を透明に ===
    # BGR → BGRA (アルファチャンネル追加)
    texture_rgba = cv2.cvtColor(texture_crop, cv2.COLOR_BGR2BGRA)

    if sam_mask is not None:
        # SAM マスクを crop 領域に合わせて切り出し
        mask_crop = sam_mask[crop_y1:crop_y2, crop_x1:crop_x2].astype(bool)
        # マスク外を透明に (alpha=0)
        texture_rgba[:, :, 3] = np.where(mask_crop, 255, 0).astype(np.uint8)
        n_transparent = int(np.count_nonzero(~mask_crop))
        print(f"[Texture] SAM mask applied: {n_transparent} transparent pixels "
              f"({n_transparent / (tex_w * tex_h) * 100:.1f}%)")
    else:
        texture_rgba[:, :, 3] = 255  # マスクなし → 全体不透明

    # === 6. ピクセル座標 → UV (0-1) ===
    uvs = np.zeros((n_verts, 2), dtype=np.float32)

    # BBox 内に投影される頂点のみ有効な UV を設定
    in_bbox = (
        in_image &
        (pixel_u >= crop_x1) & (pixel_u < crop_x2) &
        (pixel_v >= crop_y1) & (pixel_v < crop_y2)
    )

    uvs[in_bbox, 0] = (pixel_u[in_bbox] - crop_x1) / tex_w
    uvs[in_bbox, 1] = 1.0 - (pixel_v[in_bbox] - crop_y1) / tex_h  # V 上下反転

    # BBox 外の頂点 → テクスチャ範囲外 = 透明領域に向ける
    # UV=(0, 0) は crop の左下端で、SAMマスク外なら透明になる
    uvs[~in_bbox, 0] = 0.0
    uvs[~in_bbox, 1] = 0.0

    uvs = np.clip(uvs, 0.0, 1.0)

    n_in_bbox = np.count_nonzero(in_bbox)
    print(f"[Texture] UV in BBox: {n_in_bbox}/{n_verts} ({n_in_bbox/n_verts*100:.1f}%)")

    # === 7. PNG エンコード (RGBA) + base64 ===
    # Unity の Texture2D.LoadImage は画像をそのまま読み込む。
    # UV の V=0 は画像の下端に対応するため、UV 側で 1.0 - v の反転を行っている。
    # テクスチャ画像自体は反転しない（二重反転を防止）。
    encode_params = [cv2.IMWRITE_PNG_COMPRESSION, png_compress]
    # cv2 は BGRA で扱うが、PNG ファイル標準は RGBA。
    # cv2.imencode(".png") は内部で BGRA→RGBA 変換して正しい PNG を生成するので
    # texture_rgba (BGRA) をそのまま渡せば OK。
    success, encoded = cv2.imencode(".png", texture_rgba, encode_params)
    if not success:
        print("[Texture] PNG encoding failed")
        return None

    # デバッグ: テクスチャをファイルに保存（確認用）
    try:
        import os
        debug_dir = os.path.join("captures", "_debug")
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "texture_debug.png"), texture_rgba)
        print(f"[Texture] Debug saved: {debug_dir}/texture_debug.png")
    except Exception:
        pass

    texture_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    print(f"[Texture] PNG: {len(encoded)} bytes → base64: {len(texture_b64)} chars")

    return {
        "uvs": uvs.flatten().tolist(),
        "texture_base64": texture_b64,
        "texture_width": tex_w,
        "texture_height": tex_h,
        "uv_valid_ratio": round(uv_valid_ratio, 4),
    }