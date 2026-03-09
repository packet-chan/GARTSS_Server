"""
点群 → カラーメッシュ生成モジュール

SAM マスクから得た点群を Poisson Surface Reconstruction でメッシュ化し、
RGB 画像の色情報を頂点カラーとして転送する。

Unity 側でランタイムメッシュとして生成し、Normal 方向にアニメーションさせて
「部品が動いている様子」を可視化する。
"""

import numpy as np
from typing import Optional

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
    density_quantile: float = 0.05,
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

    Args:
        points: (N, 3) float64 — 3D座標 (Unity ワールド座標系)
        colors: (N, 3) uint8 or float — RGB (0-255 or 0-1)
        target_triangles: 目標三角形数
        poisson_depth: Poisson 再構成の深度 (高いほど詳細, 遅い)
        density_quantile: 低密度除去の閾値 (0.05 = 下位5%を除去)
        normal_radius: 法線推定の検索半径 [m]
        normal_max_nn: 法線推定の最大近傍数

    Returns:
        {
            "vertex_count": int,
            "triangle_count": int,
            "vertices": list[float],   # [x0,y0,z0, x1,y1,z1, ...]
            "triangles": list[int],    # [i0,i1,i2, i3,i4,i5, ...]
            "normals": list[float],    # [nx0,ny0,nz0, ...]
            "colors": list[float],     # [r0,g0,b0, ...] (0-1)
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
