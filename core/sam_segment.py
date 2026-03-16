"""
SAM2 セグメンテーションモジュール
BBoxプロンプトからピクセルマスクを生成し、輪郭を3D座標に変換する

v2: 方向別マージン対応 + 小領域ノイズ除去
"""

import numpy as np
import cv2
from pathlib import Path

# torch は SAM2 使用時のみ必要（遅延インポート）
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# グローバルにモデルを保持（初回ロード後に再利用）
_predictor = None


def get_predictor():
    """SAM2モデルをロード（初回のみ）"""
    global _predictor
    if _predictor is not None:
        return _predictor

    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # チェックポイントを自動ダウンロード or ローカルパスから読み込み
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "sam2.1_hiera_tiny.pt"

    if not checkpoint_path.exists():
        print("[SAM2] Downloading sam2.1_hiera_tiny checkpoint...")
        import urllib.request
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
        urllib.request.urlretrieve(url, str(checkpoint_path))
        print("[SAM2] Download complete.")

    print("[SAM2] Loading model...")
    sam2_model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_t.yaml",
        str(checkpoint_path),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    _predictor = SAM2ImagePredictor(sam2_model)
    print(f"[SAM2] Model loaded on {'cuda' if torch.cuda.is_available() else 'cpu'}")
    return _predictor


def segment_with_bbox(
    image_bgr: np.ndarray,
    bbox: list[float],
    margin_ratio: float = 0.1,
    margin_ratios: dict | None = None,
) -> dict:
    """
    BBoxプロンプトでSAM2セグメンテーションを実行

    Args:
        image_bgr: BGR画像 (OpenCV形式)
        bbox: [x_min, y_min, x_max, y_max] ピクセル座標
        margin_ratio: BBoxを各方向に何%広げるか (0.1 = 10%)。
                      margin_ratios が指定されている場合はそちらが優先。
        margin_ratios: 方向別マージン比率 {"top": 0.05, "bottom": 0.3, "left": 0.1, "right": 0.1}
                       指定しない方向は margin_ratio がフォールバックとして使われる。

    Returns:
        {
            "mask": np.ndarray (H, W) bool,
            "contours": list of np.ndarray (輪郭座標),
            "score": float,
        }
    """
    predictor = get_predictor()

    # BGR → RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # 画像をセット
    predictor.set_image(image_rgb)

    # BBox にマージンを追加してSAMに渡す
    h, w = image_bgr.shape[:2]
    x_min, y_min, x_max, y_max = bbox
    bw = x_max - x_min
    bh = y_max - y_min

    # 方向別マージン
    if margin_ratios is not None:
        m_top = bh * margin_ratios.get("top", margin_ratio)
        m_bottom = bh * margin_ratios.get("bottom", margin_ratio)
        m_left = bw * margin_ratios.get("left", margin_ratio)
        m_right = bw * margin_ratios.get("right", margin_ratio)
    else:
        m_top = bh * margin_ratio
        m_bottom = bh * margin_ratio
        m_left = bw * margin_ratio
        m_right = bw * margin_ratio

    expanded_bbox = [
        max(0, x_min - m_left),
        max(0, y_min - m_top),
        min(w, x_max + m_right),
        min(h, y_max + m_bottom),
    ]
    print(f"  [SAM2] BBox: [{x_min:.0f},{y_min:.0f},{x_max:.0f},{y_max:.0f}] "
          f"→ expanded: [{expanded_bbox[0]:.0f},{expanded_bbox[1]:.0f},"
          f"{expanded_bbox[2]:.0f},{expanded_bbox[3]:.0f}] "
          f"(margins: top={m_top:.0f}, bottom={m_bottom:.0f}, "
          f"left={m_left:.0f}, right={m_right:.0f})")

    # BBoxプロンプト
    input_box = np.array(expanded_bbox)  # [x_min, y_min, x_max, y_max]

    # 推論
    masks, scores, _ = predictor.predict(
        box=input_box,
        multimask_output=False,  # 最も確信度の高いマスク1つだけ
    )

    mask = masks[0]  # (H, W) bool
    score = float(scores[0])

    # === BBox クリッピング ===
    # SAM にはexpanded BBoxを渡すが、結果は元のBBox範囲にクリップ。
    # expanded 領域にはみ出したセグメントは対象物ではない可能性が高い。
    bbox_clipped = mask.copy()
    x_min_i, y_min_i = int(round(x_min)), int(round(y_min))
    x_max_i, y_max_i = int(round(x_max)), int(round(y_max))
    bbox_clipped[:y_min_i, :] = False
    bbox_clipped[y_max_i:, :] = False
    bbox_clipped[:, :x_min_i] = False
    bbox_clipped[:, x_max_i:] = False

    clipped_pixels = np.count_nonzero(mask) - np.count_nonzero(bbox_clipped)
    if clipped_pixels > 0:
        print(f"  [SAM2] BBox clipping removed {clipped_pixels} pixels outside original BBox")
    mask = bbox_clipped

    # === 最大連結成分のみ保持 ===
    total_mask_pixels = np.count_nonzero(mask)
    if total_mask_pixels > 0:
        mask_u8 = (mask * 255).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

        if num_labels > 2:  # 背景 + 2つ以上の領域がある場合
            # 最大の連結成分のみ残す
            largest_label = 0
            largest_area = 0
            for label_id in range(1, num_labels):
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area > largest_area:
                    largest_area = area
                    largest_label = label_id

            noise_removed = total_mask_pixels - largest_area
            mask = (labels == largest_label)
            if noise_removed > 0:
                print(f"  [SAM2] Kept largest component only: removed {noise_removed} pixels "
                      f"({noise_removed / total_mask_pixels * 100:.1f}%) from {num_labels - 1} regions")

    # 輪郭を抽出
    mask_u8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"  [SAM2] Mask pixels: {np.count_nonzero(mask)}, score: {score:.3f}, contours: {len(contours)}")

    return {
        "mask": mask,
        "contours": contours,
        "score": score,
    }


def mask_contour_to_3d(
    contours: list,
    engine,
    simplify_epsilon: float = 3.0,
    sample_step: int = 1,
) -> list[list[float]]:
    """
    マスクの輪郭ピクセル座標をDepthで3D座標に変換

    Args:
        contours: cv2.findContoursの出力
        engine: AlignmentEngine (get_3d_point_unity メソッドを持つ)
        simplify_epsilon: 輪郭を簡略化する閾値 (大きいほど頂点が減る)
        sample_step: 頂点のサンプリングステップ

    Returns:
        3D頂点リスト [[x, y, z], ...]
    """
    if not contours:
        return []

    # 最大の輪郭を使用
    largest_contour = max(contours, key=cv2.contourArea)

    # 輪郭を簡略化 (頂点数を減らす)
    simplified = cv2.approxPolyDP(largest_contour, simplify_epsilon, closed=True)

    vertices_3d = []
    for i in range(0, len(simplified), sample_step):
        px, py = simplified[i][0]
        point_3d = engine.get_3d_point_unity(float(px), float(py))
        if point_3d is not None:
            vertices_3d.append(point_3d.tolist())

    print(f"  [SAM2] Contour: {len(largest_contour)} pts → simplified: {len(simplified)} pts → 3D: {len(vertices_3d)} pts")
    return vertices_3d


def save_mask_visualization(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    contours: list,
    bbox: list[float],
    output_path: str,
    label: str = "",
):
    """マスク結果を画像に描画して保存"""
    vis = image_bgr.copy()

    # マスクを半透明で重ねる (緑色)
    overlay = vis.copy()
    overlay[mask.astype(bool)] = [0, 255, 0]
    vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

    # 輪郭を描画 (赤)
    cv2.drawContours(vis, contours, -1, (0, 0, 255), 2)

    # BBoxを描画 (黄)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # ラベル
    if label:
        cv2.putText(vis, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imwrite(output_path, vis)
    print(f"  [SAM2] Saved: {output_path}")


def mask_to_point_cloud(
    mask: np.ndarray,
    engine,
    image_bgr: np.ndarray = None,
    sample_step: int = 1,
    sor_neighbors: int = 20,
    sor_std_ratio: float = 2.0,
) -> dict:
    """
    SAM マスク領域内の全有効ピクセルを 3D 点群に変換する。

    Args:
        mask: (H, W) bool マスク
        engine: AlignmentEngine (get_3d_points_batch, _aligned_depth を持つ)
        image_bgr: BGR 画像 (色付き点群にする場合)
        sample_step: ピクセルサンプリング間隔 (1=全ピクセル, 2=1/4, 3=1/9...)
        sor_neighbors: 統計的外れ値除去の近傍数 (0で無効化)
        sor_std_ratio: 平均距離 + std_ratio * 標準偏差 より遠い点を外れ値とする

    Returns:
        {
            "points": np.ndarray (N, 3) Unity ワールド座標,
            "colors": np.ndarray (N, 3) RGB 0-255 or None,
            "count": int,
        }
    """
    if mask is None or engine._aligned_depth is None:
        return {"points": np.zeros((0, 3)), "colors": None, "count": 0}

    # マスク内の有効ピクセル座標を取得
    v_indices, u_indices = np.where(mask)

    if len(v_indices) == 0:
        return {"points": np.zeros((0, 3)), "colors": None, "count": 0}

    # サンプリング
    if sample_step > 1:
        indices = np.arange(0, len(v_indices), sample_step)
        v_indices = v_indices[indices]
        u_indices = u_indices[indices]

    print(f"  [PointCloud] Mask pixels: {np.count_nonzero(mask)}, sampled: {len(u_indices)}")

    # バッチで 3D 変換
    u_float = u_indices.astype(np.float64)
    v_float = v_indices.astype(np.float64)
    points_unity, valid = engine.get_3d_points_batch(u_float, v_float)

    print(f"  [PointCloud] Valid 3D points: {len(points_unity)}")

    # 色情報
    colors = None
    if image_bgr is not None and len(points_unity) > 0:
        u_valid = u_indices[valid]
        v_valid = v_indices[valid]
        # BGR → RGB
        colors_bgr = image_bgr[v_valid, u_valid]
        colors = colors_bgr[:, ::-1].copy()  # BGR → RGB

    # === 統計的外れ値除去 (SOR) ===
    # Depth のノイズで飛び散った点を除去する。
    # 各点の k 近傍までの平均距離を計算し、
    # mean + std_ratio * std より遠い点を外れ値として除去。
    if sor_neighbors > 0 and len(points_unity) > sor_neighbors:
        from scipy.spatial import KDTree
        tree = KDTree(points_unity)
        dists, _ = tree.query(points_unity, k=sor_neighbors + 1)  # 自分自身を含む
        mean_dists = dists[:, 1:].mean(axis=1)  # 自分自身(0距離)を除く

        global_mean = mean_dists.mean()
        global_std = mean_dists.std()
        threshold = global_mean + sor_std_ratio * global_std

        inlier_mask = mean_dists < threshold
        n_before = len(points_unity)
        points_unity = points_unity[inlier_mask]
        if colors is not None:
            colors = colors[inlier_mask]
        n_removed = n_before - len(points_unity)
        if n_removed > 0:
            print(f"  [PointCloud] SOR: removed {n_removed} outliers "
                  f"({n_removed / n_before * 100:.1f}%), threshold={threshold:.5f}m")

    return {
        "points": points_unity,
        "colors": colors,
        "count": len(points_unity),
    }


def save_ply(
    output_path: str,
    points: np.ndarray,
    colors: np.ndarray = None,
):
    """
    点群を PLY 形式で保存する。Open3D なしで直接書き出し。

    Args:
        output_path: 保存先パス (.ply)
        points: (N, 3) float64 座標
        colors: (N, 3) uint8 RGB (省略可)
    """
    n = len(points)
    if n == 0:
        print(f"  [PLY] No points to save")
        return

    has_color = colors is not None and len(colors) == n

    with open(output_path, "w") as f:
        # PLY ヘッダー
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_color:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        # データ
        for i in range(n):
            x, y, z = points[i]
            if has_color:
                r, g, b = colors[i]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

    print(f"  [PLY] Saved: {output_path} ({n} points, color={'yes' if has_color else 'no'})")