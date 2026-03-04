"""
SAM2 セグメンテーションモジュール
BBoxプロンプトからピクセルマスクを生成し、輪郭を3D座標に変換する
"""

import numpy as np
import cv2
import torch
from pathlib import Path

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
) -> dict:
    """
    BBoxプロンプトでSAM2セグメンテーションを実行

    Args:
        image_bgr: BGR画像 (OpenCV形式)
        bbox: [x_min, y_min, x_max, y_max] ピクセル座標

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

    # BBoxプロンプト
    input_box = np.array(bbox)  # [x_min, y_min, x_max, y_max]

    # 推論
    masks, scores, _ = predictor.predict(
        box=input_box,
        multimask_output=False,  # 最も確信度の高いマスク1つだけ
    )

    mask = masks[0]  # (H, W) bool
    score = float(scores[0])

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
