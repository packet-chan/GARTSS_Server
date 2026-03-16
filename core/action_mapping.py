"""
タスク → 操作方向マッピング

各タスク (drip_tray, rotary_knob 等) に対して、
どの種類の AR コンテンツを表示し、操作方向をどう決めるかを定義する。

v2: sam_margin_ratios を追加。タスクに応じた方向別SAMマージン。
"""

# action_type:
#   "pull"       — サーフェス面上でカメラ方向に引き出す (矢印)
#   "push"       — Normal 逆方向に押し込む (矢印)
#   "press"      — push と同じだが UI が異なる (プレスアイコン)
#   "rotate_cw"  — Normal 軸周りに時計回り (回転ガイド)
#   "rotate_ccw" — Normal 軸周りに反時計回り (回転ガイド)

# sam_margin_ratios:
#   SAM BBox を拡張する方向別比率。
#   {"top": 0.05, "bottom": 0.3, "left": 0.1, "right": 0.1}
#   指定しない場合はデフォルト (全方向 0.1) が使われる。

ACTION_MAP: dict[str, dict] = {
    "drip_tray": {
        "action_type": "pull",
        "ar_content": "arrow",
        "arrow_length_m": 0.15,
        "label_ja": "トレーを手前に引き出す",
        "label_en": "Pull the drip tray toward you",
        # トレーは上面＋前面の2面構造。
        # BBoxが上にずれがちなので、下方向に大きくマージンを取り前面もカバーする。
        # 上方向は小さくしてノズル付近を含めない。
        "sam_margin_ratios": {"top": 0.05, "bottom": 0.35, "left": 0.05, "right": 0.05},
    },
    "water_tank": {
        "action_type": "pull",
        "ar_content": "arrow",
        "arrow_length_m": 0.12,
        "label_ja": "タンクを持ち上げて外す",
        "label_en": "Lift and remove the water tank",
    },
    "rotary_knob": {
        "action_type": "rotate_cw",
        "ar_content": "rotation_guide",
        "arrow_length_m": 0.05,
        "label_ja": "ダイヤルを時計回りに回す",
        "label_en": "Turn the knob clockwise",
        # ノブはタイトにセグメンテーションしたい
        "sam_margin_ratios": {"top": 0.08, "bottom": 0.08, "left": 0.08, "right": 0.08},
    },
    "power_button": {
        "action_type": "press",
        "ar_content": "arrow",
        "arrow_length_m": 0.03,
        "label_ja": "電源ボタンを押す",
        "label_en": "Press the power button",
        "sam_margin_ratios": {"top": 0.08, "bottom": 0.08, "left": 0.08, "right": 0.08},
    },
    "bean_hopper_lid": {
        "action_type": "pull",
        "ar_content": "arrow",
        "arrow_length_m": 0.08,
        "label_ja": "豆ホッパーの蓋を開ける",
        "label_en": "Open the bean hopper lid",
    },
}

DEFAULT_ACTION = {
    "action_type": "pull",
    "ar_content": "arrow",
    "arrow_length_m": 0.10,
    "label_ja": "この部品を操作する",
    "label_en": "Interact with this component",
}


def get_action_config(task: str) -> dict:
    """タスク名から操作設定を取得。未知のタスクはデフォルトを返す。"""
    return ACTION_MAP.get(task, DEFAULT_ACTION)