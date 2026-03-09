"""
タスク → 操作方向マッピング

各タスク (drip_tray, rotary_knob 等) に対して、
どの種類の AR コンテンツを表示し、操作方向をどう決めるかを定義する。
"""

# action_type:
#   "pull"       — サーフェス面上でカメラ方向に引き出す (矢印)
#   "push"       — Normal 逆方向に押し込む (矢印)
#   "press"      — push と同じだが UI が異なる (プレスアイコン)
#   "rotate_cw"  — Normal 軸周りに時計回り (回転ガイド)
#   "rotate_ccw" — Normal 軸周りに反時計回り (回転ガイド)

ACTION_MAP: dict[str, dict] = {
    "drip_tray": {
        "action_type": "pull",
        "ar_content": "arrow",
        "arrow_length_m": 0.15,
        "label_ja": "トレーを手前に引き出す",
        "label_en": "Pull the drip tray toward you",
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
    },
    "power_button": {
        "action_type": "press",
        "ar_content": "arrow",
        "arrow_length_m": 0.03,
        "label_ja": "電源ボタンを押す",
        "label_en": "Press the power button",
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
