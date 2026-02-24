# AR Work Assist Server

Zero-shot AR Authoring System — Quest 3 RGB-Depth Alignment Server

## Phase 1: サーバー基盤

### ファイル構成

```
ar-work-assist-server/
├── server.py                    # FastAPI メインサーバー
├── core/
│   ├── alignment.py             # AlignmentEngine (3D再投影コア)
│   ├── coordinate_utils.py      # 座標変換 (Camera2→Unity→Open3D)
│   └── pose_interpolator.py     # HMD姿勢補間 (自前実装)
├── models/
│   └── schemas.py               # Pydantic スキーマ
├── vision/                      # [Phase 3] LLM + SAM
├── tests/
│   └── test_phase1.py           # Phase 1 テスト
├── requirements.txt
└── README.md
```

### セットアップ

```bash
pip install -r requirements.txt
```

### サーバー起動

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### API エンドポイント

| Method | Path | 説明 |
|--------|------|------|
| POST | `/session/init` | セッション初期化 (カメラパラメータ) |
| POST | `/session/{id}/capture` | RGB+Depth受信 → アライメント |
| GET | `/session/{id}/depth?u=&v=` | ピクセル→Depth/3D座標クエリ |
| POST | `/session/{id}/analyze` | [Phase 3] LLM+SAM解析 |
| GET | `/session/{id}/info` | セッション状態 |
| DELETE | `/session/{id}` | セッション削除 |
| GET | `/health` | ヘルスチェック |

### テスト実行

```bash
PYTHONPATH=. python tests/test_phase1.py
```

### 変更点 (reproject_align_v2.py からの変更)

- ファイルシステム直読み → APIでデータ受け取り
- PoseInterpolator → 自前実装 (外部リポジトリ依存排除)
- DepthToRGBProjector → AlignmentEngine (セッション管理対応)
- matplotlib可視化 → 削除 (サーバーには不要)
- CLI → FastAPI エンドポイント
