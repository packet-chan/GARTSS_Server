"""
HMD姿勢補間モジュール
"""

import csv
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R, Slerp
from typing import Optional


class PoseInterpolator:
    def __init__(self, poses=None):
        self.timestamps: np.ndarray = np.array([], dtype=np.int64)
        self.positions: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        self.rotations: np.ndarray = np.zeros((0, 4), dtype=np.float64)

        if poses is not None:
            if isinstance(poses, (str, Path)):
                self.load_csv(poses)
            elif isinstance(poses, list):
                self.load_poses(poses)

    def load_csv(self, csv_path):
        csv_path = Path(csv_path)
        rows = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        self.load_poses(rows)

    def load_poses(self, poses: list):
        n = len(poses)
        self.timestamps = np.zeros(n, dtype=np.int64)
        self.positions = np.zeros((n, 3), dtype=np.float64)
        self.rotations = np.zeros((n, 4), dtype=np.float64)

        for i, row in enumerate(poses):
            self.timestamps[i] = int(row["timestamp_ms"])
            self.positions[i] = [
                float(row["pos_x"]),
                float(row["pos_y"]),
                float(row["pos_z"]),
            ]
            self.rotations[i] = [
                float(row["rot_x"]),
                float(row["rot_y"]),
                float(row["rot_z"]),
                float(row["rot_w"]),
            ]

        sort_idx = np.argsort(self.timestamps)
        self.timestamps = self.timestamps[sort_idx]
        self.positions = self.positions[sort_idx]
        self.rotations = self.rotations[sort_idx]

    def interpolate_pose(self, timestamp_ms: int) -> Optional[tuple]:
        if len(self.timestamps) == 0:
            return None

        if timestamp_ms <= self.timestamps[0]:
            return self.positions[0].copy(), self.rotations[0].copy()
        if timestamp_ms >= self.timestamps[-1]:
            return self.positions[-1].copy(), self.rotations[-1].copy()

        idx = np.searchsorted(self.timestamps, timestamp_ms, side="right") - 1
        idx = max(0, min(idx, len(self.timestamps) - 2))

        t0, t1 = self.timestamps[idx], self.timestamps[idx + 1]
        if t1 == t0:
            return self.positions[idx].copy(), self.rotations[idx].copy()

        alpha = (timestamp_ms - t0) / (t1 - t0)
        pos = (1 - alpha) * self.positions[idx] + alpha * self.positions[idx + 1]

        key_rots = R.from_quat(self.rotations[idx : idx + 2])
        slerp = Slerp([0.0, 1.0], key_rots)
        rot = slerp(alpha).as_quat()

        return pos, rot

    def __len__(self) -> int:
        return len(self.timestamps)
