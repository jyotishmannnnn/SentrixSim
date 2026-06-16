"""Dataset loading + task construction for the baseline benchmarks.

Reads dataset_v0.1 per-episode Parquet files and builds three task datasets with
a SINGLE episode-level train/test split (stratified by event) so no window leaks
across the split.

Tasks
-----
* event   : per-EPISODE, 9-class (idle..grasp). Clip-level gesture id.
* contact : per-WINDOW binary (any-finger contact, majority of the window).
* slip    : per-WINDOW binary (any-finger slip, any sample in the window).

Representations
---------------
* tree models : engineered features (per-channel mean/std/min/max).
* CNN         : standardized raw [C, T] tensors (episodes resampled to T=256;
                windows are L=160 at 1600 Hz = 100 ms).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

EVENTS = ["idle", "tap", "press", "hold", "shear", "slip", "release", "pinch", "grasp"]
EVENT_ID = {e: i for i, e in enumerate(EVENTS)}
FINGERS = ["thumb", "index", "middle", "ring", "pinky", "palm"]

WIN = 160          # window length (samples) = 100 ms @ 1600 Hz
EP_LEN = 256       # resample length for the per-episode CNN
SEED = 42


def sensor_columns() -> list[str]:
    cols = [f"tactile.b{i:02d}.{ax}_uT" for i in range(21) for ax in ("bx", "by", "bz")]
    cols += [f"dyn.{f}.{ax}_g" for f in ("thumb", "index", "middle") for ax in ("ax", "ay", "az")]
    cols += [f"dyn.{f}.temp_degC" for f in ("thumb", "index", "middle")]
    return cols


N_CH = len(sensor_columns())  # 75


@dataclass
class EpisodeData:
    event: str
    X: np.ndarray          # (n_samples, N_CH) sensor matrix
    contact: np.ndarray    # (n_samples,) bool
    slip: np.ndarray       # (n_samples,) bool


def _read_episode(path: Path) -> EpisodeData:
    t = pq.read_table(path)
    names = set(t.column_names)
    cols = sensor_columns()
    X = np.column_stack([t.column(c).to_numpy(zero_copy_only=False) for c in cols]).astype(np.float32)
    n = X.shape[0]
    contact = np.zeros(n, bool)
    slip = np.zeros(n, bool)
    for f in FINGERS:
        cc = f"label.{f}.contact"
        sc = f"label.{f}.slip"
        if cc in names:
            contact |= t.column(cc).to_numpy(zero_copy_only=False).astype(bool)
        if sc in names:
            slip |= t.column(sc).to_numpy(zero_copy_only=False).astype(bool)
    event = path.stem.split("__")[0]
    return EpisodeData(event=event, X=X, contact=contact, slip=slip)


def load_all(parquet_root: str | Path) -> list[tuple[Path, str]]:
    root = Path(parquet_root)
    items = []
    for ev in EVENTS:
        for p in sorted((root / ev).glob("*.parquet")):
            items.append((p, ev))
    return items


def episode_split(items, test_size=0.2, seed=SEED):
    paths = [p for p, _ in items]
    evs = [e for _, e in items]
    tr, te = train_test_split(range(len(items)), test_size=test_size,
                              random_state=seed, stratify=evs)
    return ([paths[i] for i in tr], [paths[i] for i in te])


# ---- feature engineering ----
def _win_features(Xw: np.ndarray) -> np.ndarray:
    """Xw: (nw, WIN, N_CH) -> (nw, N_CH*4)."""
    return np.concatenate([Xw.mean(1), Xw.std(1), Xw.min(1), Xw.max(1)], axis=1)


def _ep_features(X: np.ndarray) -> np.ndarray:
    """X: (n, N_CH) -> (N_CH*4 + 1,) incl. duration proxy (n)."""
    f = np.concatenate([X.mean(0), X.std(0), X.min(0), X.max(0)])
    return np.append(f, X.shape[0]).astype(np.float32)


def _resample(X: np.ndarray, length: int) -> np.ndarray:
    """Linear-resample (n, C) -> (length, C)."""
    n = X.shape[0]
    if n == length:
        return X
    xi = np.linspace(0, n - 1, length)
    xp = np.arange(n)
    return np.stack([np.interp(xi, xp, X[:, c]) for c in range(X.shape[1])], axis=1).astype(np.float32)


def _windows(X: np.ndarray, win: int = WIN):
    nw = X.shape[0] // win
    if nw == 0:
        return np.empty((0, win, X.shape[1]), np.float32)
    return X[: nw * win].reshape(nw, win, X.shape[1])


def build_tasks(parquet_root):
    """Returns a dict with arrays for the 3 tasks (train/test) in both
    feature (tree) and raw (cnn) forms."""
    items = load_all(parquet_root)
    train_paths, test_paths = episode_split(items)

    def proc(paths):
        ep_feat, ep_raw, ep_y = [], [], []
        w_feat, w_raw, w_contact, w_slip = [], [], [], []
        for p in paths:
            ed = _read_episode(p)
            # event (per-episode)
            ep_feat.append(_ep_features(ed.X))
            ep_raw.append(_resample(ed.X, EP_LEN))
            ep_y.append(EVENT_ID[ed.event])
            # windows
            Xw = _windows(ed.X)
            if Xw.shape[0] == 0:
                continue
            cw = _windows(ed.contact[:, None].astype(np.float32))[:, :, 0]
            sw = _windows(ed.slip[:, None].astype(np.float32))[:, :, 0]
            w_feat.append(_win_features(Xw))
            w_raw.append(Xw)
            w_contact.append((cw.mean(1) >= 0.5).astype(int))
            w_slip.append((sw.mean(1) > 0.0).astype(int))
        return {
            "ep_feat": np.array(ep_feat, np.float32),
            "ep_raw": np.array(ep_raw, np.float32),       # (E, EP_LEN, C)
            "ep_y": np.array(ep_y, int),
            "w_feat": np.concatenate(w_feat),
            "w_raw": np.concatenate(w_raw),               # (W, WIN, C)
            "w_contact": np.concatenate(w_contact),
            "w_slip": np.concatenate(w_slip),
        }

    return {"train": proc(train_paths), "test": proc(test_paths),
            "n_train_ep": len(train_paths), "n_test_ep": len(test_paths)}
