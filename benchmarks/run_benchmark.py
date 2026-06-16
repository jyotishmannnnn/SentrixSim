"""Baseline benchmark: 3 tasks x 3 models on Sentrix Dataset v0.1.

Tasks   : event (9-class, per-episode), contact (binary, per-window),
          slip (binary, per-window).
Models  : XGBoost, RandomForest (engineered features); small 1D CNN (raw).
Reports : accuracy, F1 (macro; + positive-class F1 for binary), confusion matrix.

Run:  python benchmarks/run_benchmark.py --parquet ./dataset_v0.1/parquet --out ./benchmarks
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from xgboost import XGBClassifier

import data as D


# --------------------------------------------------------------------------- #
# 1D CNN (PyTorch, CPU)
# --------------------------------------------------------------------------- #
def train_cnn(Xtr, ytr, Xte, yte, n_classes, epochs, batch, class_weight=None,
              seed=42):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)

    # (N,T,C) -> (N,C,T); per-channel standardization from train
    def prep(X):
        return np.transpose(X, (0, 2, 1)).astype(np.float32)
    Xtr, Xte = prep(Xtr), prep(Xte)
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sd = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    C, T = Xtr.shape[1], Xtr.shape[2]

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(C, 32, 7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(64, n_classes)

        def forward(self, x):
            return self.head(self.net(x).squeeze(-1))

    model = CNN()
    dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr).long()),
                    batch_size=batch, shuffle=True)
    w = None if class_weight is None else torch.tensor(class_weight, dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(Xte)).argmax(1).numpy()
    return pred


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def evaluate(y_true, y_pred, labels, binary: bool):
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if binary:
        out["f1_positive"] = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    return out


def run_tree_models(Xtr, ytr, Xte, yte, labels, binary, results, task):
    n_cls = len(labels)
    # class imbalance handling
    spw = None
    if binary:
        pos = max(int((ytr == 1).sum()), 1)
        neg = int((ytr == 0).sum())
        spw = neg / pos

    xgb_params = dict(
        n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.9,
        colsample_bytree=0.9, tree_method="hist", n_jobs=-1,
        eval_metric="logloss", random_state=42,
    )
    if binary:
        xgb_params["objective"] = "binary:logistic"
        xgb_params["scale_pos_weight"] = spw
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(Xtr, ytr)
    results[task]["XGBoost"] = evaluate(yte, xgb.predict(Xte), labels, binary)

    rf = RandomForestClassifier(
        n_estimators=300, n_jobs=-1, random_state=42,
        class_weight="balanced" if binary else None,
    )
    rf.fit(Xtr, ytr)
    results[task]["RandomForest"] = evaluate(yte, rf.predict(Xte), labels, binary)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="./dataset_v0.1/parquet")
    ap.add_argument("--out", default="./benchmarks")
    args = ap.parse_args()

    print("Loading dataset ...")
    ds = D.build_tasks(args.parquet)
    tr, te = ds["train"], ds["test"]
    results = {"event": {}, "contact": {}, "slip": {}}
    meta = {
        "n_train_episodes": ds["n_train_ep"], "n_test_episodes": ds["n_test_ep"],
        "n_train_windows": int(tr["w_feat"].shape[0]),
        "n_test_windows": int(te["w_feat"].shape[0]),
        "n_features": int(tr["w_feat"].shape[1]),
        "n_channels": D.N_CH, "window_samples": D.WIN, "episode_resample": D.EP_LEN,
    }
    print(f"  episodes: {meta['n_train_episodes']} train / {meta['n_test_episodes']} test")
    print(f"  windows : {meta['n_train_windows']} train / {meta['n_test_windows']} test")

    # ---- Task 1: event (9-class, per-episode) ----
    print("Task: event classification ...")
    ev_labels = list(range(len(D.EVENTS)))
    run_tree_models(tr["ep_feat"], tr["ep_y"], te["ep_feat"], te["ep_y"],
                    ev_labels, binary=False, results=results, task="event")
    pred = train_cnn(tr["ep_raw"], tr["ep_y"], te["ep_raw"], te["ep_y"],
                     n_classes=9, epochs=60, batch=32)
    results["event"]["CNN1D"] = evaluate(te["ep_y"], pred, ev_labels, binary=False)

    # ---- Task 2: contact (binary, per-window) ----
    print("Task: contact detection ...")
    run_tree_models(tr["w_feat"], tr["w_contact"], te["w_feat"], te["w_contact"],
                    [0, 1], binary=True, results=results, task="contact")
    cw = _class_weight(tr["w_contact"])
    pred = train_cnn(tr["w_raw"], tr["w_contact"], te["w_raw"], te["w_contact"],
                     n_classes=2, epochs=15, batch=128, class_weight=cw)
    results["contact"]["CNN1D"] = evaluate(te["w_contact"], pred, [0, 1], binary=True)

    # ---- Task 3: slip (binary, per-window) ----
    print("Task: slip detection ...")
    run_tree_models(tr["w_feat"], tr["w_slip"], te["w_feat"], te["w_slip"],
                    [0, 1], binary=True, results=results, task="slip")
    sw = _class_weight(tr["w_slip"])
    pred = train_cnn(tr["w_raw"], tr["w_slip"], te["w_raw"], te["w_slip"],
                     n_classes=2, epochs=15, batch=128, class_weight=sw)
    results["slip"]["CNN1D"] = evaluate(te["w_slip"], pred, [0, 1], binary=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "results": results,
               "class_balance": {
                   "contact_test_pos_frac": float(te["w_contact"].mean()),
                   "slip_test_pos_frac": float(te["w_slip"].mean()),
               }}
    (out / "benchmark_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_report(out, payload)
    print(f"\nWrote {out/'benchmark_report.md'}")


def _class_weight(y):
    n = len(y)
    n1 = max(int((y == 1).sum()), 1)
    n0 = max(int((y == 0).sum()), 1)
    return [n / (2 * n0), n / (2 * n1)]


def _fmt_cm(cm, labels):
    head = "      " + " ".join(f"{l:>6}" for l in labels)
    rows = [head]
    for i, l in enumerate(labels):
        rows.append(f"{l:>5} " + " ".join(f"{v:>6}" for v in cm[i]))
    return "```\n" + "\n".join(rows) + "\n```"


def _write_report(out: Path, payload: dict):
    m, r = payload["meta"], payload["results"]
    L = []
    A = L.append
    A("# Sentrix Dataset v0.1 - Baseline Benchmark Report\n")
    A("Three tasks x three model families, trained on `dataset_v0.1`. "
      "Split is **episode-level** (stratified by event), so no window leaks "
      "between train and test.\n")
    A("## Setup\n")
    A(f"- Channels: {m['n_channels']} (63 BMM350 axes + 9 accel + 3 temp)")
    A(f"- Event task: per-episode, 9-class; {m['n_train_episodes']} train / "
      f"{m['n_test_episodes']} test episodes; CNN on resampled [{m['n_channels']}, "
      f"{m['episode_resample']}] series; trees on {m['n_features']} features "
      "(per-channel mean/std/min/max).")
    A(f"- Contact/slip tasks: per-window ({m['window_samples']} samples = 100 ms @ "
      f"1600 Hz); {m['n_train_windows']} train / {m['n_test_windows']} test windows.")
    A(f"- Test positive fraction - contact: {payload['class_balance']['contact_test_pos_frac']:.3f}, "
      f"slip: {payload['class_balance']['slip_test_pos_frac']:.3f}.")
    A("- Trees: XGBoost (hist, 300 trees) & RandomForest (300, balanced for "
      "binary). CNN: Conv1d(7)->BN->ReLU->Pool->Conv1d(5)->BN->ReLU->GAP->FC, "
      "Adam 1e-3, CE (class-weighted for binary).\n")

    task_titles = {"event": "1. Event classification (9-class)",
                   "contact": "2. Contact detection (binary)",
                   "slip": "3. Slip detection (binary)"}
    label_sets = {"event": D.EVENTS, "contact": ["no", "yes"], "slip": ["no", "yes"]}
    for task in ("event", "contact", "slip"):
        A(f"## {task_titles[task]}\n")
        binary = task != "event"
        A("| Model | Accuracy | F1 (macro) |" + (" F1 (positive) |" if binary else ""))
        A("|---|---|---|" + ("---|" if binary else ""))
        for mdl in ("XGBoost", "RandomForest", "CNN1D"):
            v = r[task][mdl]
            row = f"| {mdl} | {v['accuracy']:.4f} | {v['f1_macro']:.4f} |"
            if binary:
                row += f" {v.get('f1_positive', 0):.4f} |"
            A(row)
        A("")
        # confusion matrices
        for mdl in ("XGBoost", "RandomForest", "CNN1D"):
            A(f"**{mdl} confusion matrix** (rows=true, cols=pred):\n")
            A(_fmt_cm(r[task][mdl]["confusion"], label_sets[task]))
            A("")

    A("## Notes & caveats\n")
    A("- Data is SentrixSim v0.1 **relative/shape-only** physics: signal timing, "
      "noise, quantization and topology are datasheet/spec-true, but absolute "
      "force/field magnitudes are not. Scores measure separability of the "
      "simulated structure, **not** real-hardware performance.")
    A("- Event labels are clip-level; the pre-contact (idle/reach) portions of "
      "non-idle gestures resemble idle, which bounds achievable per-episode "
      "accuracy and is the main confusion source.")
    A("- Slip is rare and only present in the `slip` gesture after onset, so the "
      "slip task is highly imbalanced; read F1(positive) and the confusion "
      "matrix, not accuracy.")
    A("- Drift/noise realizations are the only cross-instance variation; these "
      "baselines will need re-fitting once calibrated hardware data replaces the "
      "relative scales (dataset v0.2).")
    (out / "benchmark_report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
