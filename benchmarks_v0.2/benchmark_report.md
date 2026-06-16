# Sentrix Dataset v0.2 - Baseline Benchmark Report

Three tasks x three model families, trained on `dataset_v0.2`. Split is **episode-level** (stratified by event), so no window leaks between train and test.

## Comparison vs v0.1

| Task | Model | v0.1 | v0.2 | delta |
|---|---|---|---|---|
| event | XGBoost | 0.994 | 0.904 | -0.090 |
| event | RandomForest | 1.000 | 0.860 | -0.140 |
| event | CNN1D | 0.829 | 0.686 | -0.143 |
| contact | XGBoost | 1.000 | 0.989 | -0.010 |
| contact | RandomForest | 0.998 | 0.991 | -0.008 |
| contact | CNN1D | 0.995 | 0.982 | -0.013 |
| slip | XGBoost | 1.000 | 0.983 | -0.017 |
| slip | RandomForest | 1.000 | 0.970 | -0.030 |
| slip | CNN1D | 0.987 | 0.861 | -0.126 |

(metric = F1 macro; targets - event 0.80-0.95, slip 0.75-0.95, contact 0.90-0.99)

## Setup

- Channels: 75 (63 BMM350 axes + 9 accel + 3 temp)
- Event task: per-episode, 9-class; 720 train / 180 test episodes; CNN on resampled [75, 256] series; trees on 300 features (per-channel mean/std/min/max).
- Contact/slip tasks: per-window (160 samples = 100 ms @ 1600 Hz); 12637 train / 3203 test windows.
- Test positive fraction - contact: 0.487, slip: 0.153.
- Trees: XGBoost (hist, 300 trees) & RandomForest (300, balanced for binary). CNN: Conv1d(7)->BN->ReLU->Pool->Conv1d(5)->BN->ReLU->GAP->FC, Adam 1e-3, CE (class-weighted for binary).

## 1. Event classification (9-class)

| Model | Accuracy | F1 (macro) |
|---|---|---|
| XGBoost | 0.9056 | 0.9040 |
| RandomForest | 0.8667 | 0.8596 |
| CNN1D | 0.6889 | 0.6860 |

**XGBoost confusion matrix** (rows=true, cols=pred):

```
        idle    tap  press   hold  shear   slip release  pinch  grasp
 idle     20      0      0      0      0      0      0      0      0
  tap      0     20      0      0      0      0      0      0      0
press      0      0     11      1      0      0      8      0      0
 hold      0      0      5     14      0      0      1      0      0
shear      0      0      0      0     20      0      0      0      0
 slip      0      0      0      0      1     19      0      0      0
release      0      0      1      0      0      0     19      0      0
pinch      0      0      0      0      0      0      0     20      0
grasp      0      0      0      0      0      0      0      0     20
```

**RandomForest confusion matrix** (rows=true, cols=pred):

```
        idle    tap  press   hold  shear   slip release  pinch  grasp
 idle     20      0      0      0      0      0      0      0      0
  tap      0     20      0      0      0      0      0      0      0
press      0      0      7      6      0      0      7      0      0
 hold      0      0      8     10      0      0      2      0      0
shear      0      0      0      0     20      0      0      0      0
 slip      0      0      0      0      0     20      0      0      0
release      0      0      1      0      0      0     19      0      0
pinch      0      0      0      0      0      0      0     20      0
grasp      0      0      0      0      0      0      0      0     20
```

**CNN1D confusion matrix** (rows=true, cols=pred):

```
        idle    tap  press   hold  shear   slip release  pinch  grasp
 idle     14      6      0      0      0      0      0      0      0
  tap      2     18      0      0      0      0      0      0      0
press      0      0      6     10      0      0      4      0      0
 hold      0      0      5     12      0      0      3      0      0
shear      0      0      0      0      9     11      0      0      0
 slip      0      0      0      1      6     13      0      0      0
release      0      0      6      2      0      0     12      0      0
pinch      0      0      0      0      0      0      0     20      0
grasp      0      0      0      0      0      0      0      0     20
```

## 2. Contact detection (binary)

| Model | Accuracy | F1 (macro) | F1 (positive) |
|---|---|---|---|
| XGBoost | 0.9894 | 0.9894 | 0.9891 |
| RandomForest | 0.9906 | 0.9906 | 0.9904 |
| CNN1D | 0.9825 | 0.9825 | 0.9819 |

**XGBoost confusion matrix** (rows=true, cols=pred):

```
          no    yes
   no   1630     13
  yes     21   1539
```

**RandomForest confusion matrix** (rows=true, cols=pred):

```
          no    yes
   no   1632     11
  yes     19   1541
```

**CNN1D confusion matrix** (rows=true, cols=pred):

```
          no    yes
   no   1624     19
  yes     37   1523
```

## 3. Slip detection (binary)

| Model | Accuracy | F1 (macro) | F1 (positive) |
|---|---|---|---|
| XGBoost | 0.9916 | 0.9834 | 0.9717 |
| RandomForest | 0.9844 | 0.9698 | 0.9489 |
| CNN1D | 0.9176 | 0.8614 | 0.7732 |

**XGBoost confusion matrix** (rows=true, cols=pred):

```
          no    yes
   no   2712      2
  yes     25    464
```

**RandomForest confusion matrix** (rows=true, cols=pred):

```
          no    yes
   no   2689     25
  yes     25    464
```

**CNN1D confusion matrix** (rows=true, cols=pred):

```
          no    yes
   no   2489    225
  yes     39    450
```

## Notes & caveats

- Data is SentrixSim v0.1 **relative/shape-only** physics: signal timing, noise, quantization and topology are datasheet/spec-true, but absolute force/field magnitudes are not. Scores measure separability of the simulated structure, **not** real-hardware performance.
- Event labels are clip-level; the pre-contact (idle/reach) portions of non-idle gestures resemble idle, which bounds achievable per-episode accuracy and is the main confusion source.
- Slip is rare and only present in the `slip` gesture after onset, so the slip task is highly imbalanced; read F1(positive) and the confusion matrix, not accuracy.
- Drift/noise realizations are the only cross-instance variation; these baselines will need re-fitting once calibrated hardware data replaces the relative scales (dataset v0.2).