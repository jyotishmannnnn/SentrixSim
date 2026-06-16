# Sentrix Dataset v0.1 - Baseline Benchmark Report

Three tasks x three model families, trained on `dataset_v0.1`. Split is **episode-level** (stratified by event), so no window leaks between train and test.

## Setup

- Channels: 75 (63 BMM350 axes + 9 accel + 3 temp)
- Event task: per-episode, 9-class; 720 train / 180 test episodes; CNN on resampled [75, 256] series; trees on 300 features (per-channel mean/std/min/max).
- Contact/slip tasks: per-window (160 samples = 100 ms @ 1600 Hz); 12637 train / 3203 test windows.
- Test positive fraction - contact: 0.765, slip: 0.051.
- Trees: XGBoost (hist, 300 trees) & RandomForest (300, balanced for binary). CNN: Conv1d(7)->BN->ReLU->Pool->Conv1d(5)->BN->ReLU->GAP->FC, Adam 1e-3, CE (class-weighted for binary).

## 1. Event classification (9-class)

| Model | Accuracy | F1 (macro) |
|---|---|---|
| XGBoost | 0.9944 | 0.9944 |
| RandomForest | 1.0000 | 1.0000 |
| CNN1D | 0.8278 | 0.8292 |

**XGBoost confusion matrix** (rows=true, cols=pred):

```
        idle    tap  press   hold  shear   slip release  pinch  grasp
 idle     20      0      0      0      0      0      0      0      0
  tap      0     20      0      0      0      0      0      0      0
press      0      0     20      0      0      0      0      0      0
 hold      0      0      0     20      0      0      0      0      0
shear      0      0      0      0     20      0      0      0      0
 slip      0      0      0      0      0     20      0      0      0
release      0      0      1      0      0      0     19      0      0
pinch      0      0      0      0      0      0      0     20      0
grasp      0      0      0      0      0      0      0      0     20
```

**RandomForest confusion matrix** (rows=true, cols=pred):

```
        idle    tap  press   hold  shear   slip release  pinch  grasp
 idle     20      0      0      0      0      0      0      0      0
  tap      0     20      0      0      0      0      0      0      0
press      0      0     20      0      0      0      0      0      0
 hold      0      0      0     20      0      0      0      0      0
shear      0      0      0      0     20      0      0      0      0
 slip      0      0      0      0      0     20      0      0      0
release      0      0      0      0      0      0     20      0      0
pinch      0      0      0      0      0      0      0     20      0
grasp      0      0      0      0      0      0      0      0     20
```

**CNN1D confusion matrix** (rows=true, cols=pred):

```
        idle    tap  press   hold  shear   slip release  pinch  grasp
 idle     12      8      0      0      0      0      0      0      0
  tap      6     14      0      0      0      0      0      0      0
press      0      0     14      3      0      0      3      0      0
 hold      0      0      7     13      0      0      0      0      0
shear      0      0      0      0     20      0      0      0      0
 slip      0      0      0      0      0     20      0      0      0
release      0      0      4      0      0      0     16      0      0
pinch      0      0      0      0      0      0      0     20      0
grasp      0      0      0      0      0      0      0      0     20
```

## 2. Contact detection (binary)

| Model | Accuracy | F1 (macro) | F1 (positive) |
|---|---|---|---|
| XGBoost | 0.9997 | 0.9996 | 0.9998 |
| RandomForest | 0.9988 | 0.9983 | 0.9992 |
| CNN1D | 0.9966 | 0.9952 | 0.9978 |

**XGBoost confusion matrix** (rows=true, cols=pred):

```
          no    yes
   no    754      0
  yes      1   2448
```

**RandomForest confusion matrix** (rows=true, cols=pred):

```
          no    yes
   no    754      0
  yes      4   2445
```

**CNN1D confusion matrix** (rows=true, cols=pred):

```
          no    yes
   no    752      2
  yes      9   2440
```

## 3. Slip detection (binary)

| Model | Accuracy | F1 (macro) | F1 (positive) |
|---|---|---|---|
| XGBoost | 1.0000 | 1.0000 | 1.0000 |
| RandomForest | 1.0000 | 1.0000 | 1.0000 |
| CNN1D | 0.9975 | 0.9874 | 0.9760 |

**XGBoost confusion matrix** (rows=true, cols=pred):

```
          no    yes
   no   3039      0
  yes      0    164
```

**RandomForest confusion matrix** (rows=true, cols=pred):

```
          no    yes
   no   3039      0
  yes      0    164
```

**CNN1D confusion matrix** (rows=true, cols=pred):

```
          no    yes
   no   3032      7
  yes      1    163
```

## Notes & caveats

- Data is SentrixSim v0.1 **relative/shape-only** physics: signal timing, noise, quantization and topology are datasheet/spec-true, but absolute force/field magnitudes are not. Scores measure separability of the simulated structure, **not** real-hardware performance.
- Event labels are clip-level; the pre-contact (idle/reach) portions of non-idle gestures resemble idle, which bounds achievable per-episode accuracy and is the main confusion source.
- Slip is rare and only present in the `slip` gesture after onset, so the slip task is highly imbalanced; read F1(positive) and the confusion matrix, not accuracy.
- Drift/noise realizations are the only cross-instance variation; these baselines will need re-fitting once calibrated hardware data replaces the relative scales (dataset v0.2).