from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             matthews_corrcoef, roc_auc_score)


def ensure_sample_ids(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    out = df.copy()
    out["sample_id"] = [f"{split_name}_{i:06d}" for i in range(len(out))]
    return out


def load_anticrispr_with_ids(
    benchmarks_dir: str,
    benchmark_name: str = "anticrispr_binary",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_path = f"{benchmarks_dir}/{benchmark_name}.train.csv"
    test_path = f"{benchmarks_dir}/{benchmark_name}.test.csv"
    train_df = pd.read_csv(train_path).dropna().drop_duplicates().reset_index(drop=True)
    test_df = pd.read_csv(test_path).dropna().drop_duplicates().reset_index(drop=True)
    return ensure_sample_ids(train_df, "train"), ensure_sample_ids(test_df, "test")


def load_feature_cache(feature_cache_path: str) -> Tuple[pd.DataFrame, List[str]]:
    if feature_cache_path.endswith(".parquet"):
        feat_df = pd.read_parquet(feature_cache_path)
    elif feature_cache_path.endswith(".csv"):
        feat_df = pd.read_csv(feature_cache_path)
    else:
        raise ValueError("Feature cache must be parquet or csv.")

    if "sample_id" not in feat_df.columns:
        raise ValueError("feature cache missing sample_id column.")

    feature_cols = [c for c in feat_df.columns if c.startswith("feat_")]
    if not feature_cols:
        raise ValueError("feature cache missing feat_* columns.")
    return feat_df, feature_cols


def attach_pssm_features(
    seq_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    feature_cols: Sequence[str],
    fill_value: float = 0.0,
) -> pd.DataFrame:
    out = seq_df.merge(feature_df[["sample_id", *feature_cols]], on="sample_id", how="left")
    out.loc[:, feature_cols] = out.loc[:, feature_cols].fillna(fill_value)
    return out


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        m = ids == b
        if np.any(m):
            conf = float(np.mean(y_prob[m]))
            acc = float(np.mean(y_true[m]))
            ece += (np.sum(m) / n) * abs(acc - conf)
    return float(ece)


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_cls = (y_prob >= threshold).astype(int)
    return {
        "AUC": float(roc_auc_score(y_true, y_prob)),
        "AUPRC": float(average_precision_score(y_true, y_prob)),
        "F1": float(f1_score(y_true, y_cls)),
        "MCC": float(matthews_corrcoef(y_true, y_cls)),
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "ECE": expected_calibration_error(y_true, y_prob, n_bins=10),
        "Threshold": float(threshold),
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, grid: Optional[Iterable[float]] = None) -> float:
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    best_thr = 0.5
    best_f1 = -1.0
    for thr in grid:
        cur = f1_score(y_true, (y_prob >= thr).astype(int))
        if cur > best_f1:
            best_f1 = cur
            best_thr = float(thr)
    return best_thr
