#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


AA_COLUMNS = 20
LAG = 10
AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AA_ORDER)}

AA_GROUPS = [
    [0, 4],
    [3, 6],
    [13, 16],
    [7, 8],
    [9, 10],
    [11, 12],
    [14, 15],
    [1, 2],
    [5, 17],
    [18, 19],
    [0, 1, 2, 3],
]


def parse_pssm_ascii(pssm_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    rows: List[List[float]] = []
    residues: List[int] = []
    with pssm_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 22:
                continue
            if not parts[0].isdigit():
                continue
            try:
                vals = [float(x) for x in parts[2:22]]
            except ValueError:
                continue
            rows.append(vals)
            residue = parts[1].upper()
            residues.append(AA_TO_INDEX.get(residue, -1))
    if len(rows) == 0:
        raise ValueError(f"no valid PSSM rows in {pssm_path}")
    return np.asarray(rows, dtype=np.float32), np.asarray(residues, dtype=np.int32)


def sigmoid_norm(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def autocov_features(x: np.ndarray, lag: int) -> np.ndarray:
    length, dims = x.shape
    feats = []
    mean = x.mean(axis=0)
    centered = x - mean
    for d in range(dims):
        col = centered[:, d]
        for k in range(1, lag + 1):
            if length <= k:
                feats.append(0.0)
            else:
                feats.append(float(np.mean(col[:-k] * col[k:])))
    return np.asarray(feats, dtype=np.float32)


def build_rpssm_110(pssm: np.ndarray, lag: int = LAG) -> np.ndarray:
    p = sigmoid_norm(pssm)
    reduced_cols = []
    for group in AA_GROUPS:
        valid = [idx for idx in group if idx < p.shape[1]]
        reduced_cols.append(p[:, valid].mean(axis=1))
    reduced = np.stack(reduced_cols, axis=1)
    feats = autocov_features(reduced, lag=lag)
    if feats.shape[0] != 110:
        raise ValueError(f"RPSSM dim mismatch: {feats.shape[0]}")
    return feats


def build_pssm_ac_200(pssm: np.ndarray, lag: int = LAG) -> np.ndarray:
    p = sigmoid_norm(pssm)
    feats = autocov_features(p[:, :AA_COLUMNS], lag=lag)
    if feats.shape[0] != 200:
        raise ValueError(f"PSSM-AC dim mismatch: {feats.shape[0]}")
    return feats


def build_pssm_composition_400(pssm: np.ndarray, residue_ids: np.ndarray) -> np.ndarray:
    p = sigmoid_norm(pssm)[:, :AA_COLUMNS]
    comp = np.zeros((AA_COLUMNS, AA_COLUMNS), dtype=np.float32)
    counts = np.zeros((AA_COLUMNS,), dtype=np.float32)
    for pos in range(p.shape[0]):
        r = int(residue_ids[pos])
        if 0 <= r < AA_COLUMNS:
            comp[r] += p[pos]
            counts[r] += 1.0
    for r in range(AA_COLUMNS):
        if counts[r] > 0:
            comp[r] /= counts[r]
    feats = comp.reshape(-1)
    if feats.shape[0] != 400:
        raise ValueError(f"PSSM-composition dim mismatch: {feats.shape[0]}")
    return feats


def build_dpc_pssm_400(pssm: np.ndarray) -> np.ndarray:
    p = sigmoid_norm(pssm)[:, :AA_COLUMNS]
    if p.shape[0] <= 1:
        dpc = np.zeros((AA_COLUMNS, AA_COLUMNS), dtype=np.float32)
    else:
        dpc = np.matmul(p[:-1].T, p[1:]) / float(p.shape[0] - 1)
    feats = dpc.astype(np.float32).reshape(-1)
    if feats.shape[0] != 400:
        raise ValueError(f"DPC-PSSM dim mismatch: {feats.shape[0]}")
    return feats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract RPSSM(110), PSSM-AC(200), PSSM-composition(400), DPC-PSSM(400)."
    )
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--lag", type=int, default=LAG)
    args = parser.parse_args()

    work_root = Path(args.work_root)
    feat_dir = work_root / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    status_path = feat_dir / "feature_status.csv"

    manifest = pd.read_csv(args.manifest_csv)
    rows = []
    ok_count = 0
    for row in manifest.itertuples(index=False):
        pssm_path = Path(row.pssm_path)
        out_310_legacy = feat_dir / f"{row.sample_id}.npy"
        out_310 = feat_dir / f"{row.sample_id}_pssm310.npy"
        out_710 = feat_dir / f"{row.sample_id}_pssm710.npy"
        out_1110 = feat_dir / f"{row.sample_id}_pssm1110.npy"
        status = {
            "sample_id": row.sample_id,
            "pssm_ok": 0,
            "feature_ok": 0,
            "feat_310_ok": 0,
            "feat_710_ok": 0,
            "feat_1110_ok": 0,
            "retry_count": 0,
        }

        try:
            pssm, residue_ids = parse_pssm_ascii(pssm_path)
            rpssm = build_rpssm_110(pssm, lag=args.lag)
            pssmac = build_pssm_ac_200(pssm, lag=args.lag)
            pssm_comp = build_pssm_composition_400(pssm, residue_ids)
            dpc_pssm = build_dpc_pssm_400(pssm)

            feat_310 = np.concatenate([rpssm, pssmac], axis=0).astype(np.float32)
            feat_710 = np.concatenate([feat_310, pssm_comp], axis=0).astype(np.float32)
            feat_1110 = np.concatenate([feat_710, dpc_pssm], axis=0).astype(np.float32)
            if feat_310.shape[0] != 310:
                raise ValueError(f"expected 310 dims, got {feat_310.shape[0]}")
            if feat_710.shape[0] != 710:
                raise ValueError(f"expected 710 dims, got {feat_710.shape[0]}")
            if feat_1110.shape[0] != 1110:
                raise ValueError(f"expected 1110 dims, got {feat_1110.shape[0]}")

            np.save(out_310_legacy, feat_310)
            np.save(out_310, feat_310)
            np.save(out_710, feat_710)
            np.save(out_1110, feat_1110)
            status["pssm_ok"] = 1
            status["feature_ok"] = 1
            status["feat_310_ok"] = 1
            status["feat_710_ok"] = 1
            status["feat_1110_ok"] = 1
            ok_count += 1
        except Exception:
            pass

        rows.append(status)

    status_df = pd.DataFrame(rows)
    status_df.to_csv(status_path, index=False)
    print(f"Feature extraction done. ok={ok_count}/{len(status_df)}")
    print(f"Status table: {status_path}")


if __name__ == "__main__":
    main()
