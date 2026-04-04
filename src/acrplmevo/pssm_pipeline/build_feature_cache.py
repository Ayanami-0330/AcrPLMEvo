#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


VARIANT_TO_DIM = {
    "310": 310,
    "710": 710,
    "1110": 1110,
}


def _resolve_variant_path(features_dir: Path, sample_id: str, variant: str) -> Path:
    if variant == "310":
        modern = features_dir / f"{sample_id}_pssm310.npy"
        legacy = features_dir / f"{sample_id}.npy"
        return modern if modern.exists() else legacy
    return features_dir / f"{sample_id}_pssm{variant}.npy"


def _write_feature_schema(features_dir: Path) -> None:
    rows = [
        {"start": 0, "end": 109, "dim": 110, "block": "RPSSM", "description": "11 AA groups x lag(10) autocovariance"},
        {"start": 110, "end": 309, "dim": 200, "block": "PSSM-AC", "description": "20 AA columns x lag(10) autocovariance"},
        {"start": 310, "end": 709, "dim": 400, "block": "PSSM-composition", "description": "20x20 composition matrix (residue-conditioned PSSM mean)"},
        {"start": 710, "end": 1109, "dim": 400, "block": "DPC-PSSM", "description": "20x20 adjacent-position coupling matrix"},
    ]
    schema = pd.DataFrame(rows)
    schema.to_csv(features_dir / "pssm_feature_schema.csv", index=False)


def _build_single_variant(manifest: pd.DataFrame, features_dir: Path, variant: str) -> None:
    dim = VARIANT_TO_DIM[variant]
    cache_parquet = features_dir / f"pssm_features_{variant}.parquet"
    cache_npy = features_dir / f"pssm_features_{variant}.npy"
    cache_ids = features_dir / f"pssm_feature_ids_{variant}.csv"

    mat = []
    ids = []
    for row in manifest.itertuples(index=False):
        feat_path = _resolve_variant_path(features_dir, row.sample_id, variant)
        if not feat_path.exists():
            continue
        feat = np.load(feat_path)
        if feat.shape[0] != dim:
            continue
        ids.append(row.sample_id)
        mat.append(feat.astype(np.float32))

    if len(mat) == 0:
        raise RuntimeError(f"No valid feature vectors found for variant {variant}.")

    mat_np = np.stack(mat, axis=0)
    cols = [f"feat_{i:04d}" for i in range(mat_np.shape[1])]
    df = pd.DataFrame(mat_np, columns=cols)
    df.insert(0, "sample_id", ids)

    try:
        df.to_parquet(cache_parquet, index=False)
        print(f"[{variant}] Wrote parquet: {cache_parquet}")
    except Exception as exc:
        fallback_csv = str(cache_parquet).replace(".parquet", ".csv")
        df.to_csv(fallback_csv, index=False)
        print(f"[{variant}] Parquet unavailable ({exc}); wrote csv fallback: {fallback_csv}")

    np.save(cache_npy, mat_np)
    pd.DataFrame({"sample_id": ids}).to_csv(cache_ids, index=False)
    print(f"[{variant}] Wrote npy: {cache_npy}")
    print(f"[{variant}] Wrote ids: {cache_ids}")
    print(f"[{variant}] Shape: {mat_np.shape}")

    if variant == "310":
        pd.DataFrame({"sample_id": ids}).to_csv(features_dir / "pssm_feature_ids.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multi-variant feature cache from sample npy files.")
    parser.add_argument("--manifest-csv", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument(
        "--variants",
        default="310,710,1110",
        help="Comma separated variants to build. Supported: 310,710,1110",
    )
    args = parser.parse_args()

    work_root = Path(args.work_root)
    features_dir = work_root / "features"
    manifest = pd.read_csv(args.manifest_csv)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    invalid = [v for v in variants if v not in VARIANT_TO_DIM]
    if invalid:
        raise ValueError(f"Unsupported variants: {invalid}; choose from {sorted(VARIANT_TO_DIM)}")

    for variant in variants:
        _build_single_variant(manifest, features_dir, variant)
    _write_feature_schema(features_dir)
    print(f"Wrote feature schema: {features_dir / 'pssm_feature_schema.csv'}")


if __name__ == "__main__":
    main()
