#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


def ensure_sample_ids(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    out = df.copy()
    out["sample_id"] = [f"{split_name}_{i:06d}" for i in range(len(out))]
    return out


def write_fasta(path: Path, sample_id: str, seq: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(f">{sample_id}\n")
        f.write(f"{seq}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare FASTA files with stable sample ids.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--work-root", required=True)
    args = parser.parse_args()

    work_root = Path(args.work_root)
    fasta_root = work_root / "fasta"
    manifest_path = work_root / "sample_manifest.csv"

    train_df = pd.read_csv(args.train_csv).dropna().drop_duplicates().reset_index(drop=True)
    test_df = pd.read_csv(args.test_csv).dropna().drop_duplicates().reset_index(drop=True)
    train_df = ensure_sample_ids(train_df, "train")
    test_df = ensure_sample_ids(test_df, "test")
    all_df = pd.concat([train_df.assign(split="train"), test_df.assign(split="test")], ignore_index=True)

    rows = []
    for row in all_df.itertuples(index=False):
        fasta_path = fasta_root / f"{row.sample_id}.fa"
        pssm_path = work_root / "pssm" / f"{row.sample_id}.pssm"
        write_fasta(fasta_path, row.sample_id, row.seq)
        rows.append(
            {
                "sample_id": row.sample_id,
                "split": row.split,
                "label": int(row.label),
                "seq": row.seq,
                "fasta_path": str(fasta_path),
                "pssm_path": str(pssm_path),
            }
        )

    out_df = pd.DataFrame(rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(manifest_path, index=False)
    print(f"Wrote manifest: {manifest_path}")
    print(f"Total samples: {len(out_df)}")


if __name__ == "__main__":
    main()
