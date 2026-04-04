#!/usr/bin/env python3
"""
Supplemental frozen-adapter re-evaluation driver.

补充冻结适配器复评脚本。

Purpose / 作用:
- EN: This script is NOT the main 10-group protocol runner.
    中文：本脚本不是主10组协议入口。
- EN: It is used after main.py run-10 has produced adapter checkpoints.
    中文：它用于 main.py run-10 已产出 adapter checkpoint 之后。
- EN: It reloads existing adapters in frozen mode (is_trainable=False), trains
    only a small head, and writes an independent supplemental result set.
    中文：它以冻结方式（is_trainable=False）加载已有 adapter，仅训练小头，
    并输出独立的补充结果集。

Typical use case / 常见用途:
- EN: Re-check same-variant frozen adapter performance without re-running
    adapter fine-tuning from scratch.
    中文：在不重新做 adapter 微调的前提下，复查同配变体的冻结评估表现。
"""

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from peft import PeftModel
from sklearn.decomposition import PCA
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import sys

SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
EXP_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
LEGACY_PROJECT_ROOT = WORKSPACE_ROOT / "protein_bert"
for path in (SCRIPTS_ROOT, EXP_ROOT / "src", LEGACY_PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from main import (  # type: ignore
    BACKBONE_SPECS,
    SequenceBinaryClassifier,
    SequenceDataset,
    TrainConfig,
    get_autocast_dtype,
    get_device,
    load_split_data,
    load_tokenizer_and_model,
    make_collator,
)
from acrplmevo.pssm_fusion import evaluate_binary, find_best_threshold

DEFAULT_MODELS = ["ankh", "esm2", "protbert", "prott5"]
DEFAULT_ADAPTER_TYPES = ["lora", "dora"]
DEFAULT_SEEDS = [11, 22, 33, 44, 55]


@dataclass
class FrozenRunConfig:
    adapter_type: str
    model_name: str
    seed: int
    variant: str = "lm_only"
    pca_dim: int = 128
    head_epochs: int = 40
    head_lr: float = 1e-3
    head_batch_size: int = 64
    patience: int = 3
    dropout: float = 0.3


class SmallHead(nn.Module):
    def __init__(self, in_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_csv_list(value: str, cast=str) -> List:
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_features(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fit_pca_128(x_train: np.ndarray, x_valid: np.ndarray, x_test: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    x_train = sanitize_features(x_train)
    x_valid = sanitize_features(x_valid)
    x_test = sanitize_features(x_test)

    n_components = min(128, x_train.shape[1], x_train.shape[0])
    pca = PCA(n_components=n_components, random_state=seed)
    tr = pca.fit_transform(x_train)
    va = pca.transform(x_valid)
    te = pca.transform(x_test)

    explained = float(np.sum(pca.explained_variance_ratio_))

    if n_components < 128:
        pad = 128 - n_components
        tr = np.pad(tr, ((0, 0), (0, pad)))
        va = np.pad(va, ((0, 0), (0, pad)))
        te = np.pad(te, ((0, 0), (0, pad)))

    info = {"pca_n_components": int(n_components), "pca_explained_variance_sum": explained}
    return tr.astype(np.float32), va.astype(np.float32), te.astype(np.float32), info


def build_loader(df: pd.DataFrame, tokenizer, spec, batch_size: int) -> DataLoader:
    collate_fn = make_collator(tokenizer, spec.seq_mode, spec.max_length)
    dataset = SequenceDataset(df, [])
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def extract_features(df: pd.DataFrame, tokenizer, backbone: nn.Module, spec, device: torch.device) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    loader = build_loader(df, tokenizer, spec, spec.batch_size)
    autocast_dtype = get_autocast_dtype(device)

    feats: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    sample_ids: List[str] = []

    backbone.eval()
    for raw_batch in loader:
        ids = list(raw_batch["sample_ids"])
        y = raw_batch["labels"].numpy().astype(np.int32)

        kwargs: Dict[str, torch.Tensor] = {
            "input_ids": raw_batch["input_ids"].to(device),
            "attention_mask": raw_batch["attention_mask"].to(device),
            "return_dict": True,
        }
        if "token_type_ids" in raw_batch:
            kwargs["token_type_ids"] = raw_batch["token_type_ids"].to(device)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                out = backbone(**kwargs)
            pooled = SequenceBinaryClassifier.masked_mean(out.last_hidden_state, kwargs["attention_mask"]).detach().cpu().float().numpy()

        sample_ids.extend(ids)
        labels.append(y)
        feats.append(pooled)

    return sanitize_features(np.concatenate(feats, axis=0)), np.concatenate(labels, axis=0), sample_ids


def train_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    cfg: FrozenRunConfig,
    device: torch.device,
) -> Tuple[SmallHead, List[Dict[str, float]]]:
    model = SmallHead(x_train.shape[1], cfg.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.head_lr)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32)))
    valid_ds = TensorDataset(torch.from_numpy(x_valid), torch.from_numpy(y_valid.astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=cfg.head_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.head_batch_size, shuffle=False)

    history: List[Dict[str, float]] = []
    best_state = None
    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(1, cfg.head_epochs + 1):
        model.train()
        train_losses: List[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        valid_losses: List[float] = []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                loss = criterion(model(xb), yb)
                valid_losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        valid_loss = float(np.mean(valid_losses)) if valid_losses else 0.0
        history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})

        if valid_loss < best_val:
            best_val = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict_prob(model: SmallHead, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    ds = TensorDataset(torch.from_numpy(x.astype(np.float32)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            p = torch.sigmoid(model(xb)).detach().cpu().numpy()
            probs.append(p)

    return np.concatenate(probs).astype(np.float32)


def state_file(output_root: Path, adapter_type: str, model_name: str, seed: int, variant: str) -> Path:
    return output_root / "_state" / f"{adapter_type}__{model_name}__seed_{seed}__{variant}.done"


def run_dir(output_root: Path, adapter_type: str, model_name: str, seed: int, variant: str) -> Path:
    return output_root / adapter_type / model_name / variant / f"seed_{seed}"


def update_results_csv(results_csv: Path, row: Dict[str, object]) -> None:
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        mask = ~(
            (df["adapter_type"] == row["adapter_type"])
            & (df["model"] == row["model"])
            & (df["variant"] == row["variant"])
            & (df["seed"] == row["seed"])
        )
        df = df.loc[mask].copy()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df = df.sort_values(["adapter_type", "model", "variant", "seed"]).reset_index(drop=True)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_csv, index=False)


def run_one_combo(
    cfg: FrozenRunConfig,
    adapter_root: Path,
    output_root: Path,
    results_csv: Path,
    resume: bool,
    dry_run: bool,
    max_train_samples: int | None,
    max_valid_samples: int | None,
    max_test_samples: int | None,
) -> None:
    marker = state_file(output_root, cfg.adapter_type, cfg.model_name, cfg.seed, cfg.variant)
    if resume and marker.exists():
        print(f"skip done: {marker.name}")
        return

    run_out = run_dir(output_root, cfg.adapter_type, cfg.model_name, cfg.seed, cfg.variant)
    if dry_run:
        print(f"[dry-run] {cfg.adapter_type} {cfg.model_name} seed={cfg.seed} -> {run_out}")
        return

    adapter_dir = adapter_root / cfg.adapter_type / cfg.model_name / cfg.variant / f"seed_{cfg.seed}" / "adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"missing adapter dir: {adapter_dir}")

    set_seed(cfg.seed)
    device = get_device()
    spec = BACKBONE_SPECS[cfg.model_name]

    train_cfg = TrainConfig(
        seed=cfg.seed,
        model_name=cfg.model_name,
        variant=cfg.variant,
        batch_size=spec.batch_size,
        max_length=spec.max_length,
        adapter_type=cfg.adapter_type,
        max_train_samples=max_train_samples,
        max_valid_samples=max_valid_samples,
        max_test_samples=max_test_samples,
    )

    train_df, valid_df, test_df, feature_cols = load_split_data(cfg.variant, cfg.seed, train_cfg)

    model_id, tokenizer, backbone, _, _ = load_tokenizer_and_model(spec)
    backbone = PeftModel.from_pretrained(backbone, str(adapter_dir), is_trainable=False)
    backbone = backbone.to(device)
    backbone.eval()

    x_train_raw, y_train, train_ids = extract_features(train_df, tokenizer, backbone, spec, device)
    x_valid_raw, y_valid, valid_ids = extract_features(valid_df, tokenizer, backbone, spec, device)
    x_test_raw, y_test, test_ids = extract_features(test_df, tokenizer, backbone, spec, device)

    if feature_cols:
        # lm_pssm: concatenate PSSM features (already StandardScaler'd in load_split_data)
        pssm_train = sanitize_features(train_df[feature_cols].values.astype(np.float32))
        pssm_valid = sanitize_features(valid_df[feature_cols].values.astype(np.float32))
        pssm_test = sanitize_features(test_df[feature_cols].values.astype(np.float32))
        x_train_raw = np.concatenate([x_train_raw, pssm_train], axis=1)
        x_valid_raw = np.concatenate([x_valid_raw, pssm_valid], axis=1)
        x_test_raw = np.concatenate([x_test_raw, pssm_test], axis=1)

    x_train_128, x_valid_128, x_test_128, pca_info = fit_pca_128(x_train_raw, x_valid_raw, x_test_raw, cfg.seed)
    head, history = train_head(x_train_128, y_train, x_valid_128, y_valid, cfg, device)

    valid_prob = predict_prob(head, x_valid_128, cfg.head_batch_size, device)
    test_prob = predict_prob(head, x_test_128, cfg.head_batch_size, device)
    best_thr = float(find_best_threshold(y_valid, valid_prob))
    test_pred = (test_prob >= best_thr).astype(np.int32)
    test_acc = float(np.mean(test_pred == y_test.astype(np.int32)))

    valid_metrics = evaluate_binary(y_valid, valid_prob, threshold=best_thr)
    test_metrics = evaluate_binary(y_test, test_prob, threshold=best_thr)

    run_out.mkdir(parents=True, exist_ok=True)

    pred_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "sample_id": valid_ids,
                    "split": "valid",
                    "y_true": y_valid.astype(int),
                    "y_prob": valid_prob.astype(float),
                    "y_pred": (valid_prob >= best_thr).astype(int),
                    "threshold": best_thr,
                    "seed": cfg.seed,
                    "model": cfg.model_name,
                    "variant": cfg.variant,
                    "adapter_type": cfg.adapter_type,
                }
            ),
            pd.DataFrame(
                {
                    "sample_id": test_ids,
                    "split": "test",
                    "y_true": y_test.astype(int),
                    "y_prob": test_prob.astype(float),
                    "y_pred": (test_prob >= best_thr).astype(int),
                    "threshold": best_thr,
                    "seed": cfg.seed,
                    "model": cfg.model_name,
                    "variant": cfg.variant,
                    "adapter_type": cfg.adapter_type,
                }
            ),
        ],
        ignore_index=True,
    )
    pred_df.to_csv(run_out / "predictions.csv", index=False)
    pd.DataFrame(history).to_csv(run_out / "training_history.csv", index=False)

    metrics_payload = {
        "adapter_type": cfg.adapter_type,
        "model": cfg.model_name,
        "variant": cfg.variant,
        "seed": cfg.seed,
        "hf_model_id": model_id,
        "pca_info": pca_info,
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "threshold": best_thr,
    }
    (run_out / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    config_payload = asdict(cfg)
    config_payload.update(
        {
            "adapter_dir": str(adapter_dir),
            "output_dir": str(run_out),
            "resume": resume,
            "max_train_samples": max_train_samples,
            "max_valid_samples": max_valid_samples,
            "max_test_samples": max_test_samples,
        }
    )
    (run_out / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    row = {
        "adapter_type": cfg.adapter_type,
        "model": cfg.model_name,
        "variant": cfg.variant,
        "seed": cfg.seed,
        "AUC": float(test_metrics["AUC"]),
        "AUPRC": float(test_metrics["AUPRC"]),
        "ACC": test_acc,
        "F1": float(test_metrics["F1"]),
        "MCC": float(test_metrics["MCC"]),
        "Threshold": best_thr,
        "metrics_path": str(run_out / "metrics.json"),
        "predictions_path": str(run_out / "predictions.csv"),
        "config_path": str(run_out / "config.json"),
    }
    update_results_csv(results_csv, row)

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("done\n", encoding="utf-8")
    print(f"done: {cfg.adapter_type} {cfg.model_name} seed={cfg.seed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Supplemental frozen-adapter re-evaluation driver "
            "(reload existing adapters, train only PCA128+small head)."
        )
    )
    parser.add_argument("--adapter-types", default=",".join(DEFAULT_ADAPTER_TYPES))
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--variants", default="lm_only,lm_pssm")
    parser.add_argument("--adapter-root", default=str(EXP_ROOT / "results" / "runs"))
    parser.add_argument("--output-root", default=str(EXP_ROOT / "results" / "runs_frozen"))
    parser.add_argument("--results-csv", default=str(EXP_ROOT / "results" / "experiments_frozen.csv"))
    parser.add_argument("--head-epochs", type=int, default=40)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--head-batch-size", type=int, default=64)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-valid-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_types = parse_csv_list(args.adapter_types, str)
    models = parse_csv_list(args.models, str)
    seeds = parse_csv_list(args.seeds, int)

    variants = parse_csv_list(args.variants, str)
    for model in models:
        if model not in BACKBONE_SPECS:
            raise ValueError(f"unsupported model: {model}")
    for adapter in adapter_types:
        if adapter not in {"lora", "dora"}:
            raise ValueError(f"unsupported adapter type: {adapter}")
    for variant in variants:
        if variant not in {"lm_only", "lm_pssm"}:
            raise ValueError(f"unsupported variant: {variant}")

    adapter_root = Path(args.adapter_root)
    output_root = Path(args.output_root)
    results_csv = Path(args.results_csv)

    for adapter_type in adapter_types:
        for model_name in models:
            for seed in seeds:
                for variant in variants:
                    cfg = FrozenRunConfig(
                        adapter_type=adapter_type,
                        model_name=model_name,
                        seed=seed,
                        variant=variant,
                        head_epochs=args.head_epochs,
                        head_lr=args.head_lr,
                        head_batch_size=args.head_batch_size,
                        patience=args.patience,
                    )
                    run_one_combo(
                        cfg=cfg,
                        adapter_root=adapter_root,
                        output_root=output_root,
                        results_csv=results_csv,
                        resume=args.resume,
                        dry_run=args.dry_run,
                        max_train_samples=args.max_train_samples,
                        max_valid_samples=args.max_valid_samples,
                        max_test_samples=args.max_test_samples,
                    )


if __name__ == "__main__":
    main()
