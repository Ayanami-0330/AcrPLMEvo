#!/usr/bin/env python3
"""
Main benchmark entry for Anti-CRISPR LLM experiments.

Anti-CRISPR 大模型主实验统一入口脚本。

Role of this script / 脚本职责
EN:
- This is the primary protocol runner (A): run the full reviewer-facing 10-group
    experiment from scratch.
- It covers end-to-end steps: fine-tune adapter (LoRA/DoRA) -> save adapter ->
    reload adapter for frozen evaluation cells -> train small head -> export results.
中文：
- 本脚本是主协议执行器（A）：从零开始完整执行审稿10组实验。
- 完整流程包含：适配器微调（LoRA/DoRA）-> 保存 adapter -> 在冻结评估单元
    复用 adapter -> 训练小头 -> 导出结果。

Scope boundary / 边界说明
EN: Supplemental re-evaluation with existing adapters is handled by
        scripts/frozen_baseline/run_supplemental_frozen_eval.py (not this file).
中文：基于已有 adapter 的补充复评由
        scripts/frozen_baseline/run_supplemental_frozen_eval.py 负责（不在本脚本内）。

Reviewer-facing protocol / 审稿人可读实验协议
EN: This script implements a complete 5x2 design table (10 groups):
    5 strategy rows (S1-S5) x 2 prediction-time input columns (lm_only/lm_pssm).
中文：本脚本完整实现 5x2 设计表（10组）：
    5条策略行（S1-S5）x 2条预测输入列（lm_only/lm_pssm）。

Prediction-time input columns / 预测阶段输入列
- external=lm_only:
    EN: use LM representation only.
    中文：仅使用大模型序列表征。
- external=lm_pssm:
    EN: LM representation plus external PSSM before PCA/fusion.
    中文：在PCA/融合前拼接外部PSSM特征。

Strategy rows / 策略行定义
- S1 native:
    EN: frozen backbone, no LoRA/DoRA adapter.
    中文：主干冻结，不使用LoRA/DoRA适配器。
- S2 tuned_lm_only + LoRA:
    EN: adapter_variant=lm_only, adapter_type=lora.
    中文：adapter_variant=lm_only，adapter_type=lora。
- S3 tuned_lm_only + DoRA:
    EN: adapter_variant=lm_only，adapter_type=dora.
    中文：adapter_variant=lm_only，adapter_type=dora。
- S4 tuned_lm_pssm + LoRA:
    EN: adapter_variant=lm_pssm，adapter_type=lora.
    中文：adapter_variant=lm_pssm，adapter_type=lora。
- S5 tuned_lm_pssm + DoRA:
    EN: adapter_variant=lm_pssm，adapter_type=dora.
    中文：adapter_variant=lm_pssm，adapter_type=dora。

Command mapping / 命令映射
- run-10:
    EN: execute the complete 10-group protocol (S1-S5 x two columns) and rebuild unified summaries.
    中文：执行完整10组协议（S1-S5 x 两列）并自动重建统一汇总表。
- run:
    EN: optional single-cell adapter training entry for debugging.
    中文：可选的单单元适配器训练入口（调试用途）。
    EN example (LoRA):
        python main.py run --model esm2 --variant lm_only --seed 11 --adapter-type lora --save-adapter
    中文示例（LoRA）：
        python main.py run --model esm2 --variant lm_only --seed 11 --adapter-type lora --save-adapter
    EN example (DoRA):
        python main.py run --model esm2 --variant lm_pssm --seed 11 --adapter-type dora --save-adapter
    中文示例（DoRA）：
        python main.py run --model esm2 --variant lm_pssm --seed 11 --adapter-type dora --save-adapter
- summary:
    EN: rebuild summary tables from existing run outputs.
    中文：根据已有运行结果重建汇总表。

10-group index / 10组编号
- G01: S1 native + external=lm_only
- G02: S1 native + external=lm_pssm
- G03: S2 tuned_lm_only LoRA + external=lm_only
- G04: S2 tuned_lm_only LoRA + external=lm_pssm
- G05: S3 tuned_lm_only DoRA + external=lm_only
- G06: S3 tuned_lm_only DoRA + external=lm_pssm
- G07: S4 tuned_lm_pssm LoRA + external=lm_pssm
- G08: S4 tuned_lm_pssm LoRA + external=lm_only
- G09: S5 tuned_lm_pssm DoRA + external=lm_pssm
- G10: S5 tuned_lm_pssm DoRA + external=lm_only
"""

import argparse
import copy
import json
import os
import random
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_HF_CACHE = Path(
    os.environ.get("ACRPLMEVO_HF_CACHE_DIR")
    or os.environ.get("HF_HOME")
    or str((PROJECT_DIR / "cache" / "hf_cache").resolve())
)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(DEFAULT_HF_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DEFAULT_HF_CACHE))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
offline_flag = os.environ.get("ACRPLMEVO_OFFLINE", "0").strip().lower()
if offline_flag in {"1", "true", "yes", "on"}:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
else:
    os.environ.setdefault("HF_HUB_OFFLINE", "0")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import (AutoModel, AutoTokenizer, BertModel,
                          BertTokenizer, EsmModel, T5EncoderModel)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
LEGACY_PROJECT_DIR = WORKSPACE_ROOT / "protein_bert"
for path in (PROJECT_DIR, PROJECT_DIR / "src", LEGACY_PROJECT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from acrplmevo.backbones import BACKBONE_SPECS, BackboneSpec
from acrplmevo.pssm_fusion import (attach_pssm_features,
                                   expected_calibration_error,
                                   find_best_threshold,
                                   load_anticrispr_with_ids,
                                   load_feature_cache)

LLM_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_DIR / "results"


def resolve_benchmarks_dir() -> Path:
    candidates = [
        PROJECT_DIR / "data" / "anticrispr_benchmarks",
        PROJECT_DIR / "anticrispr_benchmarks",
        LEGACY_PROJECT_DIR / "anticrispr_benchmarks",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


BENCHMARKS_DIR = resolve_benchmarks_dir()
HF_CACHE_DIR = Path(
    os.environ.get("ACRPLMEVO_HF_CACHE_DIR")
    or os.environ.get("HF_HOME")
    or str(DEFAULT_HF_CACHE)
)
PSSM_WORK_ROOT = Path(os.environ.get("PSSM_WORK_ROOT", str(PROJECT_DIR / "data" / "pssm_work")))
PSSM_FEATURE_DIR = PSSM_WORK_ROOT / "features"
DEFAULT_PSSM_VARIANT = "1110"
DEFAULT_SEEDS = [11, 22, 33, 44, 55]
EXPERIMENTS_CSV = RESULTS_ROOT / "experiments.csv"
SUMMARY_CSV = RESULTS_ROOT / "summary_by_model_variant.csv"
DELTA_CSV = RESULTS_ROOT / "delta_vs_lm_only.csv"
EXPERIMENTS_FROZEN_CSV = RESULTS_ROOT / "experiments_frozen.csv"
EXPERIMENTS_FROZEN_CROSS_CSV = RESULTS_ROOT / "experiments_frozen_cross_variant.csv"
EXPERIMENTS_FROZEN_NO_LORA_CSV = RESULTS_ROOT / "experiments_frozen_no_lora.csv"
SUMMARY_10GROUP_RUNS_CSV = RESULTS_ROOT / "summary_10group_runs.csv"
SUMMARY_10GROUP_BY_MODEL_CSV = RESULTS_ROOT / "summary_10group_by_model.csv"
SIX_CATEGORY_DIR = RESULTS_ROOT / "plots" / "6category"
SIX_CATEGORY_MEAN_CSV = SIX_CATEGORY_DIR / "six_category_mean_std_by_model.csv"
SIX_CATEGORY_SEEDMEAN_CSV = RESULTS_ROOT / "6categories_seedmean_auc_auprc.csv"
SIX_CATEGORY_BEST_CSV = RESULTS_ROOT / "6categories_best_single_seed_by_auc_then_auprc.csv"


@dataclass
class TrainConfig:
    seed: int
    model_name: str
    variant: str
    batch_size: int
    max_length: int
    adapter_type: str = "lora"
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    epochs: int = 8
    patience: int = 3
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_bias: str = "none"
    classifier_dropout: float = 0.2
    fusion_hidden_dim: int = 256
    pssm_hidden_dim: int = 256
    max_train_samples: Optional[int] = None
    max_valid_samples: Optional[int] = None
    max_test_samples: Optional[int] = None
    limit_train_batches: Optional[int] = None
    limit_eval_batches: Optional[int] = None


@dataclass
class CrossFrozenRunConfig:
    adapter_type: str
    model_name: str
    seed: int
    adapter_variant: str
    feature_variant: str
    pca_dim: int = 128
    head_epochs: int = 40
    head_lr: float = 1e-3
    head_batch_size: int = 64
    patience: int = 3
    dropout: float = 0.3
    max_train_samples: Optional[int] = None
    max_valid_samples: Optional[int] = None
    max_test_samples: Optional[int] = None


@dataclass
class NativeFrozenRunConfig:
    model_name: str
    seed: int
    variant: str
    pca_dim: int = 128
    head_epochs: int = 40
    head_lr: float = 1e-3
    head_batch_size: int = 64
    patience: int = 3
    dropout: float = 0.3
    max_train_samples: Optional[int] = None
    max_valid_samples: Optional[int] = None
    max_test_samples: Optional[int] = None


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


class SequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: Sequence[str]):
        self.sample_ids = df["sample_id"].astype(str).tolist()
        self.seqs = df["seq"].astype(str).tolist()
        self.labels = df["label"].astype(int).to_numpy(dtype=np.int64)
        self.has_pssm = len(feature_cols) > 0
        if self.has_pssm:
            self.pssm = df.loc[:, feature_cols].to_numpy(dtype=np.float32)
        else:
            self.pssm = None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        item = {
            "sample_id": self.sample_ids[idx],
            "seq": self.seqs[idx],
            "label": int(self.labels[idx]),
        }
        if self.pssm is not None:
            item["pssm"] = self.pssm[idx]
        return item


class SequenceBinaryClassifier(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        pssm_dim: int = 0,
        classifier_dropout: float = 0.2,
        fusion_hidden_dim: int = 256,
        pssm_hidden_dim: int = 256,
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone_ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(classifier_dropout)
        self.use_pssm = pssm_dim > 0
        self.pssm_branch: Optional[nn.Module]
        self.classifier: nn.Module

        if self.use_pssm:
            self.pssm_branch = nn.Sequential(
                nn.LayerNorm(pssm_dim),
                nn.Linear(pssm_dim, pssm_hidden_dim),
                nn.GELU(),
                nn.Dropout(classifier_dropout),
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size + pssm_hidden_dim, fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(classifier_dropout),
                nn.Linear(fusion_hidden_dim, 1),
            )
        else:
            self.pssm_branch = None
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, max(hidden_size // 2, 128)),
                nn.GELU(),
                nn.Dropout(classifier_dropout),
                nn.Linear(max(hidden_size // 2, 128), 1),
            )

    @staticmethod
    def masked_mean(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * mask).sum(dim=1)
        denom = torch.clamp(mask.sum(dim=1), min=1e-6)
        return summed / denom

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        pssm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        outputs = self.backbone(**kwargs)
        pooled = self.masked_mean(outputs.last_hidden_state, attention_mask)
        pooled = self.dropout(self.backbone_ln(pooled))

        if self.use_pssm:
            if pssm is None or self.pssm_branch is None:
                raise ValueError("PSSM features are required for lm_pssm variant.")
            pssm_feat = self.pssm_branch(pssm)
            pooled = torch.cat([pooled, pssm_feat], dim=1)

        logits = self.classifier(pooled)
        return logits.squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reviewer-facing 10-group Anti-CRISPR benchmark entry."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train one adapter checkpoint: (adapter_type in {lora,dora}) x (adapter_variant in {lm_only,lm_pssm}).
    run_parser = subparsers.add_parser("run", help="Run one benchmark cell")
    add_run_args(run_parser)

    summary_parser = subparsers.add_parser("summary", help="Rebuild aggregate tables from run outputs")
    summary_parser.add_argument("--output-root", default=str(RESULTS_ROOT))

    run10_parser = subparsers.add_parser("run-10", help="Run complete 10-group protocol and rebuild summaries")
    run10_parser.add_argument("--models", default="all", help="Comma-separated model names or 'all'.")
    run10_parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS), help="Comma-separated seeds.")
    run10_parser.add_argument("--epochs", type=int, default=8, help="Adapter training epochs for S2-S5 diagonal stage.")
    run10_parser.add_argument("--pilot", action="store_true", help="Run only the first seed.")
    run10_parser.add_argument("--resume", action="store_true", help="Skip runs that already have registry rows or done markers.")
    run10_parser.add_argument("--limit-train-batches", type=int)
    run10_parser.add_argument("--limit-eval-batches", type=int)
    run10_parser.add_argument("--max-train-samples", type=int)
    run10_parser.add_argument("--max-valid-samples", type=int)
    run10_parser.add_argument("--max-test-samples", type=int)
    run10_parser.add_argument("--head-epochs", type=int, default=40)
    run10_parser.add_argument("--head-lr", type=float, default=1e-3)
    run10_parser.add_argument("--head-batch-size", type=int, default=64)
    run10_parser.add_argument("--patience", type=int, default=3)
    run10_parser.add_argument("--dropout", type=float, default=0.3)

    return parser.parse_args()


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, choices=sorted(BACKBONE_SPECS.keys()))
    parser.add_argument("--variant", required=True, choices=["lm_only", "lm_pssm"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--adapter-type", choices=["lora", "dora"], default="lora")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--hf-model-id", default=None, help="Override the default model id.")
    parser.add_argument("--save-adapter", action="store_true")
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-eval-batches", type=int)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-valid-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_sequence(seq: str) -> str:
    seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWYUZOBX]", "X", str(seq).upper())
    return re.sub(r"[UZOB]", "X", seq)


def format_sequence_for_model(seq: str, seq_mode: str) -> str:
    normalized = normalize_sequence(seq)
    if seq_mode == "spaced":
        return " ".join(list(normalized))
    return normalized


def pick_pssm_cache(variant: str = DEFAULT_PSSM_VARIANT) -> Path:
    parquet_path = PSSM_FEATURE_DIR / f"pssm_features_{variant}.parquet"
    csv_path = PSSM_FEATURE_DIR / f"pssm_features_{variant}.csv"
    if parquet_path.exists():
        return parquet_path
    if csv_path.exists():
        return csv_path
    msg = (
        "Missing PSSM cache for lm_pssm run.\n"
        f"Expected one of:\n  - {parquet_path}\n  - {csv_path}\n"
        "Next steps:\n"
        "1) Ensure PSSM_WORK_ROOT points to your workspace (default: <repo>/data/pssm_work).\n"
        "2) Generate per-sample .pssm files with PSI-BLAST (BLAST+ + UniRef DB required).\n"
        "3) Build feature cache using:\n"
        "   python src/acrplmevo/pssm_pipeline/extract_features.py --manifest-csv <manifest.csv> --work-root <PSSM_WORK_ROOT>\n"
        "   python src/acrplmevo/pssm_pipeline/build_feature_cache.py --manifest-csv <manifest.csv> --work-root <PSSM_WORK_ROOT> --variants 1110\n"
        "4) Or run lm_only variant if you do not need PSSM fusion."
    )
    raise FileNotFoundError(msg)


def maybe_trim_df(df: pd.DataFrame, max_rows: Optional[int]) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.iloc[:max_rows].reset_index(drop=True)


def load_split_data(variant: str, seed: int, cfg: TrainConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    train_pool_df, test_df = load_anticrispr_with_ids(str(BENCHMARKS_DIR), benchmark_name="anticrispr_binary")
    feature_cols: List[str] = []

    if variant == "lm_pssm":
        feature_df, feature_cols = load_feature_cache(str(pick_pssm_cache()))
        train_pool_df = attach_pssm_features(train_pool_df, feature_df, feature_cols)
        test_df = attach_pssm_features(test_df, feature_df, feature_cols)

    train_df, valid_df = train_test_split(
        train_pool_df,
        test_size=0.1,
        stratify=train_pool_df["label"],
        random_state=seed,
    )

    train_df = maybe_trim_df(train_df, cfg.max_train_samples)
    valid_df = maybe_trim_df(valid_df, cfg.max_valid_samples)
    test_df = maybe_trim_df(test_df, cfg.max_test_samples)

    if feature_cols:
        scaler = StandardScaler()
        train_df.loc[:, feature_cols] = scaler.fit_transform(train_df.loc[:, feature_cols].to_numpy(dtype=np.float32))
        valid_df.loc[:, feature_cols] = scaler.transform(valid_df.loc[:, feature_cols].to_numpy(dtype=np.float32))
        test_df.loc[:, feature_cols] = scaler.transform(test_df.loc[:, feature_cols].to_numpy(dtype=np.float32))

    for df in (train_df, valid_df, test_df):
        df.loc[:, "label"] = df["label"].astype(int)

    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True), feature_cols


def load_tokenizer_and_model(spec: BackboneSpec, override_model_id: Optional[str] = None):
    model_id = override_model_id or spec.hf_model_id
    model_dtype = get_preferred_model_dtype()
    model_kwargs = {
        "cache_dir": str(HF_CACHE_DIR),
        "low_cpu_mem_usage": True,
        "local_files_only": True,
        "use_safetensors": False,
    }
    tokenizer_kwargs = {
        "cache_dir": str(HF_CACHE_DIR),
        "local_files_only": True,
    }
    if model_dtype is not None:
        model_kwargs["torch_dtype"] = model_dtype

    if spec.family == "bert":
        tokenizer = BertTokenizer.from_pretrained(model_id, do_lower_case=False, **tokenizer_kwargs)
        backbone = BertModel.from_pretrained(model_id, **model_kwargs)
    elif spec.family == "esm":
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        try:
            backbone = EsmModel.from_pretrained(model_id, **model_kwargs)
        except Exception:
            backbone = AutoModel.from_pretrained(model_id, **model_kwargs)
    elif spec.family == "t5":
        tokenizer = AutoTokenizer.from_pretrained(model_id, **tokenizer_kwargs)
        backbone = T5EncoderModel.from_pretrained(model_id, **model_kwargs)
    else:
        raise ValueError(f"Unsupported family: {spec.family}")

    # Default to throughput mode; can re-enable checkpointing by setting ENABLE_GRAD_CHECKPOINTING=1.
    if os.environ.get("ENABLE_GRAD_CHECKPOINTING", "0") == "1" and hasattr(backbone, "gradient_checkpointing_enable"):
        backbone.gradient_checkpointing_enable()
    if hasattr(backbone, "config") and hasattr(backbone.config, "use_cache"):
        backbone.config.use_cache = False
    if hasattr(backbone, "enable_input_require_grads"):
        backbone.enable_input_require_grads()

    hidden_size = int(getattr(backbone.config, "hidden_size", getattr(backbone.config, "d_model", 0)))
    if hidden_size <= 0:
        raise ValueError(f"Unable to infer hidden size for {model_id}")

    target_modules = resolve_target_modules(backbone, spec.target_module_candidates)
    return model_id, tokenizer, backbone, hidden_size, target_modules


def resolve_target_modules(model: nn.Module, candidates: Sequence[str]) -> List[str]:
    leaf_names = {name.split(".")[-1] for name, _ in model.named_modules() if name}
    selected = [candidate for candidate in candidates if candidate in leaf_names]
    if not selected:
        raise ValueError(f"Unable to resolve target modules from candidates={candidates}; found leaves={sorted(leaf_names)[:40]}")
    return selected


def make_collator(tokenizer, seq_mode: str, max_length: int):
    def collate_fn(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
        seqs = [format_sequence_for_model(str(item["seq"]), seq_mode) for item in batch]
        toks = tokenizer(
            seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        payload: Dict[str, object] = {
            "sample_ids": [str(item["sample_id"]) for item in batch],
            "labels": torch.tensor([int(item["label"]) for item in batch], dtype=torch.float32),
        }
        for key, value in toks.items():
            payload[key] = value
        if "pssm" in batch[0]:
            payload["pssm"] = torch.tensor(np.stack([item["pssm"] for item in batch]), dtype=torch.float32)
        return payload

    return collate_fn


def build_dataloaders(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Sequence[str],
    tokenizer,
    spec: BackboneSpec,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    collate_fn = make_collator(tokenizer, spec.seq_mode, spec.max_length)
    train_loader = DataLoader(SequenceDataset(train_df, feature_cols), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(SequenceDataset(valid_df, feature_cols), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(SequenceDataset(test_df, feature_cols), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, valid_loader, test_loader


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_preferred_model_dtype() -> Optional[torch.dtype]:
    if not torch.cuda.is_available():
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def get_autocast_dtype(device: torch.device) -> Optional[torch.dtype]:
    if device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def make_lora_model(
    backbone: nn.Module,
    hidden_size: int,
    pssm_dim: int,
    cfg: TrainConfig,
    target_modules: Sequence[str],
) -> SequenceBinaryClassifier:
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.lora_bias,
        use_dora=(cfg.adapter_type == "dora"),
        target_modules=list(target_modules),
    )
    peft_backbone = get_peft_model(backbone, lora_cfg)
    return SequenceBinaryClassifier(
        backbone=peft_backbone,
        hidden_size=hidden_size,
        pssm_dim=pssm_dim,
        classifier_dropout=cfg.classifier_dropout,
        fusion_hidden_dim=cfg.fusion_hidden_dim,
        pssm_hidden_dim=cfg.pssm_hidden_dim,
    )


def compute_pos_weight(labels: np.ndarray) -> float:
    labels = labels.astype(int)
    pos = int(labels.sum())
    neg = int(len(labels) - pos)
    if pos <= 0:
        return 1.0
    return float(max(neg, 1) / pos)


def batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    moved: Dict[str, object] = {"sample_ids": batch["sample_ids"]}
    for key in ["labels", "input_ids", "attention_mask", "token_type_ids", "pssm"]:
        if key in batch:
            moved[key] = batch[key].to(device)
    return moved


def run_epoch(
    model: SequenceBinaryClassifier,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    criterion: nn.Module,
    device: torch.device,
    train: bool,
    autocast_dtype: Optional[torch.dtype],
    scaler: Optional[torch.cuda.amp.GradScaler],
    max_batches: Optional[int],
) -> Tuple[float, np.ndarray, np.ndarray, List[str]]:
    model.train(mode=train)
    losses: List[float] = []
    probs_buffer: List[np.ndarray] = []
    labels_buffer: List[np.ndarray] = []
    sample_ids: List[str] = []

    for batch_idx, raw_batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = batch_to_device(raw_batch, device)
        labels = batch.pop("labels")
        ids = list(batch.pop("sample_ids"))

        if train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                logits = model(**batch)
                logits = logits.float()
                loss = criterion(logits, labels.float())

            if train and optimizer is not None:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

        sample_ids.extend(ids)
        labels_buffer.append(labels.detach().cpu().numpy())
        probs_buffer.append(torch.sigmoid(logits).detach().cpu().float().numpy())
        losses.append(float(loss.detach().cpu().item()))

    y_true = np.concatenate(labels_buffer).astype(np.int32) if labels_buffer else np.array([], dtype=np.int32)
    y_prob = np.concatenate(probs_buffer).astype(np.float32) if probs_buffer else np.array([], dtype=np.float32)
    avg_loss = float(np.mean(losses)) if losses else 0.0
    return avg_loss, y_true, y_prob, sample_ids


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sn = tp / (tp + fn) if (tp + fn) else 0.0
    sp = tn / (tn + fp) if (tn + fp) else 0.0
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {
        "AUC": float(roc_auc_score(y_true, y_prob)),
        "AUPRC": float(average_precision_score(y_true, y_prob)),
        "ACC": float(acc),
        "SN": float(sn),
        "SP": float(sp),
        "F1": float(f1_score(y_true, y_pred)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "ECE": float(expected_calibration_error(y_true, y_prob, n_bins=10)),
        "Threshold": float(threshold),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
    }


def count_trainable_parameters(model: nn.Module) -> Dict[str, int]:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return {"trainable_parameters": int(trainable), "total_parameters": int(total)}


def build_run_dir(adapter_type: str, model_name: str, variant: str, seed: int) -> Path:
    return RESULTS_ROOT / "runs" / adapter_type / model_name / variant / f"seed_{seed}"


def build_cross_run_dir(adapter_type: str, model_name: str, adapter_variant: str, feature_variant: str, seed: int) -> Path:
    return RESULTS_ROOT / "runs_frozen_cross_variant" / adapter_type / model_name / f"adapter_{adapter_variant}__feature_{feature_variant}" / f"seed_{seed}"


def cross_state_file(cfg: CrossFrozenRunConfig) -> Path:
    name = f"{cfg.adapter_type}__{cfg.model_name}__seed_{cfg.seed}__a_{cfg.adapter_variant}__f_{cfg.feature_variant}.done"
    return RESULTS_ROOT / "runs_frozen_cross_variant" / "_state" / name


def save_predictions(
    run_dir: Path,
    model_name: str,
    variant: str,
    seed: int,
    threshold: float,
    split_payloads: Sequence[Tuple[str, Sequence[str], np.ndarray, np.ndarray]],
) -> Path:
    frames = []
    for split_name, sample_ids, y_true, y_prob in split_payloads:
        split_df = pd.DataFrame(
            {
                "sample_id": list(sample_ids),
                "y_true": y_true.astype(int),
                "y_prob": y_prob.astype(float),
                "threshold": float(threshold),
                "seed": int(seed),
                "model": model_name,
                "variant": variant,
                "split": split_name,
                "y_pred": (y_prob >= threshold).astype(int),
            }
        )
        frames.append(split_df)

    predictions = pd.concat(frames, ignore_index=True)
    output_path = run_dir / "predictions.csv"
    predictions.to_csv(output_path, index=False)
    return output_path


def update_experiment_registry(row: Dict[str, object]) -> None:
    if EXPERIMENTS_CSV.exists():
        df = pd.read_csv(EXPERIMENTS_CSV)
        if "adapter_type" not in df.columns:
            # Backward compatibility: historical runs in this repo were LoRA.
            df["adapter_type"] = "lora"
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
    df.to_csv(EXPERIMENTS_CSV, index=False)


def experiment_row_exists(adapter_type: str, model: str, variant: str, seed: int) -> bool:
    if not EXPERIMENTS_CSV.exists():
        return False
    df = pd.read_csv(EXPERIMENTS_CSV)
    if "adapter_type" not in df.columns:
        df["adapter_type"] = "lora"
    if len(df) == 0:
        return False
    mask = (
        (df["adapter_type"].astype(str) == str(adapter_type))
        & (df["model"].astype(str) == str(model))
        & (df["variant"].astype(str) == str(variant))
        & (df["seed"].astype(int) == int(seed))
    )
    return bool(mask.any())


def _clean_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df
    first_col = str(df.columns[0])
    return df.loc[df[first_col].astype(str) != first_col].copy()


def update_frozen_cross_registry(row: Dict[str, object], drop_diagonal: bool = False) -> None:
    if EXPERIMENTS_FROZEN_CROSS_CSV.exists():
        df = pd.read_csv(EXPERIMENTS_FROZEN_CROSS_CSV)
        df = _clean_header_rows(df)
    else:
        df = pd.DataFrame()

    if len(df) > 0 and drop_diagonal:
        df = df.loc[df["adapter_variant"].astype(str) != df["feature_variant"].astype(str)].copy()

    if len(df) > 0:
        mask = ~(
            (df["adapter_type"] == row["adapter_type"])
            & (df["model"] == row["model"])
            & (df["seed"] == row["seed"])
            & (df["adapter_variant"] == row["adapter_variant"])
            & (df["feature_variant"] == row["feature_variant"])
        )
        df = df.loc[mask].copy()

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.sort_values(["adapter_type", "model", "adapter_variant", "feature_variant", "seed"]).reset_index(drop=True)
    EXPERIMENTS_FROZEN_CROSS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(EXPERIMENTS_FROZEN_CROSS_CSV, index=False)


def update_frozen_diagonal_registry(row: Dict[str, object]) -> None:
    if EXPERIMENTS_FROZEN_CSV.exists():
        df = pd.read_csv(EXPERIMENTS_FROZEN_CSV)
        df = _clean_header_rows(df)
    else:
        df = pd.DataFrame()

    if len(df) > 0:
        mask = ~(
            (df["adapter_type"].astype(str) == str(row["adapter_type"]))
            & (df["model"].astype(str) == str(row["model"]))
            & (df["variant"].astype(str) == str(row["variant"]))
            & (df["seed"].astype(int) == int(row["seed"]))
        )
        df = df.loc[mask].copy()

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.sort_values(["adapter_type", "model", "variant", "seed"]).reset_index(drop=True)
    EXPERIMENTS_FROZEN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(EXPERIMENTS_FROZEN_CSV, index=False)


def update_frozen_no_lora_registry(row: Dict[str, object]) -> None:
    if EXPERIMENTS_FROZEN_NO_LORA_CSV.exists():
        df = pd.read_csv(EXPERIMENTS_FROZEN_NO_LORA_CSV)
        df = _clean_header_rows(df)
    else:
        df = pd.DataFrame()

    if len(df) > 0:
        mask = ~(
            (df["model"].astype(str) == str(row["model"]))
            & (df["variant"].astype(str) == str(row["variant"]))
            & (df["seed"].astype(int) == int(row["seed"]))
        )
        df = df.loc[mask].copy()

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.sort_values(["model", "variant", "seed"]).reset_index(drop=True)
    EXPERIMENTS_FROZEN_NO_LORA_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(EXPERIMENTS_FROZEN_NO_LORA_CSV, index=False)


def drop_diagonal_rows_in_cross_registry() -> None:
    if not EXPERIMENTS_FROZEN_CROSS_CSV.exists():
        return
    df = pd.read_csv(EXPERIMENTS_FROZEN_CROSS_CSV)
    df = _clean_header_rows(df)
    if len(df) == 0:
        return
    filtered = df.loc[df["adapter_variant"].astype(str) != df["feature_variant"].astype(str)].copy()
    filtered = filtered.sort_values(["adapter_type", "model", "adapter_variant", "feature_variant", "seed"]).reset_index(drop=True)
    filtered.to_csv(EXPERIMENTS_FROZEN_CROSS_CSV, index=False)


def sanitize_features(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fit_pca(x_train: np.ndarray, x_valid: np.ndarray, x_test: np.ndarray, seed: int, pca_dim: int):
    x_train = sanitize_features(x_train)
    x_valid = sanitize_features(x_valid)
    x_test = sanitize_features(x_test)

    n_components = min(pca_dim, x_train.shape[1], x_train.shape[0])
    pca = PCA(n_components=n_components, random_state=seed)
    tr = pca.fit_transform(x_train)
    va = pca.transform(x_valid)
    te = pca.transform(x_test)

    explained = float(np.sum(pca.explained_variance_ratio_))
    if n_components < pca_dim:
        pad = pca_dim - n_components
        tr = np.pad(tr, ((0, 0), (0, pad)))
        va = np.pad(va, ((0, 0), (0, pad)))
        te = np.pad(te, ((0, 0), (0, pad)))

    info = {"pca_n_components": int(n_components), "pca_explained_variance_sum": explained}
    return tr.astype(np.float32), va.astype(np.float32), te.astype(np.float32), info


def build_feature_loader(df: pd.DataFrame, tokenizer, spec: BackboneSpec, batch_size: int) -> DataLoader:
    collate_fn = make_collator(tokenizer, spec.seq_mode, spec.max_length)
    dataset = SequenceDataset(df, [])
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def extract_lm_features(df: pd.DataFrame, tokenizer, backbone: nn.Module, spec: BackboneSpec, device: torch.device):
    loader = build_feature_loader(df, tokenizer, spec, spec.batch_size)
    autocast_dtype = get_autocast_dtype(device)

    feats: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    sample_ids: List[str] = []
    backbone.eval()

    for raw_batch in loader:
        ids = list(raw_batch["sample_ids"])
        y = raw_batch["labels"].numpy().astype(np.int32)

        kwargs = {
            "input_ids": raw_batch["input_ids"].to(device),
            "attention_mask": raw_batch["attention_mask"].to(device),
            "return_dict": True,
        }
        if "token_type_ids" in raw_batch:
            kwargs["token_type_ids"] = raw_batch["token_type_ids"].to(device)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                out = backbone(**kwargs)
            pooled = SequenceBinaryClassifier.masked_mean(out.last_hidden_state, kwargs["attention_mask"])
            pooled = pooled.detach().cpu().float().numpy()

        sample_ids.extend(ids)
        labels.append(y)
        feats.append(pooled)

    return sanitize_features(np.concatenate(feats, axis=0)), np.concatenate(labels, axis=0), sample_ids


def train_head(x_train, y_train, x_valid, y_valid, cfg: CrossFrozenRunConfig, device: torch.device):
    model = SmallHead(x_train.shape[1], cfg.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.head_lr)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train.astype(np.float32)))
    valid_ds = TensorDataset(torch.from_numpy(x_valid), torch.from_numpy(y_valid.astype(np.float32)))
    train_loader = DataLoader(train_ds, batch_size=cfg.head_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.head_batch_size, shuffle=False)

    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    history = []

    for epoch in range(1, cfg.head_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                valid_losses.append(float(criterion(model(xb), yb).detach().cpu().item()))

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


def predict_prob_head(model: SmallHead, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    ds = TensorDataset(torch.from_numpy(x.astype(np.float32)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    probs: List[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            probs.append(torch.sigmoid(model(xb)).detach().cpu().numpy())

    return np.concatenate(probs).astype(np.float32)


def _variant_pairs(pair_mode: str) -> List[Tuple[str, str]]:
    # Frozen evaluation pairs from the design table.
    # 冻结评估阶段使用的组合。
    # Group mapping / 组映射:
    # - (lm_only -> lm_only):  LoRA=G03, DoRA=G05
    # - (lm_only -> lm_pssm): LoRA=G04, DoRA=G06
    # - (lm_pssm -> lm_pssm): LoRA=G07, DoRA=G09
    # - (lm_pssm -> lm_only): LoRA=G08, DoRA=G10
    if pair_mode == "missing_only":
        return [("lm_only", "lm_pssm"), ("lm_pssm", "lm_only")]
    if pair_mode == "missing_plus_pssm_diagonal":
        return [("lm_only", "lm_pssm"), ("lm_pssm", "lm_only"), ("lm_pssm", "lm_pssm")]
    if pair_mode == "all_with_diagonals":
        return [
            ("lm_only", "lm_only"),
            ("lm_only", "lm_pssm"),
            ("lm_pssm", "lm_pssm"),
            ("lm_pssm", "lm_only"),
        ]
    if pair_mode == "target_only":
        return [("lm_only", "lm_pssm")]
    raise ValueError(f"Unsupported pair_mode: {pair_mode}")


def frozen_cross_row_exists(cfg: CrossFrozenRunConfig) -> bool:
    if not EXPERIMENTS_FROZEN_CROSS_CSV.exists():
        return False
    df = pd.read_csv(EXPERIMENTS_FROZEN_CROSS_CSV)
    df = _clean_header_rows(df)
    if len(df) == 0:
        return False
    mask = (
        (df["adapter_type"].astype(str) == str(cfg.adapter_type))
        & (df["model"].astype(str) == str(cfg.model_name))
        & (df["seed"].astype(int) == int(cfg.seed))
        & (df["adapter_variant"].astype(str) == str(cfg.adapter_variant))
        & (df["feature_variant"].astype(str) == str(cfg.feature_variant))
    )
    return bool(mask.any())


def frozen_no_lora_row_exists(cfg: NativeFrozenRunConfig) -> bool:
    if not EXPERIMENTS_FROZEN_NO_LORA_CSV.exists():
        return False
    df = pd.read_csv(EXPERIMENTS_FROZEN_NO_LORA_CSV)
    df = _clean_header_rows(df)
    if len(df) == 0:
        return False
    mask = (
        (df["model"].astype(str) == str(cfg.model_name))
        & (df["variant"].astype(str) == str(cfg.variant))
        & (df["seed"].astype(int) == int(cfg.seed))
    )
    return bool(mask.any())


def frozen_diagonal_row_exists(adapter_type: str, model_name: str, variant: str, seed: int) -> bool:
    if not EXPERIMENTS_FROZEN_CSV.exists():
        return False
    df = pd.read_csv(EXPERIMENTS_FROZEN_CSV)
    df = _clean_header_rows(df)
    if len(df) == 0:
        return False
    mask = (
        (df["adapter_type"].astype(str) == str(adapter_type))
        & (df["model"].astype(str) == str(model_name))
        & (df["variant"].astype(str) == str(variant))
        & (df["seed"].astype(int) == int(seed))
    )
    return bool(mask.any())


def run_one_cross_frozen(cfg: CrossFrozenRunConfig, resume: bool) -> None:
    marker = cross_state_file(cfg)
    if resume:
        if marker.exists():
            print("skip done marker:", marker.name)
            return
        if frozen_cross_row_exists(cfg):
            print(
                "skip existing row:",
                cfg.adapter_type,
                cfg.model_name,
                cfg.seed,
                cfg.adapter_variant,
                cfg.feature_variant,
            )
            return

    # Reuse a trained adapter checkpoint (tuned_lm_only or tuned_lm_pssm), then freeze it for evaluation.
    # 复用已训练的适配器checkpoint（tuned_lm_only/tuned_lm_pssm），并在评估阶段冻结。
    adapter_dir = RESULTS_ROOT / "runs" / cfg.adapter_type / cfg.model_name / cfg.adapter_variant / f"seed_{cfg.seed}" / "adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"missing adapter dir: {adapter_dir}")

    set_seed(cfg.seed)
    device = get_device()
    spec = BACKBONE_SPECS[cfg.model_name]

    # feature_variant controls prediction-time input branch:
    # lm_only => LM only; lm_pssm => LM + external PSSM.
    # feature_variant 控制预测阶段输入分支。
    train_cfg = TrainConfig(
        seed=cfg.seed,
        model_name=cfg.model_name,
        variant=cfg.feature_variant,
        batch_size=spec.batch_size,
        max_length=spec.max_length,
        adapter_type=cfg.adapter_type,
        max_train_samples=cfg.max_train_samples,
        max_valid_samples=cfg.max_valid_samples,
        max_test_samples=cfg.max_test_samples,
    )
    train_df, valid_df, test_df, feature_cols = load_split_data(cfg.feature_variant, cfg.seed, train_cfg)

    model_id, tokenizer, backbone, _, _ = load_tokenizer_and_model(spec)
    backbone = PeftModel.from_pretrained(backbone, str(adapter_dir), is_trainable=False)
    backbone = backbone.to(device)
    backbone.eval()

    x_train_raw, y_train, train_ids = extract_lm_features(train_df, tokenizer, backbone, spec, device)
    x_valid_raw, y_valid, valid_ids = extract_lm_features(valid_df, tokenizer, backbone, spec, device)
    x_test_raw, y_test, test_ids = extract_lm_features(test_df, tokenizer, backbone, spec, device)

    if feature_cols:
        pssm_train = sanitize_features(train_df[feature_cols].values.astype(np.float32))
        pssm_valid = sanitize_features(valid_df[feature_cols].values.astype(np.float32))
        pssm_test = sanitize_features(test_df[feature_cols].values.astype(np.float32))
        x_train_raw = np.concatenate([x_train_raw, pssm_train], axis=1)
        x_valid_raw = np.concatenate([x_valid_raw, pssm_valid], axis=1)
        x_test_raw = np.concatenate([x_test_raw, pssm_test], axis=1)

    x_train, x_valid, x_test, pca_info = fit_pca(x_train_raw, x_valid_raw, x_test_raw, cfg.seed, cfg.pca_dim)
    head, history = train_head(x_train, y_train, x_valid, y_valid, cfg, device)

    valid_prob = predict_prob_head(head, x_valid, cfg.head_batch_size, device)
    test_prob = predict_prob_head(head, x_test, cfg.head_batch_size, device)

    best_thr = float(find_best_threshold(y_valid, valid_prob))
    test_metrics = evaluate_predictions(y_test, test_prob, threshold=best_thr)

    out_dir = build_cross_run_dir(cfg.adapter_type, cfg.model_name, cfg.adapter_variant, cfg.feature_variant, cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

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
                    "adapter_type": cfg.adapter_type,
                    "adapter_variant": cfg.adapter_variant,
                    "feature_variant": cfg.feature_variant,
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
                    "adapter_type": cfg.adapter_type,
                    "adapter_variant": cfg.adapter_variant,
                    "feature_variant": cfg.feature_variant,
                }
            ),
        ],
        ignore_index=True,
    )
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    metrics_payload = {
        "adapter_type": cfg.adapter_type,
        "model": cfg.model_name,
        "seed": cfg.seed,
        "adapter_variant": cfg.adapter_variant,
        "feature_variant": cfg.feature_variant,
        "hf_model_id": model_id,
        "threshold": best_thr,
        "pca_info": pca_info,
        "test_metrics": test_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    config_payload = asdict(cfg)
    config_payload.update({"adapter_dir": str(adapter_dir), "output_dir": str(out_dir)})
    (out_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    row = {
        "adapter_type": cfg.adapter_type,
        "model": cfg.model_name,
        "seed": cfg.seed,
        "adapter_variant": cfg.adapter_variant,
        "feature_variant": cfg.feature_variant,
        "AUC": float(test_metrics["AUC"]),
        "AUPRC": float(test_metrics["AUPRC"]),
        "ACC": float(test_metrics["ACC"]),
        "F1": float(test_metrics["F1"]),
        "MCC": float(test_metrics["MCC"]),
        "Threshold": float(test_metrics["Threshold"]),
        "metrics_path": str(out_dir / "metrics.json"),
        "predictions_path": str(out_dir / "predictions.csv"),
        "config_path": str(out_dir / "config.json"),
    }
    update_frozen_cross_registry(row, drop_diagonal=False)

    if cfg.adapter_variant == cfg.feature_variant:
        diagonal_row = {
            "adapter_type": cfg.adapter_type,
            "model": cfg.model_name,
            "variant": cfg.adapter_variant,
            "seed": cfg.seed,
            "AUC": float(test_metrics["AUC"]),
            "AUPRC": float(test_metrics["AUPRC"]),
            "ACC": float(test_metrics["ACC"]),
            "F1": float(test_metrics["F1"]),
            "MCC": float(test_metrics["MCC"]),
            "Threshold": float(test_metrics["Threshold"]),
            "metrics_path": str(out_dir / "metrics.json"),
            "predictions_path": str(out_dir / "predictions.csv"),
            "config_path": str(out_dir / "config.json"),
        }
        update_frozen_diagonal_registry(diagonal_row)

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("done\n", encoding="utf-8")
    print(
        f"done: {cfg.adapter_type} {cfg.model_name} seed={cfg.seed} a={cfg.adapter_variant} f={cfg.feature_variant}"
    )


def run_cross_frozen(args: argparse.Namespace) -> None:
    models = expand_csv_arg(args.models, BACKBONE_SPECS.keys())
    adapter_types = expand_csv_arg(args.adapter_types, ["lora", "dora"])
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    pairs = _variant_pairs(args.pair_mode)

    for adapter_type in adapter_types:
        if adapter_type not in {"lora", "dora"}:
            raise ValueError(f"Unsupported adapter type: {adapter_type}")
        for model_name in models:
            for seed in seeds:
                for adapter_variant, feature_variant in pairs:
                    # One cross cell = (adapter_type, model, seed, adapter_variant, feature_variant)
                    # 一个交叉实验单元 = (适配器类型, 模型, 随机种子, 训练主干变体, 预测输入变体)
                    # 10-group mapping for cross cells:
                    # 该阶段对应10组中的“交叉列”:
                    # - lora + (lm_only->lm_pssm) => G04
                    # - dora + (lm_only->lm_pssm) => G06
                    # - lora + (lm_pssm->lm_only) => G08
                    # - dora + (lm_pssm->lm_only) => G10
                    cfg = CrossFrozenRunConfig(
                        adapter_type=adapter_type,
                        model_name=model_name,
                        seed=seed,
                        adapter_variant=adapter_variant,
                        feature_variant=feature_variant,
                        head_epochs=int(args.head_epochs),
                        head_lr=float(args.head_lr),
                        head_batch_size=int(args.head_batch_size),
                        patience=int(args.patience),
                        dropout=float(args.dropout),
                        max_train_samples=args.max_train_samples,
                        max_valid_samples=args.max_valid_samples,
                        max_test_samples=args.max_test_samples,
                    )
                    run_one_cross_frozen(cfg, resume=bool(args.resume))


def native_state_file(cfg: NativeFrozenRunConfig) -> Path:
    name = f"no_lora__{cfg.model_name}__{cfg.variant}__seed_{cfg.seed}.done"
    return RESULTS_ROOT / "runs_frozen_no_lora" / "_state" / name


def build_native_run_dir(model_name: str, variant: str, seed: int) -> Path:
    return RESULTS_ROOT / "runs_frozen_no_lora" / model_name / variant / f"seed_{seed}"


def run_one_native_frozen(cfg: NativeFrozenRunConfig, resume: bool) -> None:
    marker = native_state_file(cfg)
    if resume:
        if marker.exists():
            print("skip done marker:", marker.name)
            return
        if frozen_no_lora_row_exists(cfg):
            print("skip existing row:", cfg.model_name, cfg.variant, cfg.seed)
            return

    set_seed(cfg.seed)
    device = get_device()
    spec = BACKBONE_SPECS[cfg.model_name]

    train_cfg = TrainConfig(
        seed=cfg.seed,
        model_name=cfg.model_name,
        variant=cfg.variant,
        batch_size=spec.batch_size,
        max_length=spec.max_length,
        adapter_type="none",
        max_train_samples=cfg.max_train_samples,
        max_valid_samples=cfg.max_valid_samples,
        max_test_samples=cfg.max_test_samples,
    )
    train_df, valid_df, test_df, feature_cols = load_split_data(cfg.variant, cfg.seed, train_cfg)

    model_id, tokenizer, backbone, _, _ = load_tokenizer_and_model(spec)
    for param in backbone.parameters():
        param.requires_grad_(False)
    backbone = backbone.to(device)
    backbone.eval()

    x_train_raw, y_train, _ = extract_lm_features(train_df, tokenizer, backbone, spec, device)
    x_valid_raw, y_valid, valid_ids = extract_lm_features(valid_df, tokenizer, backbone, spec, device)
    x_test_raw, y_test, test_ids = extract_lm_features(test_df, tokenizer, backbone, spec, device)

    if feature_cols:
        pssm_train = sanitize_features(train_df[feature_cols].values.astype(np.float32))
        pssm_valid = sanitize_features(valid_df[feature_cols].values.astype(np.float32))
        pssm_test = sanitize_features(test_df[feature_cols].values.astype(np.float32))
        x_train_raw = np.concatenate([x_train_raw, pssm_train], axis=1)
        x_valid_raw = np.concatenate([x_valid_raw, pssm_valid], axis=1)
        x_test_raw = np.concatenate([x_test_raw, pssm_test], axis=1)

    x_train, x_valid, x_test, pca_info = fit_pca(x_train_raw, x_valid_raw, x_test_raw, cfg.seed, cfg.pca_dim)
    head, history = train_head(x_train, y_train, x_valid, y_valid, cfg, device)

    valid_prob = predict_prob_head(head, x_valid, cfg.head_batch_size, device)
    test_prob = predict_prob_head(head, x_test, cfg.head_batch_size, device)

    best_thr = float(find_best_threshold(y_valid, valid_prob))
    test_metrics = evaluate_predictions(y_test, test_prob, threshold=best_thr)

    out_dir = build_native_run_dir(cfg.model_name, cfg.variant, cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

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
                    "adapter_type": "none",
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
                    "adapter_type": "none",
                }
            ),
        ],
        ignore_index=True,
    )
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    metrics_payload = {
        "adapter_type": "none",
        "model": cfg.model_name,
        "variant": cfg.variant,
        "seed": cfg.seed,
        "hf_model_id": model_id,
        "threshold": best_thr,
        "pca_info": pca_info,
        "test_metrics": test_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    config_payload = asdict(cfg)
    config_payload.update({"output_dir": str(out_dir), "adapter_type": "none"})
    (out_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    row = {
        "adapter_type": "none",
        "model": cfg.model_name,
        "variant": cfg.variant,
        "seed": cfg.seed,
        "AUC": float(test_metrics["AUC"]),
        "AUPRC": float(test_metrics["AUPRC"]),
        "ACC": float(test_metrics["ACC"]),
        "F1": float(test_metrics["F1"]),
        "MCC": float(test_metrics["MCC"]),
        "Threshold": float(test_metrics["Threshold"]),
        "metrics_path": str(out_dir / "metrics.json"),
        "predictions_path": str(out_dir / "predictions.csv"),
        "config_path": str(out_dir / "config.json"),
    }
    update_frozen_no_lora_registry(row)

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("done\n", encoding="utf-8")
    print(f"done S1-native: {cfg.model_name} {cfg.variant} seed={cfg.seed}")


def run_native_frozen(args: argparse.Namespace) -> None:
    models = expand_csv_arg(args.models, BACKBONE_SPECS.keys())
    variants = expand_csv_arg(args.variants, ["lm_only", "lm_pssm"])
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]

    for model_name in models:
        for variant in variants:
            for seed in seeds:
                # 10-group mapping for S1 native row (implemented in-script):
                # S1原生行在主脚本内实现，对应:
                # - variant=lm_only => G01
                # - variant=lm_pssm => G02
                cfg = NativeFrozenRunConfig(
                    model_name=model_name,
                    seed=seed,
                    variant=variant,
                    head_epochs=int(args.head_epochs),
                    head_lr=float(args.head_lr),
                    head_batch_size=int(args.head_batch_size),
                    patience=int(args.patience),
                    dropout=float(args.dropout),
                    max_train_samples=args.max_train_samples,
                    max_valid_samples=args.max_valid_samples,
                    max_test_samples=args.max_test_samples,
                )
                run_one_native_frozen(cfg, resume=bool(args.resume))


def _append_group_mapping_rows(rows: List[Dict[str, object]], df: pd.DataFrame, source: str) -> None:
    for _, rec in df.iterrows():
        model = str(rec.get("model", ""))
        seed = int(rec.get("seed", -1))
        auc = float(rec.get("AUC", np.nan))
        auprc = float(rec.get("AUPRC", np.nan))
        acc = float(rec.get("ACC", np.nan))
        f1 = float(rec.get("F1", np.nan))
        mcc = float(rec.get("MCC", np.nan))

        if source == "native":
            variant = str(rec.get("variant", ""))
            if variant == "lm_only":
                gid, strategy, adapter_type, adapter_variant, external = "G01", "S1", "none", "native", "lm_only"
            elif variant == "lm_pssm":
                gid, strategy, adapter_type, adapter_variant, external = "G02", "S1", "none", "native", "lm_pssm"
            else:
                continue
        elif source == "diagonal":
            adapter_type = str(rec.get("adapter_type", ""))
            variant = str(rec.get("variant", ""))
            if adapter_type == "lora" and variant == "lm_only":
                gid, strategy, adapter_variant, external = "G03", "S2", "lm_only", "lm_only"
            elif adapter_type == "dora" and variant == "lm_only":
                gid, strategy, adapter_variant, external = "G05", "S3", "lm_only", "lm_only"
            elif adapter_type == "lora" and variant == "lm_pssm":
                gid, strategy, adapter_variant, external = "G07", "S4", "lm_pssm", "lm_pssm"
            elif adapter_type == "dora" and variant == "lm_pssm":
                gid, strategy, adapter_variant, external = "G09", "S5", "lm_pssm", "lm_pssm"
            else:
                continue
        elif source == "cross":
            adapter_type = str(rec.get("adapter_type", ""))
            adapter_variant = str(rec.get("adapter_variant", ""))
            feature_variant = str(rec.get("feature_variant", ""))
            external = feature_variant
            if adapter_type == "lora" and adapter_variant == "lm_only" and feature_variant == "lm_only":
                gid, strategy = "G03", "S2"
            elif adapter_type == "dora" and adapter_variant == "lm_only" and feature_variant == "lm_only":
                gid, strategy = "G05", "S3"
            elif adapter_type == "lora" and adapter_variant == "lm_only" and feature_variant == "lm_pssm":
                gid, strategy = "G04", "S2"
            elif adapter_type == "dora" and adapter_variant == "lm_only" and feature_variant == "lm_pssm":
                gid, strategy = "G06", "S3"
            elif adapter_type == "lora" and adapter_variant == "lm_pssm" and feature_variant == "lm_only":
                gid, strategy = "G08", "S4"
            elif adapter_type == "dora" and adapter_variant == "lm_pssm" and feature_variant == "lm_only":
                gid, strategy = "G10", "S5"
            elif adapter_type == "lora" and adapter_variant == "lm_pssm" and feature_variant == "lm_pssm":
                gid, strategy = "G07", "S4"
            elif adapter_type == "dora" and adapter_variant == "lm_pssm" and feature_variant == "lm_pssm":
                gid, strategy = "G09", "S5"
            else:
                continue
        else:
            continue

        rows.append(
            {
                "group_id": gid,
                "strategy": strategy,
                "source": source,
                "adapter_type": adapter_type,
                "adapter_variant": adapter_variant,
                "external_variant": external,
                "model": model,
                "seed": seed,
                "AUC": auc,
                "AUPRC": auprc,
                "ACC": acc,
                "F1": f1,
                "MCC": mcc,
            }
        )


def rebuild_10_group_summaries(output_root: Path = RESULTS_ROOT) -> None:
    rows: List[Dict[str, object]] = []
    cross_df: Optional[pd.DataFrame] = None
    diagonal_df: Optional[pd.DataFrame] = None

    if EXPERIMENTS_FROZEN_CROSS_CSV.exists():
        cross_df = _clean_header_rows(pd.read_csv(EXPERIMENTS_FROZEN_CROSS_CSV))

    if EXPERIMENTS_FROZEN_NO_LORA_CSV.exists():
        native_df = _clean_header_rows(pd.read_csv(EXPERIMENTS_FROZEN_NO_LORA_CSV))
        if len(native_df) > 0:
            _append_group_mapping_rows(rows, native_df, source="native")

    if EXPERIMENTS_FROZEN_CSV.exists():
        diagonal_df = _clean_header_rows(pd.read_csv(EXPERIMENTS_FROZEN_CSV))
        if len(diagonal_df) > 0:
            _append_group_mapping_rows(rows, diagonal_df, source="diagonal")

    if cross_df is not None and len(cross_df) > 0:
        cross_only = cross_df.loc[
            cross_df["adapter_variant"].astype(str) != cross_df["feature_variant"].astype(str)
        ].copy()
        if len(cross_only) > 0:
            _append_group_mapping_rows(rows, cross_only, source="cross")

    if not rows:
        return

    run_df = pd.DataFrame(rows)
    run_df = run_df.sort_values(["group_id", "model", "seed"]).reset_index(drop=True)
    run_df.to_csv(output_root / SUMMARY_10GROUP_RUNS_CSV.name, index=False)

    metric_cols = ["AUC", "AUPRC", "ACC", "F1", "MCC"]
    grouped = run_df.groupby(
        ["group_id", "strategy", "adapter_type", "adapter_variant", "external_variant", "model"], as_index=False
    )

    agg_rows = []
    for keys, grp in grouped:
        group_id, strategy, adapter_type, adapter_variant, external_variant, model = keys
        row = {
            "group_id": group_id,
            "strategy": strategy,
            "adapter_type": adapter_type,
            "adapter_variant": adapter_variant,
            "external_variant": external_variant,
            "model": model,
            "n_seeds": int(grp["seed"].nunique()),
        }
        for metric in metric_cols:
            row[f"{metric}_mean"] = float(grp[metric].mean())
            row[f"{metric}_std"] = float(grp[metric].std(ddof=0))
        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows).sort_values(["group_id", "model"]).reset_index(drop=True)
    agg_df.to_csv(output_root / SUMMARY_10GROUP_BY_MODEL_CSV.name, index=False)


def _collect_six_category_rows(output_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    if EXPERIMENTS_FROZEN_NO_LORA_CSV.exists():
        no_lora = _clean_header_rows(pd.read_csv(EXPERIMENTS_FROZEN_NO_LORA_CSV))
        for _, rec in no_lora.iterrows():
            rows.append(
                {
                    "main_backbone": "native",
                    "external_feature": str(rec.get("variant", "")),
                    "adapter_type": "none",
                    "model": str(rec.get("model", "")),
                    "seed": int(rec.get("seed", -1)),
                    "AUC": float(rec.get("AUC", np.nan)),
                    "AUPRC": float(rec.get("AUPRC", np.nan)),
                    "ACC": float(rec.get("ACC", np.nan)),
                    "F1": float(rec.get("F1", np.nan)),
                    "MCC": float(rec.get("MCC", np.nan)),
                }
            )

    if EXPERIMENTS_FROZEN_CSV.exists():
        diag = _clean_header_rows(pd.read_csv(EXPERIMENTS_FROZEN_CSV))
        for _, rec in diag.iterrows():
            variant = str(rec.get("variant", ""))
            if variant not in {"lm_only", "lm_pssm"}:
                continue
            rows.append(
                {
                    "main_backbone": f"tuned_{variant}",
                    "external_feature": variant,
                    "adapter_type": str(rec.get("adapter_type", "")),
                    "model": str(rec.get("model", "")),
                    "seed": int(rec.get("seed", -1)),
                    "AUC": float(rec.get("AUC", np.nan)),
                    "AUPRC": float(rec.get("AUPRC", np.nan)),
                    "ACC": float(rec.get("ACC", np.nan)),
                    "F1": float(rec.get("F1", np.nan)),
                    "MCC": float(rec.get("MCC", np.nan)),
                }
            )

    if EXPERIMENTS_FROZEN_CROSS_CSV.exists():
        cross = _clean_header_rows(pd.read_csv(EXPERIMENTS_FROZEN_CROSS_CSV))
        if len(cross) > 0:
            cross = cross.loc[cross["adapter_variant"].astype(str) != cross["feature_variant"].astype(str)].copy()
        for _, rec in cross.iterrows():
            rows.append(
                {
                    "main_backbone": f"tuned_{str(rec.get('adapter_variant', ''))}",
                    "external_feature": str(rec.get("feature_variant", "")),
                    "adapter_type": str(rec.get("adapter_type", "")),
                    "model": str(rec.get("model", "")),
                    "seed": int(rec.get("seed", -1)),
                    "AUC": float(rec.get("AUC", np.nan)),
                    "AUPRC": float(rec.get("AUPRC", np.nan)),
                    "ACC": float(rec.get("ACC", np.nan)),
                    "F1": float(rec.get("F1", np.nan)),
                    "MCC": float(rec.get("MCC", np.nan)),
                }
            )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values(["main_backbone", "external_feature", "adapter_type", "model", "seed"]).reset_index(drop=True)
    return out


def rebuild_six_category_tables(output_root: Path = RESULTS_ROOT) -> None:
    df = _collect_six_category_rows(output_root)
    if len(df) == 0:
        return

    grouped = df.groupby(["main_backbone", "external_feature", "adapter_type", "model"], as_index=False)

    mean_rows: List[Dict[str, object]] = []
    pretty_rows: List[Dict[str, object]] = []
    for keys, grp in grouped:
        main_backbone, external_feature, adapter_type, model = keys
        row = {
            "main_backbone": main_backbone,
            "external_feature": external_feature,
            "adapter_type": adapter_type,
            "model": model,
            "n_seeds": int(grp["seed"].nunique()),
            "mean_AUC": float(grp["AUC"].mean()),
            "mean_AUPRC": float(grp["AUPRC"].mean()),
            "mean_F1": float(grp["F1"].mean()),
            "mean_MCC": float(grp["MCC"].mean()),
        }
        mean_rows.append(row)

        method = (
            "Frozen native"
            if main_backbone == "native"
            else f"{adapter_type.upper()}-tuned on {main_backbone.replace('tuned_', '')}"
        )
        pretty_rows.append(
            {
                "external_feature": external_feature,
                "method": method,
                "model": model,
                "AUC_mean": row["mean_AUC"],
                "AUC_std": float(grp["AUC"].std(ddof=0)),
                "AUPRC_mean": row["mean_AUPRC"],
                "AUPRC_std": float(grp["AUPRC"].std(ddof=0)),
                "F1_mean": row["mean_F1"],
                "F1_std": float(grp["F1"].std(ddof=0)),
                "MCC_mean": row["mean_MCC"],
                "MCC_std": float(grp["MCC"].std(ddof=0)),
            }
        )

    mean_df = pd.DataFrame(mean_rows).sort_values(["main_backbone", "external_feature", "adapter_type", "model"]).reset_index(drop=True)
    mean_df.to_csv(output_root / SIX_CATEGORY_SEEDMEAN_CSV.name, index=False)

    best_rows: List[Dict[str, object]] = []
    for keys, grp in grouped:
        main_backbone, external_feature, adapter_type, model = keys
        ranked = grp.sort_values(["AUC", "AUPRC", "F1", "MCC", "seed"], ascending=[False, False, False, False, True])
        top = ranked.iloc[0]
        best_rows.append(
            {
                "main_backbone": main_backbone,
                "external_feature": external_feature,
                "adapter_type": adapter_type,
                "model": model,
                "best_seed": int(top["seed"]),
                "best_AUC": float(top["AUC"]),
                "best_AUPRC": float(top["AUPRC"]),
                "best_F1": float(top["F1"]),
                "best_MCC": float(top["MCC"]),
                "best_rule": "AUC desc, then AUPRC desc, then F1 desc, then MCC desc",
            }
        )

    best_df = pd.DataFrame(best_rows).sort_values(["main_backbone", "external_feature", "adapter_type", "model"]).reset_index(drop=True)
    best_df.to_csv(output_root / SIX_CATEGORY_BEST_CSV.name, index=False)

    SIX_CATEGORY_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pretty_rows).sort_values(["external_feature", "method", "model"]).reset_index(drop=True).to_csv(
        SIX_CATEGORY_MEAN_CSV, index=False
    )


def run_ten_group_protocol(args: argparse.Namespace) -> None:
    models = args.models
    seeds = args.seeds
    if args.pilot:
        parsed = [int(seed.strip()) for seed in seeds.split(",") if seed.strip()]
        if parsed:
            seeds = str(parsed[0])

    print("[run-10] stage 1/4: S1 native row (G01-G02)")
    model_list = expand_csv_arg(models, BACKBONE_SPECS.keys())
    seed_list = [int(seed.strip()) for seed in seeds.split(",") if seed.strip()]
    for model_name in model_list:
        for variant in ["lm_only", "lm_pssm"]:
            for seed in seed_list:
                cfg = NativeFrozenRunConfig(
                    model_name=model_name,
                    seed=seed,
                    variant=variant,
                    head_epochs=int(args.head_epochs),
                    head_lr=float(args.head_lr),
                    head_batch_size=int(args.head_batch_size),
                    patience=int(args.patience),
                    dropout=float(args.dropout),
                    max_train_samples=args.max_train_samples,
                    max_valid_samples=args.max_valid_samples,
                    max_test_samples=args.max_test_samples,
                )
                run_one_native_frozen(cfg, resume=bool(args.resume))

    for adapter_type in ["lora", "dora"]:
        print(f"[run-10] stage 2/4: S2-S5 diagonal cells (adapter_type={adapter_type})")
        for model_name in model_list:
            for variant in ["lm_only", "lm_pssm"]:
                for seed in seed_list:
                    if bool(args.resume) and experiment_row_exists(adapter_type, model_name, variant, seed):
                        print(f"skip existing row: adapter={adapter_type} model={model_name} variant={variant} seed={seed}")
                        continue
                    child_args = argparse.Namespace(
                        model=model_name,
                        variant=variant,
                        seed=seed,
                        adapter_type=adapter_type,
                        epochs=int(args.epochs),
                        learning_rate=2e-4,
                        weight_decay=1e-4,
                        lora_r=8,
                        lora_alpha=16,
                        lora_dropout=0.1,
                        hf_model_id=None,
                        save_adapter=True,
                        limit_train_batches=args.limit_train_batches,
                        limit_eval_batches=args.limit_eval_batches,
                        max_train_samples=args.max_train_samples,
                        max_valid_samples=args.max_valid_samples,
                        max_test_samples=args.max_test_samples,
                    )
                    print(
                        f"starting adapter={adapter_type} model={model_name} variant={variant} seed={seed}"
                    )
                    train_one_run(child_args)

    print("[run-10] stage 3/4: S2-S5 frozen cells (write diagonal to experiments_frozen, cross to cross_registry)")
    for adapter_type in ["lora", "dora"]:
        for model_name in model_list:
            for seed in seed_list:
                for adapter_variant, feature_variant in _variant_pairs("all_with_diagonals"):
                    if bool(args.resume) and adapter_variant == feature_variant:
                        if frozen_diagonal_row_exists(adapter_type, model_name, adapter_variant, seed):
                            print(
                                f"skip existing frozen diagonal: adapter={adapter_type} model={model_name} "
                                f"variant={adapter_variant} seed={seed}"
                            )
                            continue
                    cfg = CrossFrozenRunConfig(
                        adapter_type=adapter_type,
                        model_name=model_name,
                        seed=seed,
                        adapter_variant=adapter_variant,
                        feature_variant=feature_variant,
                        head_epochs=int(args.head_epochs),
                        head_lr=float(args.head_lr),
                        head_batch_size=int(args.head_batch_size),
                        patience=int(args.patience),
                        dropout=float(args.dropout),
                        max_train_samples=args.max_train_samples,
                        max_valid_samples=args.max_valid_samples,
                        max_test_samples=args.max_test_samples,
                    )
                    run_one_cross_frozen(cfg, resume=bool(args.resume))

    print("[run-10] stage 4/4: rebuild summary tables")
    rebuild_summaries(RESULTS_ROOT)
    rebuild_10_group_summaries(RESULTS_ROOT)
    print("[run-10] done: all 10 groups finished and unified summaries rebuilt.")


def rebuild_summaries(output_root: Path = RESULTS_ROOT) -> None:
    if not EXPERIMENTS_FROZEN_CSV.exists():
        return

    df = _clean_header_rows(pd.read_csv(EXPERIMENTS_FROZEN_CSV))
    if len(df) == 0:
        return

    metric_cols = ["AUC", "AUPRC", "ACC", "F1", "MCC", "Threshold"]
    grouped = df.groupby(["adapter_type", "model", "variant"], as_index=False)

    rows = []
    for (adapter_type, model_name, variant), group in grouped:
        row: Dict[str, object] = {
            "adapter_type": adapter_type,
            "model": model_name,
            "variant": variant,
            "n_seeds": int(group["seed"].nunique()),
        }
        for metric in metric_cols:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        rows.append(row)

    summary_df = pd.DataFrame(rows).sort_values(["adapter_type", "model", "variant"]).reset_index(drop=True)
    summary_df.to_csv(output_root / SUMMARY_CSV.name, index=False)

    lm_only = df[df["variant"] == "lm_only"]
    lm_pssm = df[df["variant"] == "lm_pssm"]
    if lm_only.empty or lm_pssm.empty:
        return

    merged = lm_pssm.merge(lm_only, on=["adapter_type", "model", "seed"], suffixes=("_pssm", "_lm_only"))
    if merged.empty:
        return

    delta_rows = []
    for (adapter_type, model_name), group in merged.groupby(["adapter_type", "model"]):
        row = {"adapter_type": adapter_type, "model": model_name, "n_seeds": int(group["seed"].nunique())}
        for metric in ["AUC", "AUPRC", "ACC", "F1", "MCC"]:
            delta = group[f"{metric}_pssm"] - group[f"{metric}_lm_only"]
            row[f"delta_{metric}_mean"] = float(delta.mean())
            row[f"delta_{metric}_std"] = float(delta.std(ddof=0))
        delta_rows.append(row)

    pd.DataFrame(delta_rows).sort_values(["adapter_type", "model"]).reset_index(drop=True).to_csv(
        output_root / DELTA_CSV.name, index=False
    )

    rebuild_six_category_tables(output_root)


def train_one_run(args: argparse.Namespace) -> Dict[str, object]:
    spec = BACKBONE_SPECS[args.model]
    cfg = TrainConfig(
        seed=int(args.seed),
        model_name=args.model,
        variant=args.variant,
        batch_size=spec.batch_size,
        max_length=spec.max_length,
        adapter_type=str(args.adapter_type),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        epochs=int(args.epochs),
        lora_r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        max_train_samples=args.max_train_samples,
        max_valid_samples=args.max_valid_samples,
        max_test_samples=args.max_test_samples,
        limit_train_batches=args.limit_train_batches,
        limit_eval_batches=args.limit_eval_batches,
    )

    set_seed(cfg.seed)
    train_df, valid_df, test_df, feature_cols = load_split_data(cfg.variant, cfg.seed, cfg)
    model_id, tokenizer, backbone, hidden_size, target_modules = load_tokenizer_and_model(spec, args.hf_model_id)
    model = make_lora_model(backbone, hidden_size, len(feature_cols), cfg, target_modules)

    device = get_device()
    autocast_dtype = get_autocast_dtype(device)
    model = model.to(device)
    param_info = count_trainable_parameters(model)

    train_loader, valid_loader, test_loader = build_dataloaders(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        feature_cols=feature_cols,
        tokenizer=tokenizer,
        spec=spec,
        batch_size=cfg.batch_size,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([compute_pos_weight(train_df["label"].to_numpy(dtype=int))], dtype=torch.float32, device=device)
    )
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and autocast_dtype == torch.float16))
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and autocast_dtype == torch.float16))

    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    patience_counter = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss, _, _, _ = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train=True,
            autocast_dtype=autocast_dtype,
            scaler=scaler,
            max_batches=cfg.limit_train_batches,
        )
        valid_loss, valid_y, valid_prob, valid_ids = run_epoch(
            model=model,
            loader=valid_loader,
            optimizer=None,
            criterion=criterion,
            device=device,
            train=False,
            autocast_dtype=autocast_dtype,
            scaler=None,
            max_batches=cfg.limit_eval_batches,
        )
        valid_thr = find_best_threshold(valid_y, valid_prob)
        valid_metrics = evaluate_predictions(valid_y, valid_prob, threshold=valid_thr)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "valid_auc": valid_metrics["AUC"],
                "valid_f1": valid_metrics["F1"],
            }
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} "
            f"valid_auc={valid_metrics['AUC']:.4f} valid_f1={valid_metrics['F1']:.4f} threshold={valid_thr:.2f}"
        )

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print("early stopping triggered")
                break

    model.load_state_dict(best_state)

    valid_loss, valid_y, valid_prob, valid_ids = run_epoch(
        model=model,
        loader=valid_loader,
        optimizer=None,
        criterion=criterion,
        device=device,
        train=False,
        autocast_dtype=autocast_dtype,
        scaler=None,
        max_batches=cfg.limit_eval_batches,
    )
    threshold = find_best_threshold(valid_y, valid_prob)
    valid_metrics = evaluate_predictions(valid_y, valid_prob, threshold=threshold)
    test_loss, test_y, test_prob, test_ids = run_epoch(
        model=model,
        loader=test_loader,
        optimizer=None,
        criterion=criterion,
        device=device,
        train=False,
        autocast_dtype=autocast_dtype,
        scaler=None,
        max_batches=cfg.limit_eval_batches,
    )
    test_metrics = evaluate_predictions(test_y, test_prob, threshold=threshold)

    run_dir = build_run_dir(cfg.adapter_type, cfg.model_name, cfg.variant, cfg.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = save_predictions(
        run_dir,
        model_name=cfg.model_name,
        variant=cfg.variant,
        seed=cfg.seed,
        threshold=threshold,
        split_payloads=[
            ("valid", valid_ids, valid_y, valid_prob),
            ("test", test_ids, test_y, test_prob),
        ],
    )

    metrics_payload = {
        "adapter_type": cfg.adapter_type,
        "model": cfg.model_name,
        "variant": cfg.variant,
        "seed": cfg.seed,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "hf_model_id": model_id,
        "train_pool_size": int(len(train_df) + len(valid_df)),
        "train_size": int(len(train_df)),
        "valid_size": int(len(valid_df)),
        "test_size": int(len(test_df)),
        "feature_dim": int(len(feature_cols)),
        "target_modules": list(target_modules),
        "valid_loss": float(valid_loss),
        "test_loss": float(test_loss),
        "valid_metrics": valid_metrics,
        "test_metrics": test_metrics,
        "history": history,
    }
    metrics_payload.update(param_info)

    config_payload = asdict(cfg)
    config_payload.update(
        {
            "use_dora": bool(cfg.adapter_type == "dora"),
            "hf_model_id": model_id,
            "target_modules": list(target_modules),
            "feature_cols": list(feature_cols),
            "hf_cache_dir": str(HF_CACHE_DIR),
            "pssm_cache": str(pick_pssm_cache()) if cfg.variant == "lm_pssm" else None,
            "device": str(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }
    )

    (run_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)

    if args.save_adapter:
        adapter_dir = run_dir / "adapter"
        model.backbone.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        torch.save(model.state_dict(), run_dir / "full_model_state.pt")

    registry_row: Dict[str, object] = {
        "adapter_type": cfg.adapter_type,
        "model": cfg.model_name,
        "variant": cfg.variant,
        "seed": cfg.seed,
        "hf_model_id": model_id,
        "AUC": test_metrics["AUC"],
        "AUPRC": test_metrics["AUPRC"],
        "ACC": test_metrics["ACC"],
        "SN": test_metrics["SN"],
        "SP": test_metrics["SP"],
        "F1": test_metrics["F1"],
        "MCC": test_metrics["MCC"],
        "Threshold": test_metrics["Threshold"],
        "metrics_path": str(run_dir / "metrics.json"),
        "predictions_path": str(predictions_path),
        "config_path": str(run_dir / "config.json"),
    }
    update_experiment_registry(registry_row)
    rebuild_summaries(RESULTS_ROOT)
    return registry_row


def expand_csv_arg(value: str, allowed: Iterable[str]) -> List[str]:
    if value == "all":
        return list(allowed)
    return [item.strip() for item in value.split(",") if item.strip()]


def run_matrix(args: argparse.Namespace) -> None:
    models = expand_csv_arg(args.models, BACKBONE_SPECS.keys())
    variants = expand_csv_arg(args.variants, ["lm_only", "lm_pssm"])
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    if args.pilot and seeds:
        seeds = [seeds[0]]

    for model_name in models:
        for variant in variants:
            for seed in seeds:
                if bool(getattr(args, "resume", False)) and experiment_row_exists(args.adapter_type, model_name, variant, seed):
                    print(f"skip existing row: adapter={args.adapter_type} model={model_name} variant={variant} seed={seed}")
                    continue
                # Adapter training stage (S2-S5 rows):
                # variant=lm_only/lm_pssm with adapter_type=lora/dora.
                # 适配器训练阶段（对应S2-S5）。
                # 10-group mapping for run/matrix (diagonal cells):
                # run/matrix对应10组中的“同变体列”（对角单元）:
                # - lora + lm_only => G03
                # - dora + lm_only => G05
                # - lora + lm_pssm => G07
                # - dora + lm_pssm => G09
                child_args = argparse.Namespace(
                    model=model_name,
                    variant=variant,
                    seed=seed,
                    adapter_type=args.adapter_type,
                    epochs=args.epochs,
                    learning_rate=2e-4,
                    weight_decay=1e-4,
                    lora_r=8,
                    lora_alpha=16,
                    lora_dropout=0.1,
                    hf_model_id=None,
                    save_adapter=args.save_adapter,
                    limit_train_batches=args.limit_train_batches,
                    limit_eval_batches=args.limit_eval_batches,
                    max_train_samples=args.max_train_samples,
                    max_valid_samples=args.max_valid_samples,
                    max_test_samples=args.max_test_samples,
                )
                print(
                    f"starting adapter={args.adapter_type} model={model_name} variant={variant} seed={seed}"
                )
                train_one_run(child_args)


def main() -> None:
    args = parse_args()
    if args.command == "run":
        train_one_run(args)
    elif args.command == "run-10":
        run_ten_group_protocol(args)
    elif args.command == "summary":
        rebuild_summaries(Path(args.output_root))
        rebuild_10_group_summaries(Path(args.output_root))
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()