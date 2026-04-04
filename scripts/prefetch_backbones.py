#!/usr/bin/env python3
import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from huggingface_hub import snapshot_download


HF_CACHE_DIR = Path(
    os.environ.get("ACRPLMEVO_HF_CACHE_DIR")
    or os.environ.get("HF_HOME")
    or str(Path.home() / ".cache" / "huggingface")
)


@dataclass(frozen=True)
class PrefetchSpec:
    name: str
    repo_id: str
    min_complete_bytes: int


PREFETCH_SPECS: Dict[str, PrefetchSpec] = {
    "protbert": PrefetchSpec("protbert", "Rostlab/prot_bert_bfd", 1_600_000_000),
    "prott5": PrefetchSpec("prott5", "Rostlab/prot_t5_xl_uniref50", 4_000_000_000),
    "esm2": PrefetchSpec("esm2", "facebook/esm2_t36_3B_UR50D", 9_000_000_000),
    "ankh": PrefetchSpec("ankh", "ElnaggarLab/ankh-large", 2_500_000_000),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefetch official benchmark backbones into HF cache")
    parser.add_argument("--models", default="all", help="Comma-separated model keys or 'all'.")
    parser.add_argument("--cache-dir", default=str(HF_CACHE_DIR))
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--sleep-seconds", type=int, default=15)
    parser.add_argument("--disable-xet", action="store_true")
    parser.add_argument(
        "--auth-mode",
        choices=["auto", "required", "disabled"],
        default="auto",
        help=(
            "HF auth behavior: auto=use HF_TOKEN/HUGGINGFACE_HUB_TOKEN if present else no token; "
            "required=force authenticated download; disabled=never send token"
        ),
    )
    return parser.parse_args()


def expand_models(value: str, allowed: Iterable[str]) -> List[str]:
    if value == "all":
        return list(allowed)
    return [item.strip() for item in value.split(",") if item.strip()]


def configure_environment(cache_dir: Path, disable_xet: bool) -> None:
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir)
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    if disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"


def resolve_hf_token(auth_mode: str):
    if auth_mode == "disabled":
        return False
    if auth_mode == "required":
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        return token if token else True

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    return token if token else False


def repo_cache_dir(cache_dir: Path, repo_id: str) -> Path:
    return cache_dir / f"models--{repo_id.replace('/', '--')}"


def has_incomplete_files(path: Path) -> bool:
    return path.exists() and any(path.rglob("*.incomplete"))


def has_weight_files(path: Path) -> bool:
    patterns = (
        "pytorch_model.bin",
        "pytorch_model-*.bin",
        "model.safetensors",
        "model-*.safetensors",
        "pytorch_model.bin.index.json",
        "model.safetensors.index.json",
    )
    return any(any(path.rglob(pattern)) for pattern in patterns)


def total_file_bytes(path: Path, include_incomplete: bool = True) -> int:
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            if not include_incomplete and file_path.name.endswith(".incomplete"):
                continue
            total += file_path.stat().st_size
    return total


def cache_looks_complete(spec: PrefetchSpec, cache_dir: Path) -> bool:
    target_dir = repo_cache_dir(cache_dir, spec.repo_id)
    if not target_dir.exists():
        return False
    if not has_weight_files(target_dir):
        return False
    return total_file_bytes(target_dir, include_incomplete=False) >= spec.min_complete_bytes


def remove_incomplete_files(path: Path) -> None:
    for file_path in path.rglob("*.incomplete"):
        file_path.unlink(missing_ok=True)


def prefetch_one(spec: PrefetchSpec, cache_dir: Path, max_workers: int, retries: int, sleep_seconds: int, hf_token) -> None:
    target_dir = repo_cache_dir(cache_dir, spec.repo_id)
    last_error = None
    if cache_looks_complete(spec, cache_dir):
        remove_incomplete_files(target_dir)
        size_gb = total_file_bytes(target_dir, include_incomplete=False) / (1024 ** 3)
        print(f"[prefetch] skip {spec.name}: cache already complete ({size_gb:.2f} GiB)", flush=True)
        return
    print(f"[prefetch] start {spec.name}: {spec.repo_id}", flush=True)
    for attempt in range(1, retries + 1):
        try:
            snapshot_path = snapshot_download(
                repo_id=spec.repo_id,
                cache_dir=str(cache_dir),
                local_files_only=False,
                max_workers=max_workers,
                token=hf_token,
            )
            if has_incomplete_files(target_dir):
                raise RuntimeError(f"cache still contains .incomplete files: {target_dir}")
            print(f"[prefetch] done {spec.name}: {snapshot_path}", flush=True)
            return
        except Exception as exc:
            last_error = exc
            print(
                f"[prefetch] attempt {attempt}/{retries} failed for {spec.name}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            if attempt == retries:
                break
            time.sleep(sleep_seconds)
    raise RuntimeError(f"failed to prefetch {spec.repo_id}: {last_error}")


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    configure_environment(cache_dir, args.disable_xet)
    hf_token = resolve_hf_token(args.auth_mode)

    models = expand_models(args.models, PREFETCH_SPECS.keys())
    invalid = [name for name in models if name not in PREFETCH_SPECS]
    if invalid:
        raise ValueError(f"unknown model keys: {', '.join(invalid)}")

    print(f"[prefetch] cache_dir={cache_dir}", flush=True)
    print(f"[prefetch] models={','.join(models)}", flush=True)
    print(f"[prefetch] max_workers={args.max_workers} retries={args.retries}", flush=True)
    print(f"[prefetch] disable_xet={os.environ.get('HF_HUB_DISABLE_XET', '0')}", flush=True)
    print(f"[prefetch] auth_mode={args.auth_mode}", flush=True)

    for model_name in models:
        prefetch_one(
            PREFETCH_SPECS[model_name],
            cache_dir=cache_dir,
            max_workers=args.max_workers,
            retries=args.retries,
            sleep_seconds=args.sleep_seconds,
            hf_token=hf_token,
        )

    print("[prefetch] all requested backbones are cached", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())