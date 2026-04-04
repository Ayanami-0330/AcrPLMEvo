# AcrPLMEvo：Anti-CRISPR 大模型基准

这是一个面向审稿与复现的干净基准工程，用于比较参数高效微调在 Anti-CRISPR 二分类任务上的效果。

## 项目做了什么

本项目在统一协议下比较：

- 4 个蛋白语言模型 backbone：`protbert`、`prott5`、`esm2`、`ankh`
- 5 条训练策略（native + 两类特征设定下的 LoRA/DoRA）
- 2 条预测输入列（`lm_only`、`lm_pssm`）
- 5 个随机种子（`11,22,33,44,55`）

主实验总运行数：`5 策略 x 2 输入列 x 4 backbone x 5 seeds = 200`。

## 数据划分与特征流

固定数据来自 `data/anticrispr_benchmarks`：

- 训练池：`anticrispr_binary.train.csv`（1107 条）
- 测试集：`anticrispr_binary.test.csv`（286 条）
- 每个 seed 在训练池上做分层 `9:1` 划分得到 train/valid
- `lm_only`：仅使用 LM 表征
- `lm_pssm`：在 PCA/小头前将 LM 与外部 PSSM 特征拼接

默认 PSSM 缓存：

- `/home/nemophila/data/pssm_work/features/pssm_features_1110.parquet`
- 若不存在则回退到 `/home/nemophila/data/pssm_work/features/pssm_features_1110.csv`

仓库不跟踪 PSSM 特征大文件。若需按当前实现复现缓存，可使用 `src/acrplmevo/pssm_pipeline` 下的独立流程脚本：

- 第 1 步（准备 FASTA 与样本清单）：`python src/acrplmevo/pssm_pipeline/prepare_fasta.py ...`
- 第 2 步（从 PSI-BLAST 的 PSSM 文本提取 310/710/1110 特征）：`python src/acrplmevo/pssm_pipeline/extract_features.py ...`
- 第 3 步（构建 `pssm_features_{variant}.parquet/csv` 缓存）：`python src/acrplmevo/pssm_pipeline/build_feature_cache.py ...`

要先得到每个样本的 `.pssm` 文件，需要外部工具和数据库（BLAST+/PSI-BLAST + UniRef）：

- 安装 BLAST+（确保有 `psiblast` 命令）。
- 准备可搜索的 UniRef 数据库（例如 UniRef50 FASTA 后用 `makeblastdb` 建库）。
- 对每个样本 FASTA 运行 PSI-BLAST，输出 ASCII PSSM。
- 再执行上面的第 2、3 步，构建 `pssm_features_1110.parquet/csv`。

主实验与补充复评所用的融合与阈值工具，已内置在 `src/acrplmevo/pssm_fusion.py`。

## 运行前最小检查清单

在新机器上运行前，请先确认：

- 已在 Python 环境安装依赖（可用 `pip install -e .`）。
- 若需自定义模型缓存路径：`export ACRPLMEVO_HF_CACHE_DIR=/path/to/hf_cache`。
- 仅当本地缓存已完整时再启用离线：`export ACRPLMEVO_OFFLINE=1`。
- 若运行 `lm_pssm`，需在 `PSSM_WORK_ROOT/features` 下提供：
  - `pssm_features_1110.parquet` 或
  - `pssm_features_1110.csv`
- 若尚无 PSSM 缓存，先用 `src/acrplmevo/pssm_pipeline/*.py` 生成。

## 避免阻塞的建议

为避免新机器复现中断：

- `scripts/prefetch_backbones.py` 默认 `--auth-mode auto`：
  - 若存在 `HF_TOKEN` 或 `HUGGINGFACE_HUB_TOKEN` 则使用 token；
  - 若不存在则按公开模型无 token 下载。
- 如果没有 HF token，可显式使用：
  - `python scripts/prefetch_backbones.py --models all --auth-mode disabled`
- 如果模型访问需要认证，请设置 token 后使用：
  - `export HF_TOKEN=...`
  - `python scripts/prefetch_backbones.py --models all --auth-mode required`
- 首次建议在线模式（`export ACRPLMEVO_OFFLINE=0`），待缓存完整后再启用离线。

## 主实验设置

主入口为 `scripts/main.py`，`run-10` 会执行完整 10 组设计。

- 阶段 1：native（主干冻结，不用 LoRA/DoRA）
- 阶段 2：LoRA/DoRA 微调并保存 adapter
- 阶段 3：对 S2-S5 全部单元执行冻结 adapter 特征提取评估
- 阶段 4：重建统一汇总表

面向审稿的最终结果表仅由冻结评估注册表重建：

- `results/experiments_frozen_no_lora.csv`（A/B：native 冻结主干）
- `results/experiments_frozen.csv`（C/D：微调后主干 + 同变体输入）
- `results/experiments_frozen_cross_variant.csv`（E/F：微调后主干 + 跨变体输入）

adapter 微调日志可在本地运行时生成，但不属于面向审稿的最终结果表。

## 为什么是“10组实验”但“6类结果”

`run-10` 在执行层面会跑满 G01-G10 共 10 个单元，因为 S2-S5 按 adapter 类型与输入配对被展开：

- S1：native 冻结主干（2 组）
- S2-S5：LoRA/DoRA x 同变体/跨变体输入（8 组）

但在面向审稿的类别汇总中，LoRA 与 DoRA 不作为额外类别轴拆分，而是按以下两个维度组织：

- 主干状态（`native`、`tuned_lm_only`、`tuned_lm_pssm`）
- 预测时外部特征（`lm_only`、`lm_pssm`）

因此类别数为 `3 x 2 = 6`。对于适用类别，LoRA 与 DoRA 的结果仍保留在同一类别下的 adapter 行中。

## 阈值怎么选

### 内部测试集阈值

在主实验中：

- 阈值在 valid 上通过 `find_best_threshold` 选择（以 F1 为导向）
- 选定阈值原样应用到固定内部 test 集

### 外部测试集阈值

外部验证资源在 `../external_validation`。

当前外部验证 notebook（`external_validation/results/external_validation_esm2_dora_pssm_seed44.ipynb`）采用：

- 在 valid 上按高召回策略选阈值
- 目标召回 `Recall = 0.95`
- 将该阈值应用到外部样本 `new_case.csv`

## 最小运行测试

最小演示 notebook 位于：

- `notebooks/AcrPLMEvo.demo.ipynb`

该 notebook 包含：

- 使用 `lm-hf` 环境进行约 2 分钟的最小冒烟测试
- 自动清理临时产物
- 自动重建 summary 表

## 目录结构（精简视图）

```text
llm_lora_experiments/
  README.md
  README_CN.md
  data/
    anticrispr_benchmarks/
  src/
    acrplmevo/
      pssm_fusion.py
      pssm_pipeline/
        prepare_fasta.py
        extract_features.py
        build_feature_cache.py
  scripts/
    main.py
    run_full_benchmark.sh
    prefetch_backbones.py
    pipelines/
      run_phase_a_adapters.sh
      run_supplemental_frozen_eval.sh
    frozen_baseline/
      run_supplemental_frozen_eval.py
  results/
    experiments_frozen_no_lora.csv
    experiments_frozen.csv
    experiments_frozen_cross_variant.csv
    6categories_seedmean_auc_auprc.csv
    6categories_best_single_seed_by_auc_then_auprc.csv
    plots/6category/six_category_mean_std_by_model.csv
    summary_10group_runs.csv
    summary_10group_by_model.csv
```

## 快速开始

在仓库根目录执行：

```bash
python scripts/main.py run-10 \
  --models ankh,esm2,protbert,prott5 \
  --seeds 11,22,33,44,55 \
  --epochs 8 \
  --resume
```

或直接运行封装脚本：

```bash
bash scripts/run_full_benchmark.sh
```
