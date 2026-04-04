import pandas as pd
from pathlib import Path

base = Path(__file__).resolve().parent

# 读取
train_pos = pd.read_csv(base / "train_pos.csv")
train_neg = pd.read_csv(base / "train_neg.csv")
test_pos  = pd.read_csv(base / "test_pos.csv")
test_neg  = pd.read_csv(base / "test_neg.csv")

# 合并
train_df = pd.concat([train_pos, train_neg], ignore_index=True)
test_df  = pd.concat([test_pos, test_neg], ignore_index=True)

# 打乱
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df  = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 保存
train_df.to_csv(base / "anticrispr_binary.train.csv", index=False)
test_df.to_csv(base / "anticrispr_binary.test.csv", index=False)

print("Done.")
