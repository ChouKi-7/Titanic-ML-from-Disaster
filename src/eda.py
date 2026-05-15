import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

train = pd.read_csv("data/raw/train.csv")
test  = pd.read_csv("data/raw/test.csv")

print(f"训练集: {train.shape}  测试集: {test.shape}")

print(train.head())
print(test.head())
print(train.describe())

print(train.columns.tolist())

missing = pd.DataFrame({
    "train缺失数": train.isnull().sum(),
    "train缺失率": (train.isnull().mean() * 100).round(1),
    "test缺失数":  test.isnull().sum(),
    "test缺失率":  (test.isnull().mean() * 100).round(1),
})
missing[missing["train缺失数"] > 0]
print(missing)