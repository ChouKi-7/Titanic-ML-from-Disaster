import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

train = pd.read_csv("data/raw/train.csv")
test  = pd.read_csv("data/raw/test.csv")

print(f"训练集: {train.shape}  测试集: {test.shape}")

print(train.head().T)
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

'''
列名	含义
PassengerId	乘客编号，纯流水号，无实际意义
Survived	目标变量,0 = 遇难,1 = 生还
Pclass	船票等级,1 = 一等舱(最贵),2 = 二等舱,3 = 三等舱（最便宜）
Name	乘客姓名，包含称谓(Mr/Mrs/Miss等),可以从中提取有用特征
Sex	性别,male / female
Age	年龄，有缺失值(~20%)
SibSp	同船的兄弟姐妹/配偶数量
Parch	同船的父母/子女数量
Ticket	船票编号，格式不统一，一般用处不大
Fare	票价（英镑）
Cabin	舱位号，缺失率极高（~78%），格式如 C85,首字母代表甲板层
Embarked	登船港口,S = 南安普顿,C = 瑟堡,Q = 皇后镇
'''