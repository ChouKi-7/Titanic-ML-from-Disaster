# Titanic-ML-from-Disaster
Kaggle: Titanic-ML from Disaster 


## data set
列名	含义（中文）	意味（日本語）
PassengerId	 乘客编号，纯流水号，无实际意义	 乗客番号、連番であり実際の意味はない
Survived	目标变量，0 = 遇难，1 = 生还	目的変数、0 = 死亡、1 = 生存
Pclass	船票等级，1 = 一等舱（最贵），2 = 二等舱，3 = 三等舱（最便宜）	チケットクラス、1 = 一等船室（最高級）、2 = 二等船室、3 = 三等船室（最安値）
Name	乘客姓名，包含称谓（Mr/Mrs/Miss等），可从中提取有用特征	乗客氏名、敬称（Mr/Mrs/Missなど）を含み、特徴量として抽出可能
Sex	性别，male / female	性別、male / female
Age	年龄，有缺失值（~20%）	年齢、欠損値あり（約20%）
SibSp	同船的兄弟姐妹/配偶数量	同乗している兄弟姉妹・配偶者の人数
Parch	同船的父母/子女数量	同乗している親・子供の人数
Ticket	船票编号，格式不统一，一般用处不大	チケット番号、形式が統一されておらず一般的に有用性は低い
Fare	票价（英镑）	運賃（ポンド）
Cabin	舱位号，缺失率极高（~78%），首字母代表甲板层	客室番号、欠損率が非常に高く（約78%）、頭文字がデッキ階を表す
Embarked	登船港口，S = 南安普顿，C = 瑟堡，Q = 皇后镇	乗船港、S = サウサンプトン、C = シェルブール、Q = クイーンズタウン

## 進捗記録
### データ前処理（preprocess.py）
Embarked：欠損値を最頻値で補完
Fare：欠損値を中央値で補完（テストデータのみ）
Age：客室クラスと性別ごとの中央値で補完
Cabin：有無を Has_Cabin（0/1）に変換し元の列を削除

### ベースラインモデル（baseline.py）
モデル：Random Forest（n_estimators=100, random_state=42）
エンコーディング：Sex を 0/1 に変換、Embarked を One-Hot 化
削除列：Name、Ticket、PassengerId
ベースライン精度：0.8081 ± 0.0395

### 特徴量エンジニアリング
Title（称号）の抽出 採用
Name 列から称号（Mr / Mrs / Miss / Master）を抽出
出現回数10未満の称号は Rare にまとめた
精度：0.8103 ± 0.0356（ベースラインより改善）

FamilySize / IsAlone 不採用
SibSp + Parch + 1 で家族人数を算出、単独乗船フラグも作成
両パターン試したが精度が低下したため削除
SibSp・Parch を残した場合：0.8058
SibSp・Parch を削除した場合：0.8047

FareBand 不採用
Fare を4区間に分箱（pd.qcut）
精度：0.8036（低下したため削除）

AgeBand 不採用
Age を年齢層ごとに分箱（012 / 1218 / 1835 / 3560 / 60~100）
精度：0.8025（低下したため削除）

現在の最高精度：0.8103

次のステップ候補
1.ハイパーパラメータ調整 — Random Forest の max_depth や min_samples_split を最適化する
2.モデル変更 — XGBoost や Logistic Regression など別のアルゴリズムを試す