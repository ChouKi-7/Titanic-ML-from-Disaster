import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from preprocess import preprocess

train = pd.read_csv('./data/raw/train.csv')
test = pd.read_csv('./data/raw/test.csv')

train, test = preprocess(train, test)

# カテゴリ変数のエンコーディング
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})

train = pd.get_dummies(train, columns=['Embarked', 'Title'])
test = pd.get_dummies(test, columns=['Embarked', 'Title'])

# モデルに使えない列を削除
drop_cols = ['Name', 'Ticket', 'PassengerId']
train.drop(columns=drop_cols, inplace=True)
test.drop(columns=drop_cols, inplace=True)

X = train.drop('Survived', axis=1)
y = train['Survived']

model = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"ベースライン 交差検証スコア：{scores}")
print(f"平均精度：{scores.mean():.4f} ± {scores.std():.4f}")