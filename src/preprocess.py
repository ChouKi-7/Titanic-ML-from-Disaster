import pandas as pd

def preprocess(train: pd.DataFrame, test: pd.DataFrame):
    # 1. Embarked
    train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

    # 2. Fare　欠損処理
    test['Fare'].fillna(test['Fare'].median(), inplace=True)

    # 3. Age　欠損処理
    for df in [train, test]:
        median_age = df.groupby(['Pclass', 'Sex'])['Age'].transform('median')
        df['Age'] = df['Age'].fillna(median_age)

    # 4. Title 追加
    for df in [train, test]:
        df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.', expand=False).str.strip()

    for df in [train, test]:
        counts = train['Title'].value_counts()
        rare_titles = counts[counts < 10].index
        df['Title'] = df['Title'].replace(rare_titles, 'Rare')

    # 5. Cabin　
    for df in [train, test]:
        df['Has_Cabin'] = df['Cabin'].notna().astype(int)
        df.drop('Cabin', axis=1, inplace=True)

    return train, test