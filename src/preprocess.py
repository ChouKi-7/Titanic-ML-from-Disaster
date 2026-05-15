import pandas as pd

def preprocess(train: pd.DataFrame, test: pd.DataFrame):
    # 1. Embarked
    train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

    # 2. Fare
    test['Fare'].fillna(test['Fare'].median(), inplace=True)

    # 3. Age
    for df in [train, test]:
        median_age = df.groupby(['Pclass', 'Sex'])['Age'].transform('median')
        df['Age'] = df['Age'].fillna(median_age)

    # 4. Cabin
    for df in [train, test]:
        df['Has_Cabin'] = df['Cabin'].notna().astype(int)
        df.drop('Cabin', axis=1, inplace=True)

    return train, test