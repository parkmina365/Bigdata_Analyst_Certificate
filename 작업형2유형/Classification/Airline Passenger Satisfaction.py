
# 데이터설명 : 비행탑승 경험 만족도(종속변수 : satisfaction)
# 데이터출처 : https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction?select=train.csv
# 문제타입 : 분류유형
# 평가지표 : accuracy

import pandas as pd
trainData  = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/airline/train.csv')
testData  = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/airline/test.csv')
subData  = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/airline/submission.csv')


# --------------------------------- 
# 1. 데이터 파악
# 1-1. 결측값 파악: ['Arrival Delay in Minutes']
print(trainData.isnull().sum().sum())
print(testData.isnull().sum().sum())

# 1-2. object 열 파악: 독립변수 3개, 종속변수 1개 -> get_dummies, Label Encoding 필요
print(trainData.select_dtypes('object').columns)
print(testData.select_dtypes('object').columns)

# 1-3. 제외 열 파악
print(trainData.columns)
print(testData.columns)
exclude_cols = ['Arrival Delay in Minutes', 'id']
trainData.drop(columns=exclude_cols, inplace=True)
testData.drop(columns=exclude_cols, inplace=True)

# 1-4. X, y 정의하기
X = trainData.drop(columns=['satisfaction'])
y = trainData['satisfaction']
print(X.shape, y.shape, testData.shape)


# --------------------------------- 
# 2. 전처리
# 2-1. label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = pd.Series(le.fit_transform(y))

# 2-2. pd.get_dummies
X = pd.get_dummies(X)
testData = pd.get_dummies(testData)

# 2-3. scaling
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X)
X = pd.DataFrame(ss.transform(X), columns=X.columns)
testData = pd.DataFrame(ss.transform(testData), columns=testData.columns)
print(X.mean().mean(), X.std().mean())
print(testData.mean().mean(), testData.std().mean())


# --------------------------------- 
# 3. train, test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)


# --------------------------------- 
# 4. modeling, 학습, 예측, 평가
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(accuracy_score(rf.predict(X_test), y_test))            # 0.9611428571428572
print(f1_score(rf.predict(X_test), y_test))                  # 0.9543591917479158
print(roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))  # 0.9939107809699655


# ---------------------------------
# 5. 제출
rf = RandomForestClassifier()
rf.fit(X, y)
subData['0'] = rf.predict(testData)
subData.to_csv('수험번호.csv', index=False)  # 소요시간: 19.7초

  
