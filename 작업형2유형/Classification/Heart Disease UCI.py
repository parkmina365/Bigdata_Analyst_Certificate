
# 데이터설명 : 심장질환예측(종속변수 : target)
# 데이터출처 : https://www.kaggle.com/ronitf/heart-disease-uci
# 문제타입 : 분류유형
# 평가지표 : f1-score

import pandas as pd
trainData = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/heart/train.csv')
testData = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/heart/test.csv')
subData = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/heart/submission.csv')


# --------------------------------- 
# 1. 데이터 파악
# 1-1. 결측값 파악: None
print(trainData.isnull().sum().sum(), testData.isnull().sum().sum())

# 1-2. object 열 파악: None
print(trainData.select_dtypes('object').columns)
print(testData.select_dtypes('object').columns)

# 1-3. 제거 열 파악: None
print(trainData.columns)
print(testData.columns)

# 1-4. X, y 정의하기
X = trainData.drop(columns=['target'])
y = trainData['target']
print(X.shape, y.shape, testData.shape)


# --------------------------------- 
# 2. 전처리
# scaling
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
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(f1_score(rf.predict(X_test), y_test))                  # 0.8363636363636364
print(accuracy_score(rf.predict(X_test), y_test))            # 0.8163265306122449
print(roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))  # 0.9099326599326599


# --------------------------------- 
# 5. 제출
rf = RandomForestClassifier()
rf.fit(X, y)
subData['0'] = rf.predict(testData)
subData.to_csv('수험번호.csv', index=False)  # 소요시간: 0.3초

  
