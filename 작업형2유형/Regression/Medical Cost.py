
# 데이터설명 : 의료비용 예측문제(종속변수 : charges)
# 데이터출처 : https://www.kaggle.com/mirichoi0218/insurance/code
# 문제타입 : 회귀유형
# 평가지표 : r2 score

import pandas as pd 
trainData= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/MedicalCost/train.csv')
testData= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/MedicalCost/test.csv')
subData= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/MedicalCost/submission.csv')


# --------------------------------- 
# 1. 데이터 파악
# 1-1. 결측값 확인: None
print(trainData.isnull().sum(), testData.isnull().sum())

# 1-2. object 열 파악: 독립변수 3개 -> pd.get_dummies
print(trainData.select_dtypes('object').columns)
print(testData.select_dtypes('object').columns)

# 1-3. 제거 열 파악: None
print(trainData.columns)
print(testData.columns)

# 1-4. X, y 정의하기
X = trainData.drop(columns=['charges'])
y = trainData['charges']
print(X.shape, y.shape, testData.shape)


# --------------------------------- 
# 2. 전처리
# 2-1. pd.get_dummies
X = pd.get_dummies(X)
testData = pd.get_dummies(testData)

# 2-2. scaling
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# --------------------------------- 
# 4. modeling, 학습, 예측, 평가
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
print(r2_score(rf.predict(X_test), y_test))  # 0.8445836901512348


# --------------------------------- 
# 5. 제출
rf = RandomForestRegressor()
rf.fit(X, y)
subData['0'] = rf.predict(testData)
subData.to_csv('수험번호.csv', index=False)  # 소요시간: 0.8초

  
