
# 데이터설명 : 자동차 보험 가입 예측(종속변수 : Response)
# 데이터출처 : https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction
# 문제타입 : 분류유형
# 평가지표 : f1-score

import pandas as pd
trainData = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/insurance/train.csv')
testData = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/insurance/test.csv')
subData = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/insurance/submission.csv')


# --------------------------------- 
# 1. 데이터 파악
# 1-1. 결측값 파악: 없음
print(trainData.isnull().sum().sum(), testData.isnull().sum().sum())

# 1-2. object 열 파악: 독립변수 3개 -> pd.get_dummies 필요
print(trainData.select_dtypes('object').columns)
print(testData.select_dtypes('object').columns)

# 1-3. 제거 열 파악: ['id', 'Region_Code', 'Policy_Sales_Channel'] 제외
print(trainData.columns)
print(testData.columns)
exclude_cols = ['id', 'Region_Code', 'Policy_Sales_Channel']
trainData.drop(columns=exclude_cols, inplace=True)
testData.drop(columns=exclude_cols, inplace=True)

# 1-4. X, y 정의하기
X = trainData.drop(columns=['Response'])
y = trainData['Response']
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)


# --------------------------------- 
# 4. modeling, 학습, 예측, 평가
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(f1_score(rf.predict(X_test), y_test))                   # 0.2104984093319194
print(accuracy_score(rf.predict(X_test), y_test))             # 0.8534881432647841
print(roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))   # 0.8101005272724551


# --------------------------------- 
# 5. 제출
# rf = RandomForestClassifier()
# rf.fit(X, y)
subData['0'] = rf.predict(testData)
subData.to_csv('수험번호.csv', index=False)  # 소요시간: 35.9초

