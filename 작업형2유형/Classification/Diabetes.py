
# 데이터설명 : 당뇨여부 판단하기(종속변수 : Outcome)
# 데이터출처 : https://www.kaggle.com/pritsheta/diabetes-dataset
# 문제타입 : 분류유형
# 평가지표 : f1_score

import pandas as pd 
trainData= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/diabetes/train.csv')
testData= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/diabetes/test.csv')
subData= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/diabetes/submission.csv')


# --------------------------------- 
# 1. 데이터 파악
# 1-1. 결측값 파악: 없음
print(trainData.isnull().sum().sum(), testData.isnull().sum().sum())

# 1-2. object 열 파악: 없음
print(trainData.select_dtypes('object').columns)
print(testData.select_dtypes('object').columns)

# 1-3. 제거 열 파악: 없음
print(trainData.columns)
print(testData.columns)

# 1-4. X, y 정의하기
X = trainData.drop(columns=['Outcome'])
y = trainData['Outcome']
print(X.shape, y.shape, testData.shape)


# --------------------------------- 
# 2. 전처리
# Scaling
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
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=0)


# --------------------------------- 
# 4. modeling, 학습, 예측, 평가
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
rf = RandomForestClassifier() 
rf.fit(X_train, y_train)
print(f1_score(y_test, rf.predict(X_test)))                   # 0.6582278481012659
print(accuracy_score(y_test, rf.predict(X_test)))             # 0.7804878048780488
print(roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))   # 0.8325581395348838


# ---------------------------------
# 5. 제출
rf = RandomForestClassifier() 
rf.fit(X, y)

subData['0'] = rf.predict(testData)
subData['prob'] = rf.predict_proba(testData)[:,1]
subData.rename(columns={'0':'predict'}, inplace=True)
subData.to_csv('수험번호.csv', index=False)                   # 소요시간: 2초

  
