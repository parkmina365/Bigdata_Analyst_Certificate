
# 데이터설명 : 고객의 신상정보 데이터를 통한 회사 서비스 이탈 예측(종속변수 : Exited)
# 데이터출처 : https://www.kaggle.com/shubh0799/churn-modelling
# 문제타입 : 분류유형
# 평가지표 : f1-score

import pandas as pd 
trainData = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/churn/train.csv')
testData = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/churn/test.csv')
subData = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/churn/submission.csv',index_col=0)


# --------------------------------- 
# 1. 데이터 파악
# 1-1. 결측값 파악: 없음
print(trainData.isnull().sum().sum(), testData.isnull().sum().sum())

# 1-2. 데이터 타입 파악: 독립변수 3개 -> get_dummies 필요
object_cols = trainData.select_dtypes('object').columns
object_cols_test = testData.select_dtypes('object').columns
print(object_cols, object_cols_test)

# 1-3. 제외 열 파악: RowNumber, CustomerId, Surname 제외 필요
print(trainData.columns)
exclude_cols = ['RowNumber', 'CustomerId', 'Surname']
trainData.drop(columns=exclude_cols, inplace=True)
testData.drop(columns=exclude_cols, inplace=True)

# 1-4. X,y 정의하기
X = trainData.drop(columns=['Exited'])
y = trainData['Exited']
print(X.shape, testData.shape, y.shape)


# --------------------------------- 
# 2. 전처리
# get_dummies
X = pd.get_dummies(X)
testData = pd.get_dummies(testData)

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
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=0)


# --------------------------------- 
# 4. modeling, 학습, 예측, 평가
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(f1_score(rf.predict(X_test),y_test))
print(accuracy_score(rf.predict(X_test),y_test))
print(roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))
      
      
# --------------------------------- 
# 5. 제출
rf = RandomForestClassifier()
rf.fit(X, y)
subData['0'] = rf.predict(testData)
subData.to_csv('수험번호.csv', index=False)
      
