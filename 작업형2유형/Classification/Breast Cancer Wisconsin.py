
# 데이터설명 : 유방암 발생여부 예측(종속변수 : diagnosis)
# 데이터출처 : https://archive.ics.uci.edu/ml/datasets/Breast%20Cancer%20Wisconsin%20(Diagnostic)
# 문제타입 : 분류유형
# 평가지표 : f1-score

import pandas as pd 
trainData = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/cancer/train.csv')
testData = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/cancer/test.csv')
subData = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/cancer/submission.csv')


# --------------------------------- 
# 1. 데이터 파악
# 1-1. 결측값 파악: 없음
print(trainData.isnull().sum().sum(), testData.isnull().sum().sum())

# 1-2. object인 열 파악: trainData의 종속변수 1개 -> Label Encoding 필요
print(trainData.select_dtypes('object').columns)
print(testData.select_dtypes('object').columns)

# 1-3. 제외 열 파악: ['id'] 제외 필요
print(trainData.columns, testData.columns)
exclude_cols = ['id']
trainData.drop(columns=exclude_cols, inplace=True)
testData.drop(columns=exclude_cols, inplace=True)

# 1-4. X,y 정의하기
X = trainData.drop(columns=['diagnosis'])
y = trainData['diagnosis']
print(X.shape, testData.shape, y.shape)


# --------------------------------- 
# 2. 전처리
# 2-1. 종속변수 Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = pd.Series(le.fit_transform(y))

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
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=0)


# --------------------------------- 
# 4. modeling, 학습, 예측, 평가
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(f1_score(rf.predict(X_test), y_test))  # 0.9696969696969697
print(accuracy_score(rf.predict(X_test), y_test))  # 0.978021978021978
print(roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))  # 0.9984520123839009
      
      
# --------------------------------- 
# 5. 제출
rf = RandomForestClassifier()
rf.fit(X, y)
subData['0'] = rf.predict(testData)
subData.to_csv('수험번호.csv', index=False)

      
