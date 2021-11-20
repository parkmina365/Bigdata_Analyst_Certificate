 
# 데이터설명 : 수질 음용성 여부(종속변수 : Potablility)
# 데이터출처 : https://www.kaggle.com/adityakadiwal/water-potability
# 문제타입 : 분류유형
# 평가지표 : accuracy

import pandas as pd
trainData  = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/waters/train.csv')
testData  = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/waters/test.csv')
subData  = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/waters/submission.csv')


# --------------------------------- 
# 1. 데이터 파악
# 1-1. 결측값 확인: ['ph', 'Sulfate', 'Trihalomethanes'] -> 결측값 mean으로 대체
print(trainData.isnull().sum(), testData.isnull().sum())
null_cols = trainData.isnull().sum()
print(null_cols[null_cols!=0].index)
trainData.fillna(trainData.mean(), inplace=True)
testData.fillna(testData.mean(), inplace=True)

# 1-2. object 열 파악: None
print(trainData.select_dtypes('object').columns)
print(testData.select_dtypes('object').columns)

# 1-3. 제거 열 파악: None
print(trainData.columns)
print(testData.columns)

# 1-4. X, y 정의하기
X = trainData.drop(columns=['Potability'])
y = trainData['Potability']
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(accuracy_score(rf.predict(X_test), y_test))             # 0.6774809160305344
print(f1_score(rf.predict(X_test), y_test))                   # 0.45307443365695793
print(roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))   # 0.6472273284313727


# --------------------------------- 
# 5. 제출
rf = RandomForestClassifier()
rf.fit(X, y)
subData['0'] = rf.predict(testData)
subData.to_csv('수험번호.csv', index=False)  # 소요시간: 1.4초

  
