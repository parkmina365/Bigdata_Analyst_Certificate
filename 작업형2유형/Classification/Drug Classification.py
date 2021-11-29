
# 데이터설명 : 투약하는 약을 분류(종속변수 : Drug)
# 데이터출처 : https://www.kaggle.com/prathamtripathi/drug-classification
# 문제타입 : 분류유형
# 평가지표 : accuracy

import pandas as pd 
trainData= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/drug/train.csv')
testData= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/drug/test.csv')
subData= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/drug/submission.csv')


# ---------------------------------
# 1. 데이터 파악
# 1-1. 결측값 파악: 없음
print(trainData.isnull().sum().sum(), testData.isnull().sum().sum())

# 1-2. object 열 파악: 독립변수 3개 -> pd.get_dummies 필요
print(trainData.select_dtypes('object').columns)
print(testData.select_dtypes('object').columns)

# 1_3. 제외 열 파악: 없음
print(trainData.columns)
print(testData.columns)

# 1-4. X, y 분리
# y: 다중분류(range(0,5))
X = trainData.drop(columns=['Drug'])
y = trainData['Drug']
print(X.shape, y.shape, testData.shape)
print(y.unique())


# --------------------------------- 
# 2. 전처리
# 2_1. get_dummies
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
# y가 다중분류이기에 f1_score 사용 불가
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(accuracy_score(rf.predict(X_test), y_test))   # 0.96875

      
# --------------------------------- 
# 5. 제출
rf = RandomForestClassifier()
rf.fit(X, y)
subData['0'] = rf.predict(testData)
subData.to_csv('수험번호.csv', index=False)  # 소요시간: 0.4초

