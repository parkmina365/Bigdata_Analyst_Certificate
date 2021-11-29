
# 데이터설명 : 대학원 입학 가능성 예측문제(종속변수 : price)
# 데이터출처 : https://www.kaggle.com/mysarahmadbhat/hyundai-used-car-listing
# 문제타입 : 회귀유형
# 평가지표 : r2 score

import pandas as pd 
trainData = pd.read_csv('c:/workspace/cakd3/수업/dataset/빅분기/trainData8_2.csv')
testData = pd.read_csv('c:/workspace/cakd3/수업/dataset/빅분기/testData8_2.csv')
subData = pd.read_csv('c:/workspace/cakd3/수업/dataset/빅분기/subData8_2.csv')

# --------------------------------- 
# 1. 데이터 파악
# 1-1. 결측값 확인: 없음
print(trainData.isnull().sum().sum(), testData.isnull().sum().sum())

# 1-2. object 열 확인: 독립변수 3개 -> pd.get_dummies 필요
print(trainData.select_dtypes('object').columns)
print(testData.select_dtypes('object').columns)

# 1-3. 제거 열 파악: 'model': 16가지 종류
print(trainData.columns)
print(testData.columns)
print(len(trainData.model.unique()))
trainData.drop(columns=['model'], inplace=True)
testData.drop(columns=['model'], inplace=True)

# 1-4. X,y 정의하기
X = trainData.drop(columns=['price'])
y = trainData['price']
print(X.shape, y.shape, testData.shape)


# --------------------------------- 
# 2. 전처리
# 2-1. pd.get_dummies
print(trainData.fuelType.value_counts())
print(testData.fuelType.value_counts())
X = pd.get_dummies(X)
testData = pd.get_dummies(testData)

print(testData.columns)
X['fuelType_Other'] = 0

# 2-2. Scaling
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
print(r2_score(y_test, rf.predict(X_test)))


# --------------------------------- 
# 5. 제출
rf = RandomForestRegressor()
rf.fit(X, y)
subData['0'] = rf.predict(testData)
subData.to_csv('수험번호.csv', index=False)

  
