
# 데이터설명 : 현대차 스펙에 따른 가격 예측문제(종속변수 : price)
# 데이터출처 : https://www.kaggle.com/mysarahmadbhat/hyundai-used-car-listing
# 문제타입 : 회귀유형
# 평가지표 : r2 score

import pandas as pd 
trainData= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/hyundai/train.csv')
testData= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/hyundai/test.csv')
subData= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/hyundai/submission.csv')

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
obj_cols = trainData.select_dtypes('object').columns.to_list()
obj_cols_t = testData.select_dtypes('object').columns.to_list()
for i,v in enumerate(obj_cols):
    print(v, trainData[obj_cols[i]].nunique())
for i,v in enumerate(obj_cols_t):
    print(v, testData[obj_cols_t[i]].nunique())         # object 열의 nunique가 다름

X = pd.get_dummies(X)
testData = pd.get_dummies(testData)
print(len(trainData.columns) == len(testData.columns))  # False
X['fuelType_Other'] = 0                                 # column수가 같도록 조정

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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor,\
    StackingRegressor, VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor 
from sklearn.metrics import r2_score

rf = RandomForestRegressor()
gb = GradientBoostingRegressor()
ab = AdaBoostRegressor()
bg = BaggingRegressor()
dt = DecisionTreeRegressor()
lr = LinearRegression()
svr = SVR()
kr = KNeighborsRegressor()
xgb = XGBRegressor()

for i in [rf, gb, ab, bg, dt, lr, svr, kr, xgb, st, vt]:
    i.fit(X_train, y_train)
    print(i.__class__.__name__, r2_score(y_test, i.predict(X_test)))

# RandomForestRegressor 0.9563289186250624
# GradientBoostingRegressor 0.9495437864954008
# AdaBoostRegressor 0.839433820709858
# BaggingRegressor 0.9512617990493079
# DecisionTreeRegressor 0.9324377683354683
# LinearRegression 0.8088603475940102
# SVR 0.03078458908587678
# KNeighborsRegressor 0.9507689168225735
# XGBRegressor 0.9582664573565598
# StackingRegressor 0.9616486337844243
# VotingRegressor 0.9612288574922806

# --------------------------------- 
# 5. 제출
vt = VotingRegressor([('bg', BaggingRegressor()), ('rf', RandomForestRegressor())])
vt.fit(X, y)
subData['0'] = vt.predict(testData)
subData.to_csv('수험번호.csv', index=False)               # 소요시간: 8.6초

  
