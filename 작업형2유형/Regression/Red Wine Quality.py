
# 데이터설명 : 레드 와인 퀄리티 예측문제(종속변수 : quality)
# 데이터출처 : https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009
# 문제타입 : 회귀유형
# 평가지표 : r2 score

import pandas as pd 
trainData= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/redwine/train.csv')
testData= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/redwine/test.csv')
subData= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/redwine/submission.csv')


# --------------------------------- 
# 1. 데이터 파악
# 1-1. 결측값 파악: None
print(trainData.isnull().sum(), testData.isnull().sum())

# 1-2. object 열 파악: None
print(trainData.select_dtypes('object').columns)
print(testData.select_dtypes('object').columns)

# 1-3. 제거 열 파악: None
print(trainData.columns)
print(testData.columns)

# 1-4. X, y 정의하기
X = trainData.drop(columns=['quality'])
y = trainData['quality']
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
st = StackingRegressor([('bg', bg), ('rf', rf)])
vt = VotingRegressor([('bg', bg), ('rf', rf)])

for i in [rf, gb, ab, bg, dt, lr, st, vt]:
    i.fit(X_train, y_train)
    print(i.__class__.__name__, r2_score(y_test, i.predict(X_test)))
   
# RandomForestRegressor 0.46939813216797166
# GradientBoostingRegressor 0.41327596143189793
# AdaBoostRegressor 0.37020582160683524
# BaggingRegressor 0.40984671354732805
# DecisionTreeRegressor 0.011868053860999539
# LinearRegression 0.4069767169663012
# SVR 0.40333253568677163
# KNeighborsRegressor 0.2703724572417312
# XGBRegressor 0.3995088697553514
# StackingRegressor 0.5024251302163755
# VotingRegressor 0.4858009219508218


# --------------------------------- 
# 5. 제출
st = StackingRegressor([('bg', BaggingRegressor()), ('rf', RandomForestRegressor())])
st.fit(X, y)
subData['0'] = st.predict(testData)
subData.to_csv('수험번호.csv', index=False)

 
