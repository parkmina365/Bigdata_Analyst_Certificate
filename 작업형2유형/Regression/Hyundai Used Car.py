
# 데이터설명 : 대학원 입학 가능성 예측문제(종속변수 : price)
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
print(trainData.fuelType.value_counts())                # nunique: 3
print(testData.fuelType.value_counts())                 # nunique: 4
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
from sklearn.metrics import r2_score
rf = RandomForestRegressor()
gb = GradientBoostingRegressor()
ab = AdaBoostRegressor()
bg = BaggingRegressor()
dt = DecisionTreeRegressor()
lr = LinearRegression()
st = StackingRegressor([('bg', bg), ('rf', rf)])
vt = VotingRegressor([('bg', bg), ('rf', rf)])

for i in [rf, gb, ab, bg, dt, lr, st, vt]:
    i.fit(X_train, y_train)
    print(i.__class__.__name__, r2_score(i.predict(X_test), y_test))

# RandomForestRegressor 0.9518167304061246
# GradientBoostingRegressor 0.9443280630183511
# AdaBoostRegressor 0.6900659480140665
# BaggingRegressor 0.949578129206992
# DecisionTreeRegressor 0.9289459511671029
# LinearRegression 0.7236097731943167
# StackingRegressor 0.952265383247458
# VotingRegressor 0.952670337306776

# --------------------------------- 
# 5. 제출
vt = VotingRegressor([('bg', BaggingRegressor()), ('rf', RandomForestRegressor())])
vt.fit(X, y)
subData['0'] = vt.predict(testData)
subData.to_csv('수험번호.csv', index=False)               # 소요시간: 3.5초

  
