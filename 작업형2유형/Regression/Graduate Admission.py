
# 데이터설명 : 대학원 입학 가능성 예측문제(종속변수 : Chance of Admit)
# 데이터출처 : https://www.kaggle.com/mohansacharya/graduate-admissions
# 문제타입 : 회귀유형
# 평가지표 : r2 score

import pandas as pd 
trainData= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/admission/train.csv')
testData= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/admission/test.csv')
subData= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/admission/submission.csv')

# --------------------------------- 
# 1. 데이터 파악
# 1-1. 결측값 파악: None
print(trainData.isnull().sum().sum(), testData.isnull().sum().sum())

# 1-2. object 열 파악: None
print(trainData.select_dtypes('object').columns)
print(testData.select_dtypes('object').columns)

# 1-3. 제외 열 파악: ['Serial No.']
print(trainData.columns)
print(testData.columns)
trainData.drop(columns=['Serial No.'], inplace=True)
testData.drop(columns=['Serial No.'], inplace=True)

# 1-4. X, y 정의하기
X = trainData.drop(columns=['Chance of Admit'])
y = trainData['Chance of Admit']
print(X.shape, y.shape, testData.shape)


# --------------------------------- 
# 2. 전처리
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
st = StackingRegressor([('gb', gb), ('lr', lr)])
vt = VotingRegressor([('rf', rf), ('lr', lr)])

for i in [rf, gb, ab, bg, dt, lr, st, vt]:
    i.fit(X_train, y_train)
    print(i.__class__.__name__, r2_score(i.predict(X_test), y_test))
    
# RandomForestRegressor 0.7214050581297056
# GradientBoostingRegressor 0.7275072304922003
# AdaBoostRegressor 0.6515247728629934
# BaggingRegressor 0.6901237973470049
# DecisionTreeRegressor 0.5440878612135013
# LinearRegression 0.7529249518456086
# StackingRegressor 0.7484764470365417
# VotingRegressor 0.7437236342975894

# --------------------------------- 
# 5. 제출
lr = LinearRegression()
lr.fit(X, y)
subData['0'] = lr.predict(testData)
subData.to_csv('수험번호.csv', index=False)

  
