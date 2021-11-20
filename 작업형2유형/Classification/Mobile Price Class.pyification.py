
# 데이터설명 : 핸드폰 가격예측(종속변수 : price_range)
# 데이터출처 : https://www.kaggle.com/iabhishekofficial/mobile-price-classification?select=train.csv
# 문제타입 : 분류유형
# 평가지표 : accuracy

import pandas as pd
trainData  = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/mobile/train.csv')
testData  = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/mobile/test.csv')


# --------------------------------- 
# 1. 데이터 파악
# 1-1. 결측값 파악: None
print(trainData.isnull().sum().sum(), testData.isnull().sum().sum())

# 1-2. object 열 파악: None
print(trainData.select_dtypes('object').columns)
print(testData.select_dtypes('object').columns)

# 1-3. 제외 열 파악: ['id']
print(trainData.columns)
print(testData.columns)
testData.drop(columns=['id'], inplace=True)

# 1-4. X, y 정의하기
X = trainData.drop(columns=['price_range'])
y = trainData['price_range']
print(X.shape, testData.shape, y.shape)


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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)


# --------------------------------- 
# 4. modeling, 학습, 예측, 평가
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(accuracy_score(rf.predict(X_test), y_test))


# --------------------------------- 
# 5. 제출
testData['predict'] = rf.predict(testData)
testData[['predict']].to_csv('수험번호.csv', index=False)

           
