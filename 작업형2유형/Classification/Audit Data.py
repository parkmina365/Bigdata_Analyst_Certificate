
# 데이터설명 : 사기회사 분류(종속변수 : Risk)
# 데이터출처 : https://www.kaggle.com/sid321axn/audit-data
# 문제타입 : 분류유형
# 평가지표 : f1_score

import pandas as pd 
trainData= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/audit/train.csv')
testData= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/audit/test.csv')
subData= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/audit/submission.csv')


# ---------------------------------
# 1. 데이터 파악
# 1-1. 결측값 파악: 'Money_Value': 1개 -> dropna
print(trainData.isnull().sum().sum(), testData.isnull().sum().sum())
print(trainData.isnull().sum().sum()/len(trainData)) # null 값의 비중이 매우 낮은편
trainData.dropna(axis=1, inplace=True)
testData.dropna(axis=1, inplace=True)
print(trainData.isnull().sum().sum(), testData.isnull().sum().sum())

# 1-2. object 열 파악: 독립변수 1개. 'LOCATION_ID': int로 보이나 object -> 문자가 포함됨
# BUT 'LOCATION_ID'는 우편번호의 개념이므로 결측 처리 불가. 삭제 필요
print(trainData.select_dtypes('object').columns)
print(testData.select_dtypes('object').columns)
print(trainData.LOCATION_ID.unique())
print(testData.LOCATION_ID.unique())

# 1-3. 제외 열 파악
# 'LOCATION_ID': 문자열 섞임. 우편번호 개념이므로 의미 없는 값
# 'PROB', 'Prob': 거의 같은 값. 한 열만 사용해도 됨
print(trainData.columns.sort_values())
print(testData.columns.sort_values())
X = trainData[['PROB', 'Prob']]
print(len(X[X.Prob==X.PROB])/len(X))  # 0.917741935483871
drop_col = ['LOCATION_ID', 'Prob']
trainData.drop(columns=drop_col, inplace=True)
testData.drop(columns=drop_col, inplace=True)

# 1-4. X, y 정의
X = trainData.drop(columns=['Risk'])
y = trainData['Risk']
print(X.shape, y.shape, testData.shape)


# --------------------------------- 
# 2. 전처리
# Scaling
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X)

X = pd.DataFrame(ss.transform(X), columns=X.columns)
testData = pd.DataFrame(ss.transform(testData), columns=testData.columns)

print(X.mean().mean(), X.std().mean())
print(testData.mean().mean(), testData.std().mean()) # testData의 std 불균형(0.7565735195776944)


# --------------------------------- 
# 3. train, test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=0)


# --------------------------------- 
# 4. modeling, 학습, 예측, 평가
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(f1_score(y_test, rf.predict(X_test)))
print(roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))
print(accuracy_score(y_test, rf.predict(X_test)))

      
# --------------------------------- 
# 5. 제출
rf = RandomForestClassifier()
rf.fit(X, y)
subData['0'] = rf.predict(testData)
subData.to_csv('수험번호.csv', index=False)  # 소요시간: 2초

  
