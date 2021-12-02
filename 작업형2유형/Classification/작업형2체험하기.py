
# 출력을 원하실 경우 print() 함수 활용
# 예시) print(df.head())

# getcwd(), chdir() 등 작업 폴더 설정 불필요
# 파일 경로 상 내부 드라이브 경로(C: 등) 접근 불가

# 데이터 파일 읽기 예제
import pandas as pd
X_test = pd.read_csv("data/X_test.csv")
X_train = pd.read_csv("data/X_train.csv")
y_train = pd.read_csv("data/y_train.csv")

# 사용자 코딩

# 답안 제출 참고
# 아래 코드 예측변수와 수험번호를 개인별로 변경하여 활용
# pd.DataFrame({'cust_id': X_test.cust_id, 'gender': pred}).to_csv('003000000.csv', index=False)

# ----------------------------
X_test_copy =  X_test.copy()
X_train_copy =  X_train.copy()
y_train_copy =  y_train.copy()

# 1. 데이터 파악
# 1-1. 결측값 파악: 환불금액 결측이 꽤 큼. -> drop
print(X_train.isnull().sum())
print(X_test.isnull().sum())
X_train.drop(columns=['환불금액'], inplace=True)
X_test.drop(columns=['환불금액'], inplace=True)
print(X_train.columns)
print(X_test.columns)
print(X_train.isnull().sum().sum())
print(X_test.isnull().sum().sum())

# 1-2. object 열 파악: ['주구매상품', '주구매지점'] -> nuique가 (42,41), (24,24)개 -> encoding 없이 drop 필요
object_col = X_train.select_dtypes('object').columns
print(X_train.select_dtypes('object').columns)
print(X_test.select_dtypes('object').columns)
for i in object_col:
	print(X_train[i].nunique())
	print(X_test[i].nunique())
	
# 1-3. 제외 열 파악: 환불금액(1-1에서 삭제), object 열 삭제, cust_id를 인덱스로 만들기
X_train.drop(columns=object_col, inplace=True)
X_test.drop(columns=object_col, inplace=True)

print(X_train.shape, X_test.shape)
print(X_train.cust_id.nunique(), X_test.cust_id.nunique(), y_train.cust_id.nunique())
X_train.set_index('cust_id', inplace=True)
X_test.set_index('cust_id', inplace=True)
y_train.set_index('cust_id', inplace=True)

print(X_train.columns)
print(X_test.columns)
print(y_train.columns)

# 1-4. X,y 정의하기
X = X_train
y = y_train['gender']
testData = X_test
print(X.shape, y.shape, testData.shape)
print(X.head(1))
print(testData.head(1))
print(y.head(1))


# 2. 전처리
# get_dummies, label encoding: 필요없음
# Scaling
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(X)
X = pd.DataFrame(ss.transform(X), columns=X.columns, index=X.index)
testData = pd.DataFrame(ss.transform(testData), columns=testData.columns, index=testData.index)
print(X.mean().mean(), X.std().mean())
print(testData.mean().mean(), testData.std().mean())
print(X.head(3))
print(testData.head(3))


# 3. train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y, random_state=0)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


# 4. modeling, 학습, 예측, 평가
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))
print(accuracy_score(y_test, rf.predict(X_test)))
print(f1_score(y_test, rf.predict(X_test)))


# 5-0. 테스트
pred = rf.predict_proba(testData)[:,1]
print(pd.DataFrame({'cust_id': testData.index, 'gender': pred}).head(10))


# 5. 제출
rf = RandomForestClassifier()
rf.fit(X, y)
pred = rf.predict_proba(testData)[:,1]
answer = pd.DataFrame({'cust_id': testData.index, 'gender': pred})
answer.to_csv('수험번호.csv', index=False)
ans = pd.read_csv("수험번호.csv")
print(ans.head(10))

 
