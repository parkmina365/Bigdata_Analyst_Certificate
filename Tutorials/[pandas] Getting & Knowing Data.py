
# Q1~Q19 Getting & Knowing Data

# Q1~Q8
# Q1. 데이터를 로드하라. 데이터는 \t을 기준으로 구분되어있다.
# 롤 랭킹 데이터 : https://raw.githubusercontent.com/Datamanim/pandas/main/lol.csv
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/pandas/main/lol.csv', sep='\t')

# Q2. 데이터의 상위 5개행을 출력하라
print(df.head())

# Q3. 데이터의 행과 열의 갯수를 파악하라
print(df.shape[0])
print(df.shape[1])

# Q4. 전체 컬럼을 출력하라
print(df.columns)

# Q5. 6번째 컬럼명을 출력하라
print(df.columns[5])

# Q6. 6번째 컬럼의 데이터 타입을 확인하라
print(df[df.columns[5]].dtypes)
print(df.iloc[:,5].dtypes)
print(df.dtypes[5])

# Q7. 데이터셋의 인덱스 구성은 어떤가
print(df.index)

# Q8. 6번째 컬럼의 3번째 값은 무엇인가?
print(df.iloc[2,5])


# Q9~Q19
# Q9. 데이터를 로드하라.
# 제주 날씨, 인구에 따른 교통량데이터 : 출처 제주 데이터 허브 https://raw.githubusercontent.com/Datamanim/pandas/main/Jeju.csv
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/pandas/main/Jeju.csv', encoding='utf8')

# Q10. 데이터 마지막 3개행을 출력하라
print(df.tail(3))

# Q11. 수치형 변수를 가진 컬럼을 출력하라
print(df.select_dtypes(exclude='object').columns)

# Q12. 범주형 변수를 가진 컬럼을 출력하라
print(df.select_dtypes('object').columns)

# Q13. 각 컬럼의 결측치 숫자를 파악하라
print(df.isnull().sum())

# Q14. 각 컬럼의 데이터수, 데이터타입을 한번에 확인하라
print(df.info())

# Q15. 각 수치형 변수의 분포(사분위, 평균, 표준편차, 최대, 최소)를 확인하라
print(df.describe())

# Q16. 거주인구 컬럼의 값들을 출력하라
print(df['거주인구'])

# Q17. 평균 속도 컬럼의 4분위 범위(IQR) 값을 구하여라
print(df['평균 속도'].quantile(0.75) - df['평균 속도'].quantile(0.25))

# Q18. 읍면동명 컬럼의 유일값 갯수를 출력하라
print(df['읍면동명'].nunique())

# Q19. 읍면동명 컬럼의 유일값을 모두 출력하라
print(df['읍면동명'].unique())
  
 
