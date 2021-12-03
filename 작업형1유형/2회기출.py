
# 1.  컬럼1에서 상위 10개 값들을 상위 10번째 값으로 대체한 후, 컬럼2의 값이 5이상인 데이터에 대한 컬럼1의 평균값을 구하여라
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/dataq/main/mtcars.csv',index_col=0)
df.sort_values(by='mpg', ascending=False, inplace=True)
df.iloc[:9,0] = df.iloc[9,0]
print(df[df.iloc[:,1]>=5].iloc[:,0].mean())

# 2. 데이터 전체에서 80% 추출 후 컬럼1의 결측치들을 median으로 채우고 컬럼1의 표준편차의 변화값: 1.975
df_copy = df.iloc[:int(df.shape[0]*0.8), :]
df_copy.iloc[:,0].fillna(df_copy.iloc[:,0].median(), inplace=True)
df_copy_std = df_copy.iloc[:,0].std()
df_std = df.iloc[:,0].std()
print(abs(df_copy_std - df_std))

# 3. 컬럼1의 Outlier 추출 후 그 합계를 구하여라
Q1 = df.iloc[:,0].quantile(0.25)
Q3 = df.iloc[:,0].quantile(0.75)
IQR = Q3-Q1
outliers = df[(df.iloc[:,0]<= Q1-(1.5*IQR))|(df.iloc[:,0]>= Q3+(1.5*IQR))].iloc[:,0].sum()
print(outliers)
 
