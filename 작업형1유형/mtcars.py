
# Q1~Q7
# 데이터 설명 : 자동차 관련 데이터
# 데이터 출처 : https://www.kaggle.com/vik2012kvs/mt-cars


# Q1. qsec 컬럼을 최소 최대 척도(min-max scale)로 변환한 후 0.5보다 큰 값을 가지는 레코드 수를 구하시오
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/dataq/main/mtcars.csv',index_col=0)
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
df['qsec'] = mm.fit_transform(df[['qsec']])
print(len(df[df['qsec']>0.5]))


# Q2. qsec 컬럼을 표준정규분포 데이터 표준화(standardization) 변환 후 최대, 최소값을 각각 구하시오
from sklearn.preprocessing import StandardScaler 
ss = StandardScaler()
df['qsec'] = ss.fit_transform(df[['qsec']])
print(df['qsec'].min())
print(df['qsec'].max())


# Q3. wt 컬럼의 이상치(IQR 1.5 외부에 존재하는)값들을 outlier 변수에 저장하라
q1 = df['wt'].quantile(0.25)
q3 = df['wt'].quantile(0.75)
outlier = df.wt[(df.wt<= q1 - 1.5*(q3-q1))|(df.wt>= q3 + 1.5*(q3-q1))].values
print(outlier)


# Q4. mpg변수와 나머지 변수들의 상관계수를 구하여 다음과 같이 내림차순 정렬하여 표현하라
print(df.corr()[['mpg']][1:].sort_values(by='mpg', ascending=False))


# Q5. mpg변수를 제외하고 데이터 정규화 (standardscaler) 과정을 진행한 이후 PCA를 통해 변수 축소를 하려한다.\
누적설명 분산량이 92%를 넘기기 위해서는 몇개의 주성분을 선택해야하는지 설명하라
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
ss = StandardScaler()
df = df.iloc[:, 1:]
df = pd.DataFrame(ss.fit_transform(df))

pca = PCA(n_components=len(df.columns))
pca.fit_transform(df)
print(pca.explained_variance_ratio_.cumsum())


# Q6. index는 (업체명) - (모델명)으로 구성된다(valiant는 업체명). ‘brand’ 컬럼을 추가하고 value 값으로 업체명을 입력하라
df['brand'] = df.index.map(lambda x:x.split(' ')[0])
print(df)


# Q7. 추가된 brand 컬럼을 제외한 모든 컬럼을 통해 pca를 실시한다. 2개의 주성분과 brand컬럼으로 구성된 새로운 데이터 프레임을 출력하고,\
brand에 따른 2개 주성분을 시각화하여라 (brand를 구분 할수 있도록 색이다른 scatterplot, legend를 표시한다)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
df_w = df.drop(columns='brand')
ss = StandardScaler()
ss_df = pd.DataFrame(ss.fit_transform(df_w))

pca = PCA(n_components=2)
pca_df = pd.DataFrame(pca.fit_transform(ss_df), index=df_w.index)
pca_df['brand'] = df['brand']

from seaborn import scatterplot
import matplotlib.pyplot as plt
plt.figure(figsize=(12,12))
scatterplot(x=pca_df[0], y=pca_df[1], hue=pca_df['brand'])
plt.show()

  
