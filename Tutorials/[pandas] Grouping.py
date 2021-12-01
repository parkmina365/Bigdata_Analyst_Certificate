
# Q44~Q55 3. Grouping

# Q44. 데이터를 로드하고 상위 5개 컬럼을 출력하라
# 뉴욕 airBnB 데이터 : https://www.kaggle.com/ptoscano230382/air-bnb-ny-2019
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/pandas/main/AB_NYC_2019.csv')
print(df.head())

# Q45. 데이터의 각 host_name의 빈도수를 구하고 host_name으로 정렬하여 상위 5개를 출력하라
print(df['host_name'].value_counts().sort_index().head())

# Q46. 데이터의 각 host_name의 빈도수를 구하고 빈도수 기준 내림차순 정렬한 데이터 프레임을 만들어라. 빈도수 컬럼은 counts로 명명하라
df['host_name'].value_counts().sort_values(ascending=False).to_frame().rename(columns={'host_name':'counts'})

# Q47. neighbourhood_group의 값에 따른 neighbourhood컬럼 값의 갯수를 구하여라
df.groupby(['neighbourhood_group', 'neighbourhood'], as_index=False)['neighbourhood'].size()
df.groupby(['neighbourhood_group', 'neighbourhood'], as_index=False).size()

# Q48. neighbourhood_group의 값에 따른 neighbourhood컬럼 값 중 neighbourhood_group그룹의 최댓값들을 출력하라
df.groupby(['neighbourhood_group','neighbourhood'], as_index=False).size().groupby('neighbourhood_group', as_index=False).max()

# Q49. neighbourhood_group 값에 따른 price값의 평균, 분산, 최대, 최소 값을 구하여라
df.groupby('neighbourhood_group')['price'].agg(['mean','var','max','min'])

# Q50. neighbourhood_group 값에 따른 reviews_per_month 평균, 분산, 최대, 최소 값을 구하여라
df.groupby('neighbourhood_group')[['reviews_per_month']].agg(['mean','var','max','min'])

# Q51. neighbourhood 값과 neighbourhood_group 값에 따른 price 의 평균을 구하라
df.groupby([ 'neighbourhood','neighbourhood_group']).mean()[['price']]

# Q52. neighbourhood 값과 neighbourhood_group 값에 따른 price 의 평균을 계층적 indexing 없이 구하라
df.groupby(['neighbourhood','neighbourhood_group']).mean()['price'].unstack()

# Q53. neighbourhood 값과 neighbourhood_group 값에 따른 price 의 평균을 계층적 indexing 없이 구하고 nan 값은 -999값으로 채워라
df.groupby(['neighbourhood','neighbourhood_group']).mean()['price'].unstack().fillna(-999)

# Q54. 데이터중 neighbourhood_group 값이 Queens값을 가지는 데이터들 중 neighbourhood 그룹별로 price값의 평균, 분산, 최대, 최소값을 구하라
df[df['neighbourhood_group']=='Queens'].groupby('neighbourhood')['price'].agg(['mean','var','max','min'])

# Q55. 데이터중 neighbourhood_group 값에 따른 room_type 컬럼의 숫자를 구하고 neighbourhood_group 값을 기준으로 각 값의 비율을 구하여라
# 풀이1
df = df.groupby(['neighbourhood_group', 'room_type']).size().unstack()
df['total'] = df['Entire home/apt'] + df['Private room'] + df['Shared room']
df['Entire home/apt'] = df['Entire home/apt'] / df['total']
df['Private room'] = df['Private room'] / df['total']
df['Shared room'] = df['Shared room'] / df['total']
df.iloc[:, :-1]

# 풀이2
df = df.groupby(['neighbourhood_group','room_type']).size().unstack()
df.loc[:,:] = (df.values) / (df.sum(axis=1).values.reshape(-1,1))

  
