
# Q1~Q9
# 데이터 설명 : 전세계 행복도 지표 조사
# 데이터 출처 : https://www.kaggle.com/unsdsn/world-happiness


# Q1. 데이터는 2018년도와 2019년도의 전세계 행복 지수를 표현한다. 각년도의 행복랭킹 10위를 차지한 나라의 행복점수의 평균을 구하여라
import pandas as pd
df =pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/happy2/happiness.csv',encoding='utf-8')
print(df[df['행복랭킹']==10]['점수'].mean())


# Q2. 데이터는 2018년도와 2019년도의 전세계 행복 지수를 표현한다. 각년도의 행복랭킹 50위이내의 나라들의 각각의 행복점수 평균을 데이터프레임으로 표시하라
print(df[df['행복랭킹']<=50].groupby('년도')[['점수']].mean())


# Q3. 2018년도 데이터들만 추출하여 행복점수와 부패에 대한 인식에 대한 상관계수를 구하여라
print(df[df['년도']==2018][['점수','부패에 대한인식']].corr().iloc[0,1])


# Q4. 2018년도와 2019년도의 행복랭킹이 변화하지 않은 나라명의 수를 구하여라
# 풀이1
temp = df['나라명'].value_counts()
countries = temp[temp==2].index.to_list()
df_2018 = df[(df['나라명'].isin(countries))&(df['년도']==2018)].sort_values(by='행복랭킹').reset_index(drop=True)
df_2019 = df[(df['나라명'].isin(countries))&(df['년도']==2019)].sort_values(by='행복랭킹').reset_index(drop=True)
print((df_2018['나라명'] == df_2019['나라명']).sum())
# 풀이2
print(len(df[['행복랭킹','나라명']]) - len(df[['행복랭킹','나라명']].drop_duplicates()))


# Q5. 2019년도 데이터들만 추출하여 각변수간 상관계수를 구하고 내림차순으로 정렬한 후 상위 5개를 데이터 프레임으로 출력하라.\
컬럼명은 v1,v2,corr으로 표시하라
temp = df[df['년도']==2019].corr().unstack().to_frame().reset_index().dropna()
print(temp[temp[0]!=1].sort_values(0,ascending=False).drop_duplicates(0).head(5).rename(columns={'level_0':'v1','level_1':'v2', 0:'corr'}))


# Q6. 각 년도별 행복점수의 하위 5개 국가의 평균 행복점수를 구하여라
# 풀이1
print(df[df['년도']==2018]['점수'][-5:].mean())
print(df[df['년도']==2019]['점수'][-5:].mean())
# 풀이2
print(df.groupby('년도').tail(5).groupby('년도')[['점수']].mean().iloc[0,0])
print(df.groupby('년도').tail(5).groupby('년도')[['점수']].mean().iloc[1,0])


# Q7.2019년 데이터를 추출하고 해당데이터의 상대 GDP 평균 이상의 나라들과 평균 이하의 나라들의 행복점수 평균을 각각 구하고 그 차이값을 출력하라
df_2019 = df[df['년도']==2019]
print(abs(df_2019[df_2019['상대GDP']>=df_2019['상대GDP'].mean()]['점수'].mean() - df_2019[df_2019['상대GDP']<=df_2019['상대GDP'].mean()]['점수'].mean()))


# Q8.  각년도의 부패에 대한인식을 내림차순 정렬했을때 상위 20개 국가의 부패에 대한인식의 평균을 구하여라
# 풀이1
print(df[df['년도']==2018].sort_values(by='부패에 대한인식', ascending=False)[:20]['부패에 대한인식'].mean())
print(df[df['년도']==2019].sort_values(by='부패에 대한인식', ascending=False)[:20]['부패에 대한인식'].mean())
# 풀이2
print(df.sort_values(['년도','부패에 대한인식'],ascending=False).groupby('년도').head(20).groupby(['년도']).mean()[['부패에 대한인식']])


# Q9. 2018년도 행복랭킹 50위 이내에 포함됐다가 2019년 50위 밖으로 밀려난 국가의 숫자를 구하여라
# 풀이1
contries_2018 = df[(df['행복랭킹']<=50)&(df['년도']==2018)]['나라명'].values
print(50-len(df[(df['행복랭킹']<=50)&(df['년도']==2019)&(df['나라명'].isin(contries_2018))]))
# 풀이2
print(set(df[(df.년도==2018) & (df.행복랭킹 <=50)].나라명) -set(df[(df.년도==2019) & (df.행복랭킹 <=50)].나라명))


# Q10. 2018년,2019년 모두 기록이 있는 나라들 중 년도별 행복점수가 가장 증가한 나라와 그 증가 수치는?
temp = df['나라명'].value_counts()
countries = temp[temp==2].index.to_list()
df_2018 = df[(df['나라명'].isin(countries))&(df['년도']==2018)].sort_values(by='나라명').set_index('나라명')[['점수']]
df_2019 = df[(df['나라명'].isin(countries))&(df['년도']==2019)].sort_values(by='나라명').set_index('나라명')[['점수']]
print((df_2019-df_2018).sort_values(by='점수',ascending=False).iloc[0])

 
 
