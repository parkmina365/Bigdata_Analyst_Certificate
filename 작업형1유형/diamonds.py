
# Q1~Q9
# 데이터 설명 : 게임 판매량 및 평점 데이터
# 데이터 출처 : https://mlcourse.ai/


# Q1. carat과 price의 경향을 비교하기 위한 scatterplot그래프를 출력하시오
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/dataq/main/diamonds.csv',index_col=0)

from seaborn import scatterplot
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
scatterplot(data=df, x='carat', y='price')
plt.show()


# Q2*. carat과 price사이의 상관계수와 상관계수의 p-value값은?
from scipy import stats
corr, p_value = stats.pearsonr(df['carat'], df['price'])
print(corr)
print(p_value)


# Q3. Diamond의 평균가격은 3932로 알려져있다. ‘H’ color를 가지는 다이아몬드 집단의 평균에 대한 일표본 t검정을 시행하려한다.\
통계량과 pvalue값을 구하시오. 유의수준 0.05에서 귀무가설 채택여부를 boolean 값으로 표현할 변수(hypo) 만들고 출력하시오
from scipy import stats
df_h = df[df['color']=='H']
t_stats, p_value = stats.ttest_1samp(df_h['price'], 3932)
hypo = (p_value > 0.05)
print(t_stats, p_value)
print(hypo)


# Q4. 그래프상에서 ‘F’와 ‘G’는 평균이 유사해보인다. \
이를 확인하기 위해 집단간 등분산(levene,fligner,bartlett) 검정을 시행 후 결과를 출력하고 조건에 맞는 독립표본 t검정을 시행하라
from scipy.stats import levene, fligner, bartlett, ttest_ind
df_f = df[df['color']=='F']
df_g = df[df['color']=='G']
print(levene(df_f['price'], df_g['price']))
print(fligner(df_f['price'], df_g['price']))
print(bartlett(df_f['price'], df_g['price']))
# 유의수준 0.05에서 등분산검정의 귀무가설 기각. df_f, df_g의 분산은 같지 않음

print(ttest_ind(df_f['price'], df_g['price'], equal_var=False))
# 유의수준 0.05에서 ttest_ind 귀무가설 기각. df_f, df_g의 평균은 같지 않음


# Q5. color ‘F’,’G’,’D’ 세집단의 price값들에 대해 anova분석을 시행하라
from scipy.stats import levene, fligner, bartlett, ttest_ind
from scipy.stats import f_oneway
df_f = df[df['color']=='F']
df_g = df[df['color']=='G']
df_d = df[df['color']=='D']
print(levene(df_f['price'], df_g['price'], df_d['price']))
print(fligner(df_f['price'], df_g['price'], df_d['price']))
print(bartlett(df_f['price'], df_g['price'], df_d['price']))
# 유의수준 0.05에서 등분산검정의 귀무가설 기각. df_f, df_g, df_d의 분산은 같지 않음

print(f_oneway(df_f['price'], df_g['price'], df_d['price']))
# 유의수준 0.05에서 ANOVA 귀무가설 기각. df_f, df_g, df_d의 분산은 같지 않음
# 세 집단 중 어느 두 집잔의 평균도 같지 않다


# Q6. 연속형 변수(carat,depth,table,price,x,y,z) 각각의 이상치(1,3분위값에서 IQR*1.5 외의 값) 갯수를 \
데이터 프레임(변수명 ratio_df, 비율의 내림차순 정렬)으로 아래와 같이 나타내어라.
# 풀이1
cols = ['carat','depth','table','price','x','y','z']
outlier = []
for i in cols:
    q1 = df[i].quantile(0.25)
    q3 = df[i].quantile(0.75)
    iqr = q3-q1
    outlier.append(len(df[(df[i]<=q1-(1.5*iqr))|(df[i]>=q3+(1.5*iqr))]))
ratio_df = pd.DataFrame(cols, columns=['columns'])
ratio_df['ratio'] = outlier
ratio_df.sort_values(by='ratio', ascending=False, inplace=True)
print(ratio_df)

# 풀이2
cols = ['carat','depth','table','price','x','y','z']
outlier = []
for i in cols:
    q1 = df[i].quantile(0.25)
    q3 = df[i].quantile(0.75)
    iqr = q3-q1
    outlier.append([i,len(df[(df[i]<=q1-(1.5*iqr))|(df[i]>=q3+(1.5*iqr))])])
ratio_df = pd.DataFrame(outlier, columns=['columns', 'ratio']).sort_values(by='ratio', ascending=False)
print(ratio_df)


# Q7. color에 따른 price의 max, min, 평균값을 colorDf 변수에 저장하고 아래와 같이 출력하는 코드를 작성하라
# 풀이1
colorDf = df.groupby('color')[['price']].max().rename(columns={'price':'max'})
colorDf['min'] = df.groupby('color')[['price']].min()
colorDf['mean'] = df.groupby('color')[['price']].mean()
print(colorDf)

# 풀이2
colorDf = df.groupby('color')['price'].agg(['min','max','mean'])
print(colorDf)


# Q8. 전체 데이터중 color의 발생빈도수에 따라 labelEncoding(빈도수 적은것:1, 빈도수 증가할수록 1씩증가)로 colorLabel 컬럼에 저장하고\
cut에 따른 colorLabel의 평균값을 구하여라
dic = {v:i+1 \
    for i,v in enumerate(df['color'].value_counts().sort_values().index.to_list())}
df['colorLabel'] = df['color'].map(lambda x:dic[x])
df.groupby('cut')[['colorLabel']].mean()


# Q9. price의 값에 따른 구간을 1000단위로 나누고 priceLabel 컬럼에 저장하라. \
저장시 숫자 순으로 label하고(0~1000미만 : 0,1000이상~2000미만 :1 …) 최종적으로 구간별 갯수(변수명:labelCount)를 출력하라
df['priceLabel'] = df['price']//1000
labelCount = df[['priceLabel']].value_counts().to_frame().reset_index().rename(columns={0:'counts'})
print(labelCount)

  
