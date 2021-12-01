
# Q83~Q90 6. Pivot

# Q83~Q86 국가별 5세이하 사망비율 데이터
# https://www.kaggle.com/utkarshxy/who-worldhealth-statistics-2020-complete
# Q83. Indicator을 삭제하고 First Tooltip 컬럼에서 신뢰구간에 해당하는 표현을 지워라
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/pandas/main/under5MortalityRate.csv')
df.drop(columns=['Indicator'], inplace=True)
df['First Tooltip'] = df['First Tooltip'].map(lambda x:x.split(' ')[0])

# Q84. 년도가 2015년 이상, Dim1이 Both sexes인 케이스만 추출하라
x = df[(df['Period']>=2015) & (df['Dim1']=='Both sexes')]

# Q85. 84번에서 추출한 데이터로 나라에 따른 년도별 사망률의 평균을 데이터 프레임화 하라
x['First Tooltip'] = x['First Tooltip'].astype(float)
x.pivot_table(index='Location', columns='Period', values='First Tooltip', aggfunc='mean')

# Q86. Dim1에 따른 년도별 사망비율의 평균을 구하라
df['First Tooltip'] =  df['First Tooltip'].astype('float')
df.pivot_table(index='Dim1', columns='Period',values='First Tooltip', aggfunc='mean')


# Q87~Q90 올림픽 메달리스트 데이터
# https://www.kaggle.com/the-guardian/olympic-games
# Q87. 데이터에서 한국 KOR 데이터만 추출하고, 범주형 자료의 pivot table을 만들어라
df[df['Country']=='KOR'].select_dtypes('object')

# Q88. 한국 올림픽 메달리스트 데이터에서 년도에 따른 medal 갯수를 데이터프레임화 하라
df[df['Country']=='KOR'].pivot_table(index='Year', columns='Medal', aggfunc='size')
# aggfunc='size': 한 칼럼에만 해당하는 갯수 반환
# aggfunc='count': 전체 칼럼에 해당하는 갯수 반환

# Q89. 전체 데이터에서 sport종류에 따른 성별수를 구하여라
df.pivot_table(index='Sport', columns='Gender', aggfunc='size')

# Q90. 전체 데이터에서 Discipline종류에 따른 따른 Medal수를 구하여라
df.pivot_table(index='Discipline', columns='Medal', aggfunc='size')

  
