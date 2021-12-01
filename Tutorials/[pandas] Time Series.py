
# Q64~Q82 4. Time Series

# Q64~Q75 주가데이터
# Q64. 데이터를 로드하고 각 열의 데이터 타입을 파악하라
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/pandas/main/timeTest.csv')
print(df.dtypes)

# Q65. Yr_Mo_Dy을 판다스에서 인식할 수 있는 datetime64타입으로 변경하라
df['Yr_Mo_Dy'] = pd.to_datetime(df['Yr_Mo_Dy'])

# Q66. Yr_Mo_Dy에 존재하는 년도의 유일값을 모두 출력하라
print(df['Yr_Mo_Dy'].dt.year.unique())

# Q67. Yr_Mo_Dy에 년도가 2061년 이상의 경우에는 모두 잘못된 데이터이다. 해당경우의 값은 100을 빼서 새롭게 날짜를 Yr_Mo_Dy 컬럼에 정의하라
def cov(x):
    import datetime
    if x.year >= 2061: year = x.year-100
    else: year = x.year
    return pd.to_datetime(datetime.date(year,x.month,x.day))
df['Yr_Mo_Dy'] = df['Yr_Mo_Dy'].apply(cov)

# Q68. 년도별 각컬럼의 평균값을 구하여라
df.groupby(df['Yr_Mo_Dy'].dt.year).mean()

# Q69. weekday컬럼을 만들고 요일별로 매핑하라 (월요일:0 ~ 일요일:6)
df['weekday'] = df['Yr_Mo_Dy'].dt.weekday

# Q70. weekday컬럼을 기준으로 주말이면 1 평일이면 0의 값을 가지는 WeekCheck 컬럼을 만들어라
df['WeekCheck'] = df['weekday'].map(lambda x:1 if x>=5 else 0)
df['WeekCheck'] = df['weekday'].map(lambda x:1 if x in [5,6] else 0)

# Q71. 년도, 일자 상관없이 모든 컬럼의 각 달의 평균을 구하여라
df.groupby(df['Yr_Mo_Dy'].dt.month).mean()

# Q72. 모든 결측치는 컬럼기준 직전의 값으로 대체하고 첫번째 행에 결측치가 있을경우 뒤에있는 값으로 대채하라
df = df.fillna(method='ffill', axis='columns')
df = df.fillna(method='bfill', axis='columns')

# Q73. 년도 - 월을 기준으로 모든 컬럼의 평균값을 구하여라
df.groupby(df['Yr_Mo_Dy'].dt.strftime('%Y-%m')).mean()
df.groupby(df['Yr_Mo_Dy'].dt.to_period(freq='M')).mean()

# Q74. RPT 컬럼의 값을 일자별 기준으로 1차차분하라
df['RPT'] = df['RPT'].diff()

# Q75. RPT와 VAL의 컬럼을 일주일 간격으로 각각 이동평균한값을 구하여라
df[['RPT','VAL']].rolling(7).mean()


# Q76~Q82 서울시 미세먼지 데이터
# https://www.airkorea.or.kr/web/realSearch?pMENU_NO=97
# Q76. 년-월-일:시 컬럼을 pandas에서 인식할 수 있는 datetime 형태로 변경하라. 서울시의 제공데이터의 경우 0시가 24시로 표현된다
def conv_date(x):
    import datetime
    date = x.split(':')[0]
    hour = x.split(':')[1]
    if hour == '24':
        hour = '00:00:00'
        date = pd.to_datetime(date+' '+hour) + datetime.timedelta(days=1)
    else:
        hour = hour+':00:00'
        date = pd.to_datetime(date+' '+hour)
    return date
df['(년-월-일:시)']=df['(년-월-일:시)'].apply(conv_date)

# Q77. 일자별 영어요일 이름을 dayName 컬럼에 저장하라
# 풀이1
df['dayName'] = df['(년-월-일:시)'].dt.day_name()
# 풀이2
dic = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
df['dayName'] = df['(년-월-일:시)'].dt.weekday.map(lambda x:dic[x])

# Q78. 요일별 각 PM10등급의 빈도수를 파악하라
df.groupby(['dayName','PM10등급']).size().unstack()
df.groupby(['dayName','PM10등급'], as_index=False).size().pivot_table(index='dayName', columns='PM10등급', values='size')

# Q79. 시간이 연속적으로 존재하며 결측치가 없는지 확인하라
print(df["(년-월-일:시)"].diff())
print(df["(년-월-일:시)"].diff().isnull().sum())   # null이 1개
print(df['(년-월-일:시)'].diff().value_counts())   # value가 1종류이므로 시간이 연속임

# Q80. 오전 10시와 오후 10시(22시)의 PM10의 평균값을 각각 구하여라
# 풀이1
print(df[df['(년-월-일:시)'].dt.hour ==10]['PM10'].mean())
print(df[df['(년-월-일:시)'].dt.hour ==22]['PM10'].mean())
# 풀이2
df.groupby(df['(년-월-일:시)'].dt.hour).mean()[['PM10']].iloc[[10,22]]

# Q81. 날짜 컬럼을 index로 만들어라
df.set_index('(년-월-일:시)', inplace=True)

# Q82. 데이터를 주단위로 뽑아서 최소,최대 평균, 표준표차를 구하여라
df.resample('W').agg(['min','max','mean','std'])  # index가 datetime인 경우만 가능

  
