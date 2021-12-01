
# Q91~Q95 7. Merge, Concat

# 국가별 5세이하 사망비율 데이터: https://www.kaggle.com/utkarshxy/who-worldhealth-statistics-2020-complete
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/pandas/main/mergeTEst.csv',index_col= 0)
df1 = df.iloc[:4,:]
df2 = df.iloc[4:,:]
df3 = df.iloc[:2,:4]
df4 = df.iloc[5:,3:]
df5 = df.T.iloc[:7,:3]
df6 = df.T.iloc[6:,2:5]

# Q91. df1과 df2 데이터를 하나의 데이터 프레임으로 합쳐라
pd.concat([df1, df2])
pd.concat([df1, df2], axis=0)  # axis의 기본값: 0

# Q92. df3과 df4 데이터를 하나의 데이터 프레임으로 합쳐라. 둘다 포함하고 있는 년도에 대해서만 고려한다
pd.concat([df3, df4], join='inner')

# Q93. df3과 df4 데이터를 하나의 데이터 프레임으로 합쳐라. 모든 컬럼을 포함하고, 결측치는 0으로 대체한다
pd.concat([df3,df4]).fillna(0)  # join의 기본값: outer

# Q94. df5과 df6 데이터를 하나의 데이터 프레임으로 merge함수를 이용하여 합쳐라.\
Algeria컬럼을 key로 하고 두 데이터 모두 포함하는 데이터만 출력하라
pd.merge(df5, df6, on='Algeria')
pd.merge(df5, df6, on='Algeria', how='inner')  # how의 기본값: inner

# Q95. df5과 df6 데이터를 하나의 데이터 프레임으로 merge함수를 이용하여 합쳐라. Algeria컬럼을 key로 하고 합집합으로 합쳐라
pd.merge(df5, df6, on='Algeria', how='outer')
 
