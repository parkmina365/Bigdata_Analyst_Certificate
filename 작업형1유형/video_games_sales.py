
# Q1~Q9
# 데이터 설명 : 게임 판매량 및 평점 데이터
# 데이터 출처 : https://mlcourse.ai/


# Q1. 데이터 Url을 이용하여 df변수에 데이터를 로드하라
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/video/master/video_games_sale.csv', index_col=0)


# Q2. 출시년도(Year_of_Release) 컬럼을 10년단위(ex 1990~1999 : 1990)로 변환하여 새로운 컬럼()에 추가하고\
게임이 가장 많이 출시된 년도(10년단위)와 가장 적게 출시된 년도(10년단위)를 각각 구하여라
df['Year_of_ten'] = ((df['Year_of_Release']//10)*10).astype(int)
print(df['Year_of_ten'].value_counts().sort_values().index[0])
print(df['Year_of_ten'].value_counts().sort_values().index[-1])


# Q3. 플레이스테이션 플랫폼 시리즈(PS,PS2,PS3,PS4,PSV)중 장르가 Action로 발매된 게임의 총 수는?
print(len(df[(df['Platform'].isin(['PS','PS2','PS3','PS4','PSV']))&(df['Genre']=='Action')]))


# Q4. 게임이 400개 이상 출시된 플랫폼들을 추출하여 각 플랫폼의 User_Score 평균값을 구하여 df를 만들고 값을 내림차순으로 정리하여 출력하라
platform = df['Platform'].value_counts()
over400 = platform[platform>=400].index
print(df[df['Platform'].isin(over400)].groupby('Platform').mean()[['User_Score']].sort_values(by='User_Score', ascending=False))


# Q5. 게임 이름에 Mario가 들어가는 게임을 3회 개발한 개발자(Developer컬럼)을 구하여라
x = df[df['Name'].str.contains('Mario')].groupby('Developer').size()
print(x[x==3].index.to_list())


# Q6. PS2 플랫폼으로 출시된 게임들의 User_Score의 첨도를 구하여라
import scipy.stats
print(df[df['Platform']=='PS2']['User_Score'].kurtosis())


# Q7. 각 게임별 NA_Sales,EU_Sales,JP_Sales,Other_Sales 값의 합은 Global_Sales와 동일해야한다.\
소숫점 2자리 이하의 생략으로 둘의 값의 다른경우가 존재하는데, 이러한 케이스가 몇개 있는지 확인하라
print(len(df[(df['NA_Sales']+df['EU_Sales']+df['JP_Sales']+df['Other_Sales']) != df['Global_Sales']]))


# Q8. User_Count컬럼의 값이 120 이상인 게임들 중에서 User_Score의 값이 9.0이상인 게임의 수를 구하여라
print(len(df[(df['User_Count']>=120)&(df['User_Score']>=9.0)]))


# Q9. Global_Sales컬럼의 값들을 robust스케일을 진행하고 40이상인 데이터 수를 구하여라
from sklearn.preprocessing import RobustScaler
rb = RobustScaler()
df['Global_Sales'] = rb.fit_transform(df[['Global_Sales']])
print(len(df[df['Global_Sales']>=40]))

 
