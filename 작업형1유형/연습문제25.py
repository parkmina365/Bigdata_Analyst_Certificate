
# Q1~Q2
# 데이터 설명 : 2020년도 이화동(서울), 수영동(부산)의 시간단위의 기온과 강수량(시계열 데이터)
# 데이터 출처 : https://data.kma.go.kr/cmmn/static/staticPage.do?page=intro

# Q1. 여름철(6월,7월,8월) 이화동이 수영동보다 높은 기온을 가진 시간대는 몇개인가?
import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/weather/weather2.csv")
df.info()
df['time'] = pd.to_datetime(df['time'])
print(len(df[(df.time>='2020-06-01')&(df.time<='2020-08-31')&(df['이화동기온'] > df['수영동기온'])]))

# Q2. 이화동과 수영동의 최대강수량의 시간대를 각각 구하여라
print(df[df.이화동강수 == df.이화동강수.max()]['time'].values)
print(df[df.수영동강수 == df.수영동강수.max()]['time'].values)


# --------------------
# Q3~Q5
# 데이터 설명 :  고객의 신상정보 데이터를 통한 회사 서비스 이탈 예측 (종속변수 : 'Exited')
# 데이터 출처 : https://www.kaggle.com/shubh0799/churn-modelling

# Q3.  남성 이탈(Exited)이 가장 많은 국가(Geography)는 어디이고 이탈 인원은 몇명인가?
# 풀이1. 조건 파악: df.Gender=='Male', df.groupby('Geograpy')
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/churn/train.csv")
df = df[df.Gender=='Male'].groupby('Geography')['Exited'].sum().sort_values(ascending=False)
print(df.index[0])
print(df.iloc[0])

# 풀이2. pivot_table
df = df.pivot_table(index='Geography', columns='Gender', values='Exited', aggfunc='sum')['Male'].sort_values(ascending=False)
print(df.index[0])
print(df.iloc[0])

# Q4. 카드를 소유(HasCrCard ==1)하고 있으면서 활성멤버(IsActiveMember ==1) 인 고객들의 평균나이는? 
print(df[(df.HasCrCard==1)&(df.IsActiveMember==1)]['Age'].mean())

# Q5. Balance 값이 중간값 이상을 가지는 고객들의 CreditScore의 표준편차를 구하여라
print(df[df.Balance >= df.Balance.median()]['CreditScore'].std())


# --------------------
# Q6~Q9
# 데이터 설명 : 2018년도 성인의 건강검진 데이터 (종속변수: '흡연상태')
# 데이터 출처 : https://www.data.go.kr/data/15007122/fileData.do

# Q6. 시력(좌) 와 시력(우)의 값이 같은 남성의 허리둘레의 평균은?
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/smoke/train.csv")
print(df[(df['성별코드']=='M')&(df['시력(좌)']==df['시력(우)'])]['허리둘레'].mean())

# Q7. 40대(연령대코드 40,45) 여성 중 '총콜레스테롤'값이 40대 여성의 '총콜레스테롤'중간값 이상을 가지는 그룹과\
50대(연령대코드 50,55) 여성 중 '총콜레스테롤'값이 50대 여성의 '총콜레스테롤' 중간값 이상을 가지는 두 그룹의 \
'수축기혈압'이 독립성,정규성,등분산성이 만족하는것을 확인했다. \
두 그룹의 '수축기혈압'의 독립표본 t 검증 결과를 통계값, p-value 구분지어 구하여라.
from scipy.stats import ttest_ind
df40 = df[(df['연령대코드(5세단위)']==40)|(df['연령대코드(5세단위)']==45)]
df50 = df[(df['연령대코드(5세단위)']==50)|(df['연령대코드(5세단위)']==55)]
df40 = df40[df40['총콜레스테롤']>=df40['총콜레스테롤'].median()]
df50 = df50[df50['총콜레스테롤']>=df50['총콜레스테롤'].median()]
print(ttest_ind(df40['수축기혈압'], df50['수축기혈압'], equal_var=True))

# Q8. 수축기혈압과 이완기 혈압기 수치의 차이를 새로운 컬럼('혈압차') 으로 생성하고, 연령대 코드별 각 그룹 중 '혈압차' 의 분산이 5번째로 큰 연령대 코드를 구하여라
df['혈압차'] = df['수축기혈압'] - df['이완기혈압']
df.groupby('연령대코드(5세단위)')['혈압차'].var().sort_values(ascending=False).index[4]

# Q9. 비만도를 나타내는 지표인 WHtR는 허리둘레 / 키로 표현한다. 일반적으로 0.58이상이면 비만으로 분류한다. 데이터중 WHtR 지표상 비만인 인원의 남/여 비율을 구하여라
print(df[df['허리둘레']/df['신장(5Cm단위)']>=0.58].groupby('성별코드').size()/len(df[df['허리둘레']/df['신장(5Cm단위)']>=0.58]))


# --------------------
# Q10~Q11
# 데이터 설명 : 자동차 보험 가입 예측 (종속변수: 'Response')
# 데이터 출처 : https://www.kaggle.com/anmolkumar/health-insurance-cross-sell-prediction

# Q10. Vehicle_Age 값이 2년이상인 사람들 중 Annual_Premium 값이 해당 그룹의 중간값 이상인 사람들을 찾고, 그들의 Vintage값의 평균을 구하여라
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/insurance/train.csv")
df = df[(df.Vehicle_Age!='< 1 Year')]
print(df[df.Annual_Premium >= df.Annual_Premium.median()]['Vintage'].mean())

# Q11. vehicle_age에 따른 각 성별(gender)그룹의 Annual_Premium값의 평균을 구하여 아래 테이블과 동일하게 구현하라
display(df.pivot_table(index='Vehicle_Age', columns='Gender', values='Annual_Premium', aggfunc='mean'))


# --------------------
# Q12~Q13*
# 데이터 설명 : 핸드폰 가격예측 (종속변수: 'price_range') 
# 데이터 출처 : https://www.kaggle.com/iabhishekofficial/mobile-price-classification?select=train.csv

# Q12. price_range 의 각 value를 그룹핑하여, 각 그룹(price_range)의 n_cores의 빈도가 가장높은 value와 그 빈도수를 구하여라
print(df.groupby(['price_range', 'n_cores'])['n_cores'].size().sort_values(ascending=False).groupby('price_range').head(1))

# Q13. price_range 값이 3인 그룹에서 상관관계가 2번째로 높은 두 컬럼과 그 상관계수를 구하여라
df = pd.read_csv('c:/workspace/cakd3/수업/dataset/빅분기/mobile.csv',index_col=0)
df = df[df.price_range==3].corr().unstack().sort_values(ascending=False)
print(df[df!=1].index[0])
print(df[df!=1].iloc[0])


# --------------------
# Q14*
# 데이터 설명 : 비행탑승 경험 만족도 (종속변수: 'satisfaction')
# 데이터 출처 : https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction?select=train.csv

# Arrival Delay in Minutes 컬럼이 결측치인 데이터들 중 'neutral or dissatisfied' 보다 'satisfied'의 수가 더 높은 Class는 어디 인가?
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/airline/train.csv")
df = df[df['Arrival Delay in Minutes'].isnull()].groupby(['Class','satisfaction'],as_index=False).size().pivot_table(index='Class', columns='satisfaction', values='size')
df[df['neutral or dissatisfied']<= df['satisfied']]


# --------------------
# Q15
# 데이터 설명 : 수질 음용성 여부 (종속변수: 'Potablillity')
# 데이터 출처 : https://www.kaggle.com/adityakadiwal/water-potability

# ph값은 상당히 많은 결측치를 포함한다. 결측치를 제외한 나머지 데이터들 중 하위 25%의 값들의 평균값은?
df = df[df['ph'].notnull()]['ph']
# df = df['ph'].dropna()
df[df <= df.quantile(0.25)].mean()


# --------------------
# Q16
# 데이터 설명 : 의료비용 예측문제 (종속변수: 'charges')
# 데이터 출처 : https://www.kaggle.com/mirichoi0218/insurance/code

# 흡연자와 비흡연자 각각 charges의 상위 10% 그룹의 평균의 차이는?
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/MedicalCost/train.csv")
no = df[df['smoker']=='no']['charges']
yes = df[df['smoker']=='yes']['charges']
no[no >= no.quantile(0.9)].mean() - yes[yes >= yes.quantile(0.9)].mean()


# --------------------
# Q17
# 데이터 설명 : 킹카운티 주거지 가격 예측문제 (종속변수: 'price')
# 데이터 출처 : https://www.kaggle.com/harlfoxem/housesalesprediction

# bedrooms 의 빈도가 가장 높은 값을 가지는 데이터들의 price의 상위 10%와 하위 10%값의 차이를 구하여라
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/kingcountyprice/train.csv")
df = df[df['bedrooms']==df['bedrooms'].value_counts().sort_values(ascending=False).index[0]]['price']
df.quantile(0.9) - df.quantile(0.1)


# --------------------
# Q18
# 데이터 설명 : 대학원 입학 가능성 예측 (종속변수: 'Chance of Admit')
# 데이터 출처 : https://www.kaggle.com/mohansacharya/graduate-admissions

# Serial No. 컬럼을 제외하고 'Chance of Admit'을 종속변수, 나머지 변수를 독립변수라 할때, 랜덤포레스트를 통해 회귀 예측을 할 떄 변수중요도 값을 출력하라
from sklearn.ensemble import RandomForestRegressor 
df.drop(columns=['Serial No.'], inplace=True)
rf = RandomForestRegressor()
rf.fit(df.iloc[:,:-1], df.iloc[:,-1])
print(pd.Series(rf.feature_importances_, index=df.iloc[:,:-1].columns).sort_values(ascending=False))


# --------------------
# Q19
# 데이터 설명 : 레드 와인 퀄리티 예측문제 (종속변수: 'quality')
# 데이터 출처 : https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009

# quality 값이 3인 데이터와 8인 데이터의 각 독립변수의 표준편차 값이 가장 큰 컬럼을 구하여라
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/redwine/train.csv")
print(df[df.quality==3].std().sort_values(ascending=False).index[0])
print(df[df.quality==8].std().sort_values(ascending=False).index[0])


# --------------------
# Q20*
# 데이터 설명 : 투약하는 약을 분류 (종속변수: 'Drug')
# 데이터 출처 : https://www.kaggle.com/prathamtripathi/drug-classification

# 남성들의 연령대별 (10살씩 구분 0~9세 10~19세 ...) Na_to_K값의 평균값을 구해서 데이터 프레임으로 표현하여라
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/drug/train.csv")
df['Age'] = (df['Age']//10)*10
df[df['Sex']=='M'].groupby('Age')[['Na_to_K']].mean()


# --------------------
# Q21
# 데이터 설명 : 사기회사 분류 (종속변수: 'Risk')
# 데이터 출처 : https://www.kaggle.com/sid321axn/audit-data

# 데이터의 Risk 값에 따른 score_a와 score_b의 평균값을 구하여라
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/audit/train.csv")
df.groupby('Risk')[['Score_A', 'Score_B']].mean()


# --------------------
# Q22
# 데이터 설명 : 센서데이터로 동작 유형 분류 (종속변수: 'pose')
# 데이터 출처 : https://www.kaggle.com/kyr7plus/emg-4

# pose값에 따른 각 motion컬럼의 중간값의 가장 큰 차이를 보이는 motion컬럼은 어디이며 그값은?
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/muscle/train.csv")
df = df.groupby('pose').median().T
df['diff'] = (df[1]-df[0]).abs()
df['diff'].sort_values(ascending=False).index[0]


# --------------------
# Q23
# 데이터 설명 : 현대 차량가격 분류문제 (종속변수: 'price')
# 데이터 출처 : https://www.kaggle.com/mysarahmadbhat/hyundai-used-car-listing

# 정보(row수)가 가장 많은 상위 3차종의 price값의 각 평균값은?
# 풀이1
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/hyundai/train.csv")
top_3 = df['model'].value_counts().sort_values().index[-3:].to_list()
df.groupby('model').mean()[['price']].loc[top_3]

# 풀이2
df[df['model'].isin(df.model.value_counts().sort_values(ascending=False).index[:3])].groupby('model').mean()[['price']]


# --------------------
# Q24
# 데이터 설명 : 당뇨여부 판단하기 (종속변수: 'Outcome')
# 데이터 출처 : https://www.kaggle.com/pritsheta/diabetes-dataset

# Outcome 값에 따른 각 그룹의  각 컬럼의 평균 차이를 구하여라
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/diabetes/train.csv")
df = df.groupby('Outcome').mean().T
print((df[0]-df[1]).abs())


# --------------------
# Q25
# 데이터 설명 : 넷플릭스 주식데이터
# 데이터 출처 : https://www.kaggle.com/pritsheta/netflix-stock-data-from-2002-to-2021

# 매년 5월달의 open가격의 평균값을 데이터 프레임으로 표현하라
# 풀이1
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/nflx/NFLX.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].apply(lambda x:x.month)
df['Year'] = df['Date'].apply(lambda x:x.year)
print(df[df.Month==5].groupby('Year').mean()[['Open']])

# 풀이2
df['Date'] = pd.to_datetime(df['Date'])
df = df[df.Date.dt.month ==5]
print(df.groupby(df.Date.dt.strftime('%Y-%m')).mean()[['Open']])

# 풀이3
df['Date'] = pd.to_datetime(df['Date'])
df = df.groupby(df['Date'].dt.strftime('%Y-%m')).mean()
print(df.loc[df.index.str.contains('-05')][['Open']])

 
