
# Q20~Q43 2. Filtering & Sorting

# Q20. 데이터를 로드하라.
# 식당데이터 : https://raw.githubusercontent.com/Datamanim/pandas/main/chipo.csv
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/pandas/main/chipo.csv')

# Q21. quantity컬럼 값이 3인 데이터를 추출하여 첫 5행을 출력하라
print(df[df['quantity']==3].head())

# Q22. quantity컬럼 값이 3인 데이터를 추출하여 index를 0부터 정렬하고 첫 5행을 출력하라
print(df[df['quantity']==3].reset_index(drop=True).head())

# Q23. quantity, item_price 두개의 컬럼으로 구성된 새로운 데이터 프레임을 정의하라
new_df = df[['quantity', 'item_price']]

# Q24. item_price 컬럼의 달러표시 문자를 제거하고 float 타입으로 저장하여 new_price 컬럼에 저장하라
df['new_price'] = df['item_price'].map(lambda x:x.split('$')[1]).astype('float')
df['new_price'] = df['item_price'].str[1:].astype('float')

# Q25. new_price 컬럼이 5이하의 값을 가지는 데이터프레임을 추출하고, 전체 갯수를 구하여라
df_less5 = df[df['new_price']<=5]
print(len(df_less5))

# Q26. item_name명이 Chicken Salad Bowl 인 데이터 프레임을 추출하라고 index 값을 초기화 하여라
df[df['item_name']=='Chicken Salad Bowl'].reset_index(drop=True)

# Q27. new_price값이 9 이하이고 item_name 값이 Chicken Salad Bowl 인 데이터 프레임을 추출하라
df[(df['new_price']<=9)&(df['item_name']=='Chicken Salad Bowl')]

# Q28. df의 new_price 컬럼 값에 따라 오름차순으로 정리하고 index를 초기화 하여라
df.sort_values(by='new_price').reset_index(drop=True)

# Q29. df의 item_name 컬럼 값중 Chips 포함하는 경우의 데이터를 출력하라
df[df['item_name'].str.contains('Chips')]

# Q30. df의 짝수번째 컬럼만을 포함하는 데이터프레임을 출력하라
df.iloc[:,::2]

# Q31. df의 new_price 컬럼 값에 따라 내림차순으로 정리하고 index를 초기화 하여라
df.sort_values(by='new_price', ascending=False).reset_index(drop=True)

# Q32. df의 item_name 컬럼 값이 Steak Salad 또는 Bowl 인 데이터를 인덱싱하라
df[(df['item_name']=='Steak Salad')|(df['item_name']=='Bowl')]

# Q33. df의 item_name 컬럼 값이 Steak Salad 또는 Bowl 인 데이터를 데이터 프레임화 한 후,\
item_name를 기준으로 중복행이 있으면 제거하되 첫번째 케이스만 남겨라
df[(df['item_name']=='Steak Salad')|(df['item_name']=='Bowl')].drop_duplicates(subset='item_name', keep='first')

# Q34. df의 item_name 컬럼 값이 Steak Salad 또는 Bowl 인 데이터를 데이터 프레임화 한 후, \
item_name를 기준으로 중복행이 있으면 제거하되 마지막 케이스만 남겨라
df[(df['item_name']=='Steak Salad')|(df['item_name']=='Bowl')].drop_duplicates(subset='item_name', keep='last')

# Q35. df의 데이터 중 new_price값이 new_price값의 평균값 이상을 가지는 데이터들을 인덱싱하라
df[df['new_price']>=df['new_price'].mean()]

# Q36. df의 데이터 중 item_name의 값이 Izze 데이터를 Fizzy Lizzy로 수정하라
df['item_name'].replace('Izze', 'Fizzy Lizzy', inplace=True)
df.loc[df.item_name =='Izze','item_name'] = 'Fizzy Lizzy'

# Q37. df의 데이터 중 choice_description 값이 NaN 인 데이터의 갯수를 구하여라
print(len(df[df['choice_description'].isnull()]))

# Q38. df의 데이터 중 choice_description 값이 NaN 인 데이터를 NoData 값으로 대체하라(loc 이용)
df.loc[df['choice_description'].isnull(), 'choice_description'] = 'NoData'
import numpy as np
df['choice_description'].replace(np.nan, 'NoData', inplace=True)

# Q39. df의 데이터 중 choice_description 값에 Black이 들어가는 경우를 인덱싱하라
df[df['choice_description'].str.contains('Black')]

# Q40. df의 데이터 중 choice_description 값에 Vegetables 들어가지 않는 경우의 갯수를 출력하라
print(len(df[~df['choice_description'].str.contains('Vegetables')]))

# Q41. df의 데이터 중 item_name 값이 N으로 시작하는 데이터를 모두 추출하라
df[df['item_name'].str.startswith('N')]

# Q42. df의 데이터 중 item_name 값의 단어갯수가 15개 이상인 데이터를 인덱싱하라
df[df['item_name'].map(lambda x:len(x))>=15]
df[df['item_name'].str.len()>=15]

# Q43. df의 데이터 중 new_price값이 lst에 해당하는 경우의 데이터 프레임을 구하고 그 갯수를 출력하라
lst =[1.69, 2.39, 3.39, 4.45, 9.25, 10.98, 11.75, 16.98]
print(len(df[df['new_price'].isin(lst)]))

  
