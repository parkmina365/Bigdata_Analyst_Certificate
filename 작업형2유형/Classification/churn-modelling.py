import pandas as pd 
trainData = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/churn/train.csv')
testData  = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/churn/test.csv')
subData  = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/churn/submission.csv',index_col=0)
