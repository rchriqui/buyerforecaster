import pandas as pd

df=pd.read_csv('/Users/robin/Desktop/my_projects/buyerforecaster/artifacts/test.csv')
df.columns

df.info()

df['VisitorType'].value_counts()