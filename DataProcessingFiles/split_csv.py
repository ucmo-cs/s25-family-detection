import pandas as pd
df = pd.read_csv('../ActualData/pairs.csv')

train_df = df.sample(frac=0.8, random_state=42)
eval_df = df.drop(train_df.index)

train_df.to_csv('train.csv', index=False)
eval_df.to_csv('eval.csv', index=False)