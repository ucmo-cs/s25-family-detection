import pandas as pd
import numpy as np
df = pd.read_csv('../ActualData/pairs.csv')

#extract family ids from img1 column
df['family_id'] = df['img1'].str.split('/').str[0]

#get unique families
unique_families = df['family_id'].unique()
np.random.seed(42)
np.random.shuffle(unique_families)

#split families 80/20
split_idx = int(len(unique_families) * 0.8)
train_families = set(unique_families[:split_idx])
eval_families = set(unique_families[split_idx:])

#split dataframe by family membership
train_df = df[df['family_id'].isin(train_families)].drop('family_id', axis=1)
eval_df = df[df['family_id'].isin(eval_families)].drop('family_id', axis=1)

train_df.to_csv('../ActualData/train.csv', index=False)
eval_df.to_csv('../ActualData/eval.csv', index=False)

print(f"Train families: {len(train_families)}, pairs {len(train_df)}")
print(f"Eval families: {len(eval_families)}, pairs {len(eval_df)}")