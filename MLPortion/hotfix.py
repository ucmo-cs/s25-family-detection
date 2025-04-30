#ensuring proper file directory storage and finding
import pandas as pd
import os
df = pd.read_csv("cleaned_pairs.csv")
df["img1"] = df["img1"].apply(lambda x: os.path.relpath(x, start = "FIDs/FIDs"))
df["img2"] = df["img2"].apply(lambda x: os.path.relpath(x, start = "FIDs/FIDs"))
df_cleaned = df[~df['img1'].str.startswith("..") & ~df['img2'].str.startswith("..")]
df.to_csv("final.csv", index = False)
