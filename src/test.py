import pandas as pd

df = pd.read_parquet("data/權證條件.parquet")
print(df.head(), df.columns.tolist())
