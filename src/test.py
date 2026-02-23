import pandas as pd

df = pd.read_parquet("data/StockBranch.parquet")

df['Date'] = pd.to_datetime(df['Date'])

uniq_dates = pd.Series(df['Date'].dt.date.unique())
uniq_dates.to_csv('unique_dates.csv', index=False)