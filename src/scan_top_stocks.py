import pandas as pd
import os

DATA_PATH = 'data/StockBranch.parquet'

def get_top_stocks(n=50):
    print(f"Scanning {DATA_PATH} for top {n} active stocks...")
    if not os.path.exists(DATA_PATH):
        print("Data file not found.")
        return []

    # Read only CommodityId column for speed
    df = pd.read_parquet(DATA_PATH, columns=['CommodityId'])
    
    # Count occurrences
    top_stocks = df['CommodityId'].value_counts().head(n)
    
    print("\n--- Top Active Stocks ---")
    print(top_stocks)
    
    stock_list = top_stocks.index.tolist()
    print(f"\nStock List ({len(stock_list)}):")
    print(stock_list)
    
    return stock_list

if __name__ == "__main__":
    get_top_stocks(50)

