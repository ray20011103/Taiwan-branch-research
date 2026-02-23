import pandas as pd
import os
from tqdm import tqdm

TARGET_STOCKS = [
    '3013', '2365', '3450', '6558', '1815', '8096', '2408', '4931', '1514', '6215', 
    '2486', '4510', '6140', '3047', '3312', '4909', '2615', '4979', '2359', '8054', 
    '3363', '4991', '3706', '3163', '8028', '2609', '6117', '1503', '2374', '4303', 
    '2543', '8064', '1540', '6148', '5426', '8111', '2363', '5443', '4562', '2464', 
    '2312', '3379', '5251', '3535', '1519', '3062', '6442', '6462', '2468', '3376'
]

PATHS = {
    'branch': 'data/StockBranch.parquet',
    'revenue': 'data/revenue_announcements.parquet',
    'price': 'data/stock_price_history.parquet'
}

def diagnose():
    print("--- Data Range Diagnosis (2024 Coverage Check) ---")
    
    # 1. Check Revenue Announcements
    print("\n[1] Checking Revenue Announcements (2024)...")
    if os.path.exists(PATHS['revenue']):
        rev_df = pd.read_parquet(PATHS['revenue'])
        rev_2024 = rev_df[
            (rev_df['stock_id'].isin(TARGET_STOCKS)) & 
            (rev_df['announcement_date'] >= '2024-01-01') & 
            (rev_df['announcement_date'] < '2025-01-01')
        ]
        print(f"Total Revenue Events in 2024 for Target Stocks: {len(rev_2024)}")
        if len(rev_2024) > 0:
            print(f"Sample Dates: {rev_2024['announcement_date'].sample(5).tolist()}")
        else:
            print("WARNING: No revenue events found in 2024!")
    
    # 2. Check Price History
    print("\n[2] Checking Price History (2024)...")
    if os.path.exists(PATHS['price']):
        price_df = pd.read_parquet(PATHS['price'])
        price_df['date'] = pd.to_datetime(price_df['date'])
        price_2024 = price_df[
            (price_df['stock_id'].isin(TARGET_STOCKS)) & 
            (price_df['date'] >= '2024-01-01') & 
            (price_df['date'] < '2025-01-01')
        ]
        print(f"Total Price Records in 2024 for Target Stocks: {len(price_2024)}")
        print(f"Coverage: {price_2024['stock_id'].nunique()} / {len(TARGET_STOCKS)} stocks have data.")
    
    # 3. Check Smart BPS Results (Generated Files)
    print("\n[3] Checking Generated Smart BPS Results (2024)...")
    missing_2024 = []
    has_2024 = []
    
    for stock in tqdm(TARGET_STOCKS):
        path = f'data/smart_bps_result_{stock}.csv'
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                min_date = df['date'].min()
                if min_date < '2025-01-01':
                    has_2024.append(stock)
                else:
                    missing_2024.append(f"{stock} (Min: {min_date})")
            except:
                missing_2024.append(f"{stock} (Read Error)")
        else:
            missing_2024.append(f"{stock} (File Missing)")
            
    print(f"\nStocks with 2024 BPS Data: {len(has_2024)}")
    print(f"Stocks MISSING 2024 BPS Data: {len(missing_2024)}")
    if missing_2024:
        print("Sample missing:", missing_2024[:10])

if __name__ == "__main__":
    diagnose()
