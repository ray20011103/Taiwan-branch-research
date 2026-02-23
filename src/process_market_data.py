import pandas as pd
import os

CSV_PATH = 'data/market.csv'
PARQUET_PATH = 'data/market_index.parquet'

def process_market_index():
    print(f"Reading market index from {CSV_PATH}...")
    
    if not os.path.exists(CSV_PATH):
        print("Market CSV not found.")
        return

    try:
        # Read CSV
        # 證券代碼,年月日,收盤價(元)
        # Y9999 加權指數,20150105,9274.11
        df = pd.read_csv(CSV_PATH)
        
        # Rename columns
        df = df.rename(columns={
            '年月日': 'date',
            '收盤價(元)': 'close'
        })
        
        # Parse Date (YYYYMMDD -> YYYY-MM-DD)
        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
        
        # Ensure close is numeric
        # Handle potential comma or other non-numeric chars if any
        if df['close'].dtype == 'object':
             df['close'] = df['close'].astype(str).str.replace(',', '').astype(float)
        
        # Sort by date
        df = df.sort_values('date')
        
        # Calculate Daily Return
        df['market_ret'] = df['close'].pct_change()
        
        # Fill NaN for the first day
        df['market_ret'] = df['market_ret'].fillna(0)
        
        print(f"Processed {len(df)} records.")
        print(f"Date Range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(df.head())
        
        # Save to Parquet
        print(f"Saving to {PARQUET_PATH}...")
        df.to_parquet(PARQUET_PATH, index=False)
        print("Done.")
        
    except Exception as e:
        print(f"Error processing market data: {e}")

if __name__ == "__main__":
    process_market_index()
