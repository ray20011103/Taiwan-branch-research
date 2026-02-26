import pandas as pd
import os
import sys

# Add root to sys.path to allow importing from src
sys.path.append(os.getcwd())

from src.smart_bps import run_smart_bps

def batch_process_top_stocks():
    print("--- 🏭 Batch Processing Smart BPS for Top 50 Stocks ---")
    
    stock_history_path = 'data/stock_price_history.parquet'
    if not os.path.exists(stock_history_path):
        print("Price history file not found.")
        return
        
    df_prices = pd.read_parquet(stock_history_path)
    df_prices['date'] = pd.to_datetime(df_prices['date'])
    
    valid_prices = df_prices[df_prices['date'].between('2025-01-01', '2025-08-31')]
    top_stocks = valid_prices.groupby('stock_id')['volume_value_1k'].sum().sort_values(ascending=False).head(50).index.tolist()
    
    print(f"Top 50 Stocks identified. Starting processing...")
    
    success_count = 0
    for i, stock_id in enumerate(top_stocks):
        stock_id = str(stock_id)
        print(f"\n[{i+1}/50] Processing {stock_id}...")
        
        output_path = f'data/smart_bps_result_{stock_id}.csv'
        if os.path.exists(output_path):
            print(f"Result for {stock_id} already exists. Skipping.")
            success_count += 1
            continue
            
        try:
            run_smart_bps(stock_id)
            success_count += 1
        except Exception as e:
            print(f"❌ Error processing {stock_id}: {e}")

    print("\n--- Batch Process Finished ---")
    print(f"Successfully processed {success_count} / 50 stocks.")

if __name__ == "__main__":
    batch_process_top_stocks()
