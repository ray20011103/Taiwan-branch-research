import pandas as pd
import os
import sys

# Add root to sys.path to allow importing from src
sys.path.append(os.getcwd())

from src.smart_bps import run_smart_bps

def get_top_200_stocks():
    """Dynamically fetch Top 200 stocks by trading value in 2025."""
    price_history_path = 'data/stock_price_history.parquet'
    if not os.path.exists(price_history_path):
        return []
        
    df = pd.read_parquet(price_history_path)
    df['date'] = pd.to_datetime(df['date'])
    df_2025 = df[df['date'] >= '2025-01-01']
    
    top_200 = df_2025.groupby('stock_id')['volume_value_1k'].sum().sort_values(ascending=False).head(200).index.tolist()
    return top_200

def batch_process_top_stocks():
    print("--- EXPANDED Batch Processing (Top 200 + Special Targets) ---")
    
    top_200 = get_top_200_stocks()
    special_targets = ['6215', '4510', '6140'] 
    
    all_targets = list(set(top_200 + special_targets))
    all_targets = sorted(all_targets)
    
    print(f"Total Unique Targets: {len(all_targets)}")
    
    success_count = 0
    fail_count = 0
    
    for i, stock_id in enumerate(all_targets):
        stock_id = str(stock_id)
        # Check if already processed with NEW format (has bps_c1 and bps_c3)
        output_path = f'data/smart_bps_result_{stock_id}.csv'
        
        # We force refresh to ensure Performance-Based identification is applied to ALL
        print(f"\n[{i+1}/{len(all_targets)}] Processing {stock_id}...")
        
        try:
            run_smart_bps(stock_id)
            success_count += 1
        except Exception as e:
            print(f"❌ Error processing {stock_id}: {e}")
            fail_count += 1

    print("\n" + "="*40)
    print("EXPANDED BATCH PROCESS SUMMARY")
    print("="*40)
    print(f"Total Success: {success_count}")
    print(f"Total Failed: {fail_count}")

if __name__ == "__main__":
    batch_process_top_stocks()
