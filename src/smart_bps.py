import pandas as pd
import os
import glob
from src.bps_strategy import load_price_data, load_data, calculate_bps
from src.broker_clustering import run_analysis
from src.batch_clustering import identify_accumulator_cluster

def run_smart_bps(stock_id):
    print(f"\n{'='*40}")
    print(f"🚀 Running SMART BPS for Stock: {stock_id}")
    print(f"{'='*40}")

    # 1. Step 1: Run Clustering to find the 'Smart' brokers
    # Note: run_analysis in broker_clustering uses its own internal window for feature extraction.
    clustered_df = run_analysis(stock_id)
    if clustered_df is None or 'cluster' not in clustered_df.columns:
        print(f"Clustering failed for {stock_id}")
        return
        
    best_cluster_id, _ = identify_accumulator_cluster(clustered_df)
    smart_broker_ids = clustered_df[clustered_df['cluster'] == best_cluster_id]['securities_trader_id'].tolist()
    
    print(f"\nFound {len(smart_broker_ids)} smart brokers in Cluster {best_cluster_id}")

    # 2. Step 2: Load FULL Transaction Data (Not just the clustering window)
    # We load everything from StockBranch.parquet for this stock
    df_raw = load_data(stock_id)
    price_df = load_price_data(stock_id)
    
    if df_raw.empty:
        print("No transaction data found.")
        return

    # 3. Step 3: Calculate Smart BPS over the FULL available period
    print(f"Calculating Smart BPS for {len(df_raw['date'].unique())} days...")
    df_smart = df_raw[df_raw['securities_trader_id'].isin(smart_broker_ids)]
    smart_bps = calculate_bps(df_smart, price_df)

    # 4. Merge results
    if smart_bps.empty:
        print("BPS calculation resulted in empty dataframe.")
        return
        
    # SAVE TO CSV
    output_path = f'data/smart_bps_result_{stock_id}.csv'
    smart_bps.to_csv(output_path, index=False)
    print(f"✅ Results saved to {output_path} ({len(smart_bps)} rows)")

if __name__ == "__main__":
    run_smart_bps('2330')
