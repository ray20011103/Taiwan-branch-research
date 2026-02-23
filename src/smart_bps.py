import pandas as pd
import os
import glob
from bps_strategy import load_price_data, load_data, calculate_bps
from broker_clustering import run_analysis
from batch_clustering import identify_accumulator_cluster

def run_smart_bps(stock_id):
    print(f"\n{'='*40}")
    print(f"🚀 Running SMART BPS for Stock: {stock_id}")
    print(f"{'='*40}")

    # 1. Step 1: Run Clustering to find the 'Smart' brokers
    clustered_df = run_analysis(stock_id)
    if clustered_df is None:
        return
        
    best_cluster_id, _ = identify_accumulator_cluster(clustered_df)
    smart_broker_ids = clustered_df[clustered_df['cluster'] == best_cluster_id]['securities_trader_id'].tolist()
    
    print(f"\nFound {len(smart_broker_ids)} smart brokers in Cluster {best_cluster_id}")
    print(f"Top 5 Smart Brokers: {smart_broker_ids[:5]}")

    # 2. Step 2: Load Transaction Data
    df_raw = load_data(stock_id)
    price_df = load_price_data(stock_id)
    
    if df_raw.empty:
        print("No transaction data.")
        return

    # 3. Step 3: Calculate Original BPS (for comparison)
    print("\nCalculating Original BPS...")
    original_bps = calculate_bps(df_raw, price_df)
    
    # 4. Step 4: Calculate Smart BPS
    # We filter the raw transactions to ONLY include smart brokers
    print("\nCalculating Smart BPS (Filtered)...")
    df_smart = df_raw[df_raw['securities_trader_id'].isin(smart_broker_ids)]
    smart_bps = calculate_bps(df_smart, price_df)

    # 5. Merge and Compare
    comparison = original_bps[['date', 'price', 'bps_factor']].rename(columns={'bps_factor': 'original_bps'})
    comparison = comparison.merge(smart_bps[['date', 'bps_factor']].rename(columns={'bps_factor': 'smart_bps'}), on='date')
    
    # Calculate Signal Difference
    comparison['noise_removed'] = comparison['original_bps'] - comparison['smart_bps']
    
    print("\n--- Smart BPS vs Original BPS Comparison ---")
    print(comparison.tail(15).to_string(index=False))
    
    output_path = f'data/smart_bps_result_{stock_id}.csv'
    comparison.to_csv(output_path, index=False)
    print(f"\nResult saved to {output_path}")

if __name__ == "__main__":
    # Test with 3706 (神達), which had a very clear accumulator cluster
    run_smart_bps('1536')
