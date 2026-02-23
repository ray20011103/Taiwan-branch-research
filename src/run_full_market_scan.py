import pandas as pd
import os
import glob
from tqdm import tqdm
from broker_clustering import run_analysis as run_clustering
from smart_bps import run_smart_bps
from bps_strategy import load_price_data, load_data, calculate_bps
from batch_clustering import identify_accumulator_cluster

# Top 50 Stocks from scan
TARGET_STOCKS = [
    '3013', '2365', '3450', '6558', '1815', '8096', '2408', '4931', '1514', '6215', 
    '2486', '4510', '6140', '3047', '3312', '4909', '2615', '4979', '2359', '8054', 
    '3363', '4991', '3706', '3163', '8028', '2609', '6117', '1503', '2374', '4303', 
    '2543', '8064', '1540', '6148', '5426', '8111', '2363', '5443', '4562', '2464', 
    '2312', '3379', '5251', '3535', '1519', '3062', '6442', '6462', '2468', '3376'
]

def process_stock(stock_id):
    output_path = f'data/smart_bps_result_{stock_id}.csv'
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1024:
        return

    print(f"\nProcessing {stock_id}...")
    
    # 2. Run Clustering (using optimized Rolling Window + PCA)
    # This will generate data/broker_clusters_{stock_id}.csv
    clustered_df = run_clustering(stock_id)
    if clustered_df is None:
        return
        
    # 3. Identify Smart Cluster
    best_cluster_id, _ = identify_accumulator_cluster(clustered_df)
    if best_cluster_id is None:
        print("Could not identify accumulator cluster.")
        return
        
    smart_broker_ids = clustered_df[clustered_df['cluster'] == best_cluster_id]['securities_trader_id'].tolist()
    
    # 4. Calculate Smart BPS (Full History: 2024-2025)
    # Load Price
    price_df = load_price_data(stock_id)
    
    # Load Transactions (Full History for Backtest)
    file_path = 'data/StockBranch.parquet'
    try:
        df_raw = pd.read_parquet(file_path, filters=[('CommodityId', '==', str(stock_id))])
        df_raw = df_raw.rename(columns={'Date': 'date', 'CommodityId': 'stock_id', 'SecuritiesTraderId': 'securities_trader_id', 'Price': 'price', 'Buy': 'buy', 'Sell': 'sell'})
        df_raw['date'] = df_raw['date'].astype(str) # Ensure string format
        # Filter for 2024-2025
        df_raw = df_raw[df_raw['date'] >= '2024-01-01']
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if df_raw.empty: return

    # Calculate BPS
    # Original
    original_bps = calculate_bps(df_raw, price_df)
    
    # Smart (Filtered by Smart Brokers)
    df_smart = df_raw[df_raw['securities_trader_id'].isin(smart_broker_ids)]
    smart_bps = calculate_bps(df_smart, price_df)
    
    # Merge and Save
    if not original_bps.empty and not smart_bps.empty:
        comparison = original_bps[['date', 'price', 'bps_factor']].rename(columns={'bps_factor': 'original_bps'})
        comparison = comparison.merge(smart_bps[['date', 'bps_factor']].rename(columns={'bps_factor': 'smart_bps'}), on='date', how='left').fillna(0)
        
        comparison.to_csv(output_path, index=False)
        print(f"Saved Smart BPS to {output_path}")

def run_full_scan():
    print(f"Starting Full Market Scan for {len(TARGET_STOCKS)} stocks...")
    
    # Use tqdm
    for stock in tqdm(TARGET_STOCKS):
        try:
            process_stock(stock)
        except Exception as e:
            print(f"Failed on {stock}: {e}")

if __name__ == "__main__":
    run_full_scan()
