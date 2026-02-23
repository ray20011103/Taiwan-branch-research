import pandas as pd
import numpy as np
import os
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from broker_clustering import load_data, extract_features
from bps_strategy import load_price_data, calculate_bps

# Configuration
TARGET_STOCKS = ['3706', '3450', '1536', '3013', '6140', '6215']

def run_comparison():
    print(f"--- Batch Model Comparison: K-Means vs DBSCAN ---")
    
    results = []
    
    for stock_id in TARGET_STOCKS:
        print(f"\nProcessing {stock_id}...")
        
        # 1. Load Data
        df_raw = load_data(stock_id)
        price_df = load_price_data(stock_id)
        if df_raw.empty or price_df.empty:
            continue
            
        features_df = extract_features(df_raw)
        
        # Filter active brokers
        active_brokers = features_df[features_df['transaction_days'] >= 3].copy()
        if len(active_brokers) < 10: continue
        
        X = active_brokers[['frequency', 'overnight_ratio', 'log_avg_daily_vol']]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 2. Model A: K-Means (Baseline)
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        active_brokers['kmeans_cluster'] = kmeans.fit_predict(X_scaled)
        
        # Identify Accumulator Cluster (Highest Overnight Ratio)
        kmeans_summary = active_brokers.groupby('kmeans_cluster')['overnight_ratio'].mean()
        kmeans_target_cluster = kmeans_summary.idxmax()
        kmeans_brokers = active_brokers[active_brokers['kmeans_cluster'] == kmeans_target_cluster]['securities_trader_id'].tolist()

        # 3. Model B: DBSCAN (Challenger)
        # Using previous params: eps=0.5, min_samples=5
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        active_brokers['dbscan_cluster'] = dbscan.fit_predict(X_scaled)
        
        # Identify Target: Noise points (-1) WITH high overnight ratio
        dbscan_brokers = active_brokers[
            (active_brokers['dbscan_cluster'] == -1) & 
            (active_brokers['overnight_ratio'] > 0.3)
        ]['securities_trader_id'].tolist()
        
        # 4. Calculate BPS for both
        def calc_bps_return(broker_ids):
            if not broker_ids:
                return 0.0
            
            subset = df_raw[df_raw['securities_trader_id'].isin(broker_ids)]
            bps_res = calculate_bps(subset, price_df)
            signal = bps_res.set_index('date')['bps_factor']
            
            # Merge with future return
            analysis_df = price_df[['date', 'close']].copy()
            analysis_df['date'] = analysis_df['date'].astype(str)
            analysis_df['future_return_5d'] = analysis_df['close'].shift(-5) / analysis_df['close'] - 1
            
            analysis_df = analysis_df.merge(signal.rename('bps'), on='date', how='left').fillna(0)
            
            # Avg return when Signal > 0
            # Filter for meaningful signals (> 10張 to avoid noise)
            active_days = analysis_df[analysis_df['bps'] > 10000] 
            if active_days.empty:
                return 0.0
            return active_days['future_return_5d'].mean()

        km_ret = calc_bps_return(kmeans_brokers)
        db_ret = calc_bps_return(dbscan_brokers)
        
        overlap_count = len(set(kmeans_brokers) & set(dbscan_brokers))
        
        results.append({
            'stock_id': stock_id,
            'kmeans_return': km_ret * 100,
            'dbscan_return': db_ret * 100,
            'kmeans_brokers': len(kmeans_brokers),
            'dbscan_brokers': len(dbscan_brokers),
            'overlap': overlap_count,
            'winner': 'K-Means' if km_ret > db_ret else 'DBSCAN'
        })
        
        print(f"-> Winner: {'K-Means' if km_ret > db_ret else 'DBSCAN'} ({km_ret*100:.2f}% vs {db_ret*100:.2f}%)")

    # Summary
    if results:
        df_res = pd.DataFrame(results)
        print("\n" + "="*50)
        print("BATCH MODEL COMPARISON SUMMARY")
        print("="*50)
        print(df_res[['stock_id', 'winner', 'kmeans_return', 'dbscan_return', 'overlap']].to_string(index=False))
        
        print("\nAverage K-Means Return:", df_res['kmeans_return'].mean())
        print("Average DBSCAN Return:", df_res['dbscan_return'].mean())

if __name__ == "__main__":
    run_comparison()
