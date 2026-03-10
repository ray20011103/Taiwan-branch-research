import pandas as pd
import os
from src.bps_strategy import load_price_data, load_data, calculate_bps
from src.broker_clustering import run_analysis

def identify_specific_clusters(clustered_df):
    """
    Identifies Informed Clusters (Smart Money) and Momentum Clusters (Day Traders).
    - Informed: Profitable + High C3 Score (combination of profit and holding power).
    - Momentum: Low holding power but high turnover (Cluster 1).
    """
    if clustered_df is None or 'cluster' not in clustered_df.columns:
        return [], None
        
    summary = clustered_df.groupby('cluster')[['overnight_ratio', 'frequency', 'est_profit', 'log_avg_daily_vol']].mean()
    
    # Calculate C3 Score (Ranking based on Profit + Holding power)
    summary['profit_rank'] = summary['est_profit'].rank(ascending=True)
    summary['holding_rank'] = summary['overnight_ratio'].rank(ascending=True)
    summary['c3_score'] = summary['profit_rank'] + summary['holding_rank']
    
    # 1. Informed Clusters: All clusters with positive profit and score >= median
    informed_ids = summary[
        (summary['est_profit'] > 0) & 
        (summary['c3_score'] >= summary['c3_score'].median())
    ].index.tolist()
    
    # 2. Momentum Cluster (C1): Lowest overnight ratio (highest turnover)
    c1_id = summary['overnight_ratio'].idxmin()
    
    print("\n--- Optimized Cluster Identification (Informed + Momentum) ---")
    print(summary[['overnight_ratio', 'est_profit', 'c3_score']])
    print(f"Targeting Informed Clusters: {informed_ids}, Momentum Cluster: {c1_id}")
    
    return informed_ids, c1_id

def run_smart_bps(stock_id):
    print(f"\n--- Generating Multi-Cluster BPS for Stock: {stock_id} ---")

    # 1. Run Clustering
    clustered_df = run_analysis(stock_id)
    if clustered_df is None: return
        
    c1_id, c3_id = identify_specific_clusters(clustered_df)
    
    c1_brokers = clustered_df[clustered_df['cluster'] == c1_id]['securities_trader_id'].tolist()
    c3_brokers = clustered_df[clustered_df['cluster'] == c3_id]['securities_trader_id'].tolist()
    
    print(f"Cluster 1 (Story/DayTrade): {len(c1_brokers)} brokers")
    print(f"Cluster 3 (Quant/Smart): {len(c3_brokers)} brokers")

    # 2. Load Data
    df_raw = load_data(stock_id)
    price_df = load_price_data(stock_id)
    
    if df_raw.empty: return

    # 3. Calculate BPS for each
    df_c1 = df_raw[df_raw['securities_trader_id'].isin(c1_brokers)]
    df_c3 = df_raw[df_raw['securities_trader_id'].isin(c3_brokers)]
    
    bps_c1 = calculate_bps(df_c1, price_df).rename(columns={'bps_factor': 'bps_c1'})
    bps_c3 = calculate_bps(df_c3, price_df).rename(columns={'bps_factor': 'bps_c3'})

    # 4. Merge results
    final_bps = pd.merge(bps_c3, bps_c1[['date', 'bps_c1']], on='date', how='left')
    
    output_path = f'data/smart_bps_result_{stock_id}.csv'
    final_bps.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    run_smart_bps('6215')
