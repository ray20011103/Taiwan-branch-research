import pandas as pd
import os
import numpy as np
from src.broker_clustering import run_analysis
from src.batch_clustering import identify_accumulator_cluster

def get_smart_brokers(stock_id):
    """Reuses existing clustering logic to find smart brokers for a stock."""
    try:
        clustered_df = run_analysis(stock_id)
        if clustered_df is None or 'cluster' not in clustered_df.columns:
            return []
        
        best_cluster_id, _ = identify_accumulator_cluster(clustered_df)
        smart_brokers = clustered_df[clustered_df['cluster'] == best_cluster_id]['securities_trader_id'].tolist()
        return smart_brokers
    except Exception as e:
        print(f"Error clustering {stock_id}: {e}")
        return []

def run_smart_warrant_strategy():
    print("--- 🧠 Strategy: Smart Warrant Pressure (Clustering + Hedging) ---")
    
    # 1. Load basic results from previous run to get candidate stocks
    base_results_path = 'data/issuer_frontrunning_results.csv'
    if not os.path.exists(base_results_path):
        print("Base results not found. Run strategy_issuer_frontrunning.py first.")
        return
        
    base_df = pd.read_csv(base_results_path)
    base_df['date'] = pd.to_datetime(base_df['date'])
    
    # Pick top 50 stocks by hedge frequency to save time
    top_stocks = base_df['stock_id'].value_counts().head(50).index.astype(str).tolist()
    print(f"Analyzing Smart Brokers for Top {len(top_stocks)} stocks...")

    # 2. Build Smart Broker Map
    smart_broker_map = {}
    for sid in top_stocks:
        brokers = get_smart_brokers(sid)
        if brokers:
            smart_broker_map[sid] = brokers
            print(f"Stock {sid}: Found {len(brokers)} smart brokers.")

    # 3. Load Raw Warrant Trades
    w_trades = pd.read_parquet('data/分點進出.parquet')
    w_trades['date'] = pd.to_datetime(w_trades['日期'])
    w_trades['分點代號'] = w_trades['分點代號'].astype(str)
    
    w_specs = pd.read_parquet('data/權證條件.parquet')
    w_specs['標的代號'] = w_specs['標的代號'].astype(str)
    
    # 4. Filter for Smart Warrant Trades
    results = []
    for sid, smart_list in smart_broker_map.items():
        # Get warrants for this stock
        this_warrants = w_specs[w_specs['標的代號'] == sid]['權證代號'].unique()
        
        # Get trades by smart brokers in these warrants
        smart_trades = w_trades[
            (w_trades['權證代號'].isin(this_warrants)) & 
            (w_trades['分點代號'].isin(smart_list))
        ].copy()
        
        if smart_trades.empty:
            continue
            
        # Merge with Delta
        smart_merged = pd.merge(
            smart_trades,
            w_specs[['日期', '權證代號', '最新執行比例', 'IVDelta值']],
            left_on=['date', '權證代號'],
            right_on=['日期', '權證代號'],
            how='inner'
        )
        
        # Focus on Calls
        smart_merged = smart_merged[smart_merged['IVDelta值'] > 0]
        smart_merged['smart_hedge_lots'] = (smart_merged['買張'] - smart_merged['賣張']) * smart_merged['最新執行比例'] * smart_merged['IVDelta值']
        
        # Aggregate daily
        daily_smart = smart_merged.groupby('date')['smart_hedge_lots'].sum().reset_index()
        daily_smart['stock_id'] = sid
        results.append(daily_smart)

    if not results:
        print("No smart warrant trades found.")
        return
        
    smart_hedge_df = pd.concat(results)
    
    # 5. Merge with Base Performance Data
    base_df['stock_id'] = base_df['stock_id'].astype(str)
    smart_hedge_df['stock_id'] = smart_hedge_df['stock_id'].astype(str)
    
    final_df = pd.merge(
        base_df,
        smart_hedge_df,
        on=['date', 'stock_id'],
        how='left'
    ).fillna(0)
    
    # Calculate Smart Pressure Ratio
    final_df['smart_pressure_ratio'] = (final_df['smart_hedge_lots'] / final_df['avg_lots_20d']) * 100
    
    # 6. Performance Analysis
    print("--- Strategy Comparison ---")
    
    # Define Tiers for Smart Pressure
    high_smart = final_df[final_df['smart_pressure_ratio'] > 2].copy() # Lower threshold for smart money
    
    print(f"Total Events: {len(base_df)}")
    print(f"Events with Smart Broker Warrant Buying: {len(high_smart)}")
    
    if not high_smart.empty:
        win_rate = (high_smart['next_day_ret'] > 0).mean()
        avg_ret = high_smart['next_day_ret'].mean()
        print(f"--- Smart Warrant Pressure Results (Ratio > 2%) ---")
        print(f"Win Rate: {win_rate*100:.2f}%")
        print(f"Average Next Day Return: {avg_ret*100:.4f}%")
        
        # Compare with non-smart high pressure
        others = final_df[(final_df['hedge_pressure_ratio'] > 5) & (final_df['smart_pressure_ratio'] <= 0)]
        print(f"--- Non-Smart High Pressure Results (Ratio > 5%) ---")
        print(f"Win Rate: {(others['next_day_ret'] > 0).mean()*100:.2f}%")
        print(f"Average Next Day Return: {others['next_day_ret'].mean()*100:.4f}%")

    final_df.to_csv('data/smart_warrant_results.csv', index=False)
    print("Full results saved to data/smart_warrant_results.csv")

if __name__ == "__main__":
    run_smart_warrant_strategy()
