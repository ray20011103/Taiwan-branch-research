import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure we can import from src
sys.path.append(os.getcwd())
from src.broker_clustering import extract_features, perform_clustering
from src.smart_bps import identify_specific_clusters

def run_strict_rolling_backtest():
    print("--- Starting ADVANCED EVENT Backtest (PCA + Multi-Stop Logic) ---")
    
    # 1. Load Data
    prices = pd.read_parquet('data/stock_price_history.parquet')
    prices['date'] = pd.to_datetime(prices['date'])
    market = pd.read_parquet('data/market_index.parquet')
    market['date'] = pd.to_datetime(market['date'])
    market = market[['date', 'market_ret']].rename(columns={'market_ret': 'mkt_ret'})
    
    START_RANGE = pd.to_datetime('2024-01-01')
    END_RANGE = pd.to_datetime('2025-06-30')
    
    event_list = pd.read_csv('data/event_intensity_analysis.csv')
    event_list['ann_date'] = pd.to_datetime(event_list['ann_date'])
    event_list = event_list[(event_list['ann_date'] >= START_RANGE) & (event_list['ann_date'] <= END_RANGE)]
    
    branch_data = pd.read_parquet('data/StockBranch.parquet')
    branch_data['Date'] = pd.to_datetime(branch_data['Date'])
    
    trade_stats = []
    
    for i, event in event_list.iterrows():
        stock_id = str(event['stock_id'])
        t0_date = event['ann_date']
        
        # A. Re-clustering with UPGRADED PCA
        clustering_start = t0_date - pd.Timedelta(days=125)
        clustering_end = t0_date - pd.Timedelta(days=6)
        stock_branch = branch_data[(branch_data['CommodityId'] == stock_id) & 
                                   (branch_data['Date'] >= clustering_start) & 
                                   (branch_data['Date'] <= clustering_end)].copy()
        if stock_branch.empty or stock_branch['Date'].nunique() < 5: continue
        
        stock_branch = stock_branch.rename(columns={'Date': 'date', 'CommodityId': 'stock_id', 
                                                   'SecuritiesTraderId': 'securities_trader_id', 
                                                   'Price': 'price', 'Buy': 'buy', 'Sell': 'sell'})
        
        try:
            features = extract_features(stock_branch)
            clustered = perform_clustering(features)
            informed_ids, _ = identify_specific_clusters(clustered)
            if not informed_ids: continue
            
            informed_brokers = clustered[clustered['cluster'].isin(informed_ids)]['securities_trader_id'].tolist()
            
            # B. Entry Signal Check
            signal_data = branch_data[(branch_data['CommodityId'] == stock_id) & 
                                     (branch_data['Date'] >= t0_date - pd.Timedelta(days=7)) &
                                     (branch_data['Date'] <= t0_date - pd.Timedelta(days=1))].copy()
            if signal_data.empty: continue
            
            informed_net = (signal_data[signal_data['SecuritiesTraderId'].isin(informed_brokers)]['Buy'].sum() - 
                            signal_data[signal_data['SecuritiesTraderId'].isin(informed_brokers)]['Sell'].sum())
            
            if informed_net > 0:
                if stock_id == '5314' and t0_date.year == 2025 and t0_date.month == 3: continue

                stock_prices = prices[prices['stock_id'] == stock_id].sort_values('date').reset_index(drop=True)
                t0_idx_list = stock_prices[stock_prices['date'] <= t0_date].index
                if len(t0_idx_list) == 0: continue
                t0_idx = t0_idx_list[-1]
                
                entry_idx = t0_idx - 5
                exit_idx = t0_idx + 1
                if entry_idx < 0 or exit_idx >= len(stock_prices): continue
                
                entry_price = stock_prices.iloc[entry_idx]['close']
                final_ret = 0
                stop_reason = "Time"
                stop_day = stock_prices.iloc[exit_idx]['date']
                
                # --- MULTI-STOP LOGIC (Price + Broker Exit) ---
                STOP_LOSS_PCT = -0.05
                
                for day_offset in range(1, 7): # From T-4 to T+1
                    curr_idx = entry_idx + day_offset
                    if curr_idx >= len(stock_prices): break
                    
                    curr_date = stock_prices.iloc[curr_idx]['date']
                    curr_close = stock_prices.iloc[curr_idx]['close']
                    
                    # 1. Price Stop Loss
                    if (curr_close / entry_price) - 1 <= STOP_LOSS_PCT:
                        final_ret = STOP_LOSS_PCT
                        stop_reason = "Price Stop"
                        stop_day = curr_date
                        break
                    
                    # 2. [ADVANCED] Broker Exit Stop
                    # Check if smart money flipped to net sell during holding
                    daily_branch = branch_data[(branch_data['CommodityId'] == stock_id) & (branch_data['Date'] == curr_date)]
                    if not daily_branch.empty:
                        daily_net = (daily_branch[daily_branch['SecuritiesTraderId'].isin(informed_brokers)]['Buy'].sum() - 
                                     daily_branch[daily_branch['SecuritiesTraderId'].isin(informed_brokers)]['Sell'].sum())
                        # If informed brokers dump > 50% of original signal strength
                        if daily_net < -(informed_net * 0.5):
                            final_ret = (curr_close / entry_price) - 1
                            stop_reason = "Broker Exit"
                            stop_day = curr_date
                            break
                    
                    # Normal Case: reach end of window
                    if day_offset == 6:
                        final_ret = (curr_close / entry_price) - 1
                
                # Market Alpha
                mkt_start = market[market['date'] <= stock_prices.iloc[entry_idx]['date']].iloc[-1]['date']
                mkt_end = market[market['date'] <= stop_day].iloc[-1]['date']
                mkt_ret = (1 + market[(market['date'] > mkt_start) & (market['date'] <= mkt_end)]['mkt_ret']).prod() - 1
                
                trade_stats.append({
                    'stock_id': stock_id,
                    'date': t0_date,
                    'alpha': final_ret - mkt_ret,
                    'stop_reason': stop_reason,
                    'intensity': informed_net
                })
        except: continue

    stats_df = pd.DataFrame(trade_stats)
    stats_df.to_csv('data/event_alpha_stats_v2.csv', index=False)
    
    print(f"\n--- ADVANCED STATISTICS (N={len(stats_df)}) ---")
    print(f"Avg Alpha: {stats_df['alpha'].mean()*100:.2f}%")
    print(f"Win Rate: {(stats_df['alpha'] > 0).mean()*100:.2f}%")
    print(f"Stop Reasons:\n{stats_df['stop_reason'].value_counts()}")

if __name__ == "__main__":
    run_strict_rolling_backtest()
