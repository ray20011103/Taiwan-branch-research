import pandas as pd
import os
from scipy import stats
import numpy as np

def verify_high_pressure_hypothesis():
    print("--- Hypothesis Verification: Smart BPS + Hedging Flow > 5000 Lots ---")
    
    # 1. Load Data
    flow_path = 'data/issuer_hedging_flow_daily.csv'
    market_index_path = 'data/market_index.parquet'
    
    if not os.path.exists(flow_path):
        print("Hedging flow data not found. Run calculate_hedging_flow.py first.")
        return
        
    flow_df = pd.read_csv(flow_path)
    flow_df['日期'] = pd.to_datetime(flow_df['日期'])
    flow_df['標的代號'] = flow_df['標的代號'].astype(str)
    
    mkt_df = pd.read_parquet(market_index_path)
    mkt_df['date'] = pd.to_datetime(mkt_df['date'])
    mkt_df = mkt_df[['date', 'market_ret']].rename(columns={'market_ret': 'mkt_ret'})
    
    # 2. Integrate Smart BPS
    stocks = flow_df['標的代號'].unique()
    bps_list = []
    for sid in stocks:
        path = f'data/smart_bps_result_{sid}.csv'
        if os.path.exists(path):
            tmp = pd.read_csv(path)
            tmp['date'] = pd.to_datetime(tmp['date'])
            tmp['stock_id'] = str(sid)
            if 'bps_factor' in tmp.columns:
                tmp = tmp.rename(columns={'bps_factor': 'smart_bps'})
            bps_list.append(tmp[['date', 'stock_id', 'smart_bps']])
            
    if not bps_list:
        print("No Smart BPS data found.")
        return
        
    bps_all = pd.concat(bps_list)
    combined = pd.merge(flow_df, bps_all, left_on=['日期', '標的代號'], right_on=['date', 'stock_id'], how='inner')
    combined = pd.merge(combined, mkt_df, on='date', how='left')
    
    # 3. Exclude April 2025 Extreme Event
    combined = combined[~combined['日期'].between('2025-04-07', '2025-04-15')]
    
    # 4. Calculate Alpha and Next Day Return
    combined = combined.sort_values(['標的代號', '日期'])
    combined['stock_ret'] = combined.groupby('標的代號')['標的收盤價'].pct_change()
    combined['alpha_ret'] = combined['stock_ret'] - combined['mkt_ret']
    combined['next_alpha'] = combined.groupby('標的代號')['alpha_ret'].shift(-1)
    combined = combined.dropna(subset=['next_alpha'])
    
    # 5. Define the High Pressure Group
    h_threshold = 5000
    mask_high_pressure = (combined['smart_bps'] > 0) & (combined['hedging_flow_lots'] > h_threshold)
    
    group_high = combined[mask_high_pressure]['next_alpha']
    group_control = combined[~mask_high_pressure]['next_alpha']
    
    # 6. Report
    print(f"\n--- Analysis Results (Threshold: {h_threshold} Lots) ---")
    print(f"High Pressure Group Sample Size (n): {len(group_high)}")
    
    if len(group_high) > 0:
        print(f"Mean Next-Day Alpha: {group_high.mean()*100:.2f}%")
        print(f"Win Rate (Alpha > 0): {(group_high > 0).mean()*100:.2f}%")
        
        print(f"\nControl Group (Normal) Mean Alpha: {group_control.mean()*100:.2f}%")
        
        if len(group_high) > 1:
            t_stat, p_val = stats.ttest_ind(group_high, group_control, equal_var=False)
            print(f"\nStatistical Significance (P-Value): {p_val:.4f}")
            if p_val < 0.05:
                print("Verdict: SIGNIFICANT! The high hedging pressure significantly boosts Alpha.")
            else:
                print("Verdict: Not Significant. The result might be due to chance or small sample size.")
    else:
        print("No samples found matching the high pressure criteria.")

    if not group_high.empty:
        print("\n--- Top High-Pressure Examples ---")
        examples = combined[mask_high_pressure].sort_values('next_alpha', ascending=False).head(5)
        print(examples[['日期', '標的代號', 'hedging_flow_lots', 'smart_bps', 'next_alpha']])

if __name__ == "__main__":
    verify_high_pressure_hypothesis()
