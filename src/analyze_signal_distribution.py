import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# Top 50 Active Stocks (Same as backtest)
TARGET_STOCKS = [
    '3013', '2365', '3450', '6558', '1815', '8096', '2408', '4931', '1514', '6215', 
    '2486', '4510', '6140', '3047', '3312', '4909', '2615', '4979', '2359', '8054', 
    '3363', '4991', '3706', '3163', '8028', '2609', '6117', '1503', '2374', '4303', 
    '2543', '8064', '1540', '6148', '5426', '8111', '2363', '5443', '4562', '2464', 
    '2312', '3379', '5251', '3535', '1519', '3062', '6442', '6462', '2468', '3376'
]

REVENUE_DB = 'data/revenue_announcements.parquet'

def get_price_idx(bps_df, target_date):
    """Finds index of date or closest previous trading day"""
    target_dt = pd.to_datetime(target_date)
    bps_df['date_dt'] = pd.to_datetime(bps_df['date'])
    
    # Strict match first
    match = bps_df[bps_df['date_dt'] == target_dt]
    if not match.empty:
        return match.index[0]
    
    # Backward search (if announced on Sunday, look at Friday)
    prev_dates = bps_df[bps_df['date_dt'] < target_dt]
    if not prev_dates.empty:
        return prev_dates.index[-1]
    return None

def analyze_timing():
    print("--- Analyzing Smart BPS Front-Running Timing Distribution ---")
    
    if not os.path.exists(REVENUE_DB):
        print("Revenue DB missing.")
        return

    rev_df = pd.read_parquet(REVENUE_DB)
    rev_df = rev_df[rev_df['stock_id'].isin(TARGET_STOCKS)]
    # Filter for 2024-2025
    rev_df = rev_df[rev_df['announcement_date'] >= '2024-01-01']
    
    # Dictionary to store signals: {T-x : [list of smart_bps values]}
    # Analyzing T-20 to T-0
    LOOKBACK = 20
    timing_stats = {i: [] for i in range(-LOOKBACK, 1)}
    
    valid_events = 0
    
    for stock_id in tqdm(TARGET_STOCKS, desc="Scanning Stocks"):
        bps_path = f'data/smart_bps_result_{stock_id}.csv'
        if not os.path.exists(bps_path):
            continue
            
        bps_df = pd.read_csv(bps_path)
        # Ensure BPS is numeric
        bps_df['smart_bps'] = pd.to_numeric(bps_df['smart_bps'], errors='coerce').fillna(0)
        
        stock_events = rev_df[rev_df['stock_id'] == stock_id]
        
        for _, event in stock_events.iterrows():
            ann_date = event['announcement_date']
            exit_idx = get_price_idx(bps_df, ann_date)
            
            if exit_idx is None or exit_idx < LOOKBACK:
                continue
                
            valid_events += 1
            
            # Extract T-20 to T-0
            for i in range(-LOOKBACK, 1):
                # exit_idx is T-0. so T-5 is exit_idx - 5
                current_idx = exit_idx + i
                if 0 <= current_idx < len(bps_df):
                    val = bps_df.iloc[current_idx]['smart_bps']
                    # We focus on BUY signals (Positive BPS)
                    # If val is negative (selling), we treat it as 0 for "Buying Pressure" analysis
                    # Or we can keep it raw to see net flow. Let's look at raw Mean first.
                    timing_stats[i].append(val)

    # Aggregation & Visualization
    print(f"\nAnalyzed {valid_events} Announcement Events.")
    print("\n[Timing Distribution: Average Smart BPS per Day]")
    print(f"{ 'Day':<6} | {'Avg Signal':<12} | {'Win Rate (>0)':<10} | {'Chart'}")
    print("-" * 60)
    
    max_val = 0
    results = []
    
    for i in range(-LOOKBACK, 1):
        vals = timing_stats[i]
        if not vals: continue
        
        avg_val = np.mean(vals)
        pos_rate = sum(1 for v in vals if v > 0) / len(vals) * 100
        results.append((i, avg_val, pos_rate))
        max_val = max(max_val, abs(avg_val))
    
    # Sort by day
    results.sort(key=lambda x: x[0])
    
    for day, avg, rate in results:
        # Simple bar chart
        bar_len = int((abs(avg) / max_val) * 30) if max_val > 0 else 0
        bar = "#" * bar_len
        # Color logic for print? (Maybe just sign)
        sign = "+" if avg > 0 else "-"
        
        day_label = f"T{day}" if day < 0 else "Announce"
        print(f"{day_label:<6} | {avg:12.0f} | {rate:9.1f}% | {bar}")

if __name__ == "__main__":
    analyze_timing()
