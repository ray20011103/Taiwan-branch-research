import pandas as pd
import os
import glob
from datetime import datetime

def run_batch_alpha_analysis():
    print("--- Refined Batch Alpha Analysis (Source: Parquet, Target: Jan 2025) ---")
    
    # 1. Load Core Data
    if not os.path.exists('data/stock_price_history.parquet'):
        print("Price history not found.")
        return
    
    all_prices = pd.read_parquet('data/stock_price_history.parquet')
    all_prices['date'] = pd.to_datetime(all_prices['date'])
    
    if os.path.exists('data/market_index.parquet'):
        market_df = pd.read_parquet('data/market_index.parquet')
        market_df['date'] = pd.to_datetime(market_df['date'])
        market_df = market_df[['date', 'market_ret']].rename(columns={'market_ret': 'mkt_ret'})
    else:
        print("Market index not found.")
        return

    # 2. Identify Target Stocks (Based on available Smart BPS results)
    bps_files = glob.glob('data/smart_bps_result_*.csv')
    stock_ids = [f.split('_')[-1].split('.')[0] for f in bps_files]
    
    # 3. Load Accurate Announcements from Parquet
    if not os.path.exists('data/revenue_announcements.parquet'):
        print("Accurate revenue announcements parquet not found.")
        return
        
    ann_df = pd.read_parquet('data/revenue_announcements.parquet')
    ann_df['announcement_date'] = pd.to_datetime(ann_df['announcement_date'])
    
    # Target Jan 2025 Announcements (Refined actual dates)
    target_start = '2025-01-01'
    target_end = '2025-01-20' # Extended slightly to capture all Jan announcements
    
    ann_df = ann_df[(ann_df['announcement_date'] >= target_start) & (ann_df['announcement_date'] <= target_end)]
    ann_df['is_high'] = ann_df['創新高/低(歷史)'].fillna('').astype(str).str.contains('H')
    
    results = []
    
    print(f"Scanning {len(stock_ids)} stocks for accurate Jan 2025 announcements...")
    
    for stock_id in stock_ids:
        # Get matching announcement for this specific stock
        stock_ann = ann_df[ann_df['stock_id'] == stock_id].sort_values('announcement_date', ascending=False)
        if stock_ann.empty:
            continue
            
        ann_date = stock_ann.iloc[0]['announcement_date']
        is_high = stock_ann.iloc[0]['is_high']
        growth = stock_ann.iloc[0]['revenue_growth_pct']
        
        # Get Price Backbone
        stock_prices = all_prices[all_prices['stock_id'] == stock_id].sort_values('date').copy()
        if stock_prices.empty: continue
        
        stock_prices['stock_ret'] = stock_prices['close'].pct_change()
        
        # Load BPS
        bps_path = f'data/smart_bps_result_{stock_id}.csv'
        bps_df = pd.read_csv(bps_path)
        bps_df['date'] = pd.to_datetime(bps_df['date'])
        
        # Merge
        # IMPORTANT: The BPS output uses 'bps_factor' not 'smart_bps'
        merged = pd.merge(stock_prices, bps_df[['date', 'bps_factor']], on='date', how='left')
        merged = pd.merge(merged, market_df, on='date', how='left')
        merged['alpha'] = (merged['stock_ret'] - merged['mkt_ret']).fillna(0)
        merged['bps_factor'] = merged['bps_factor'].fillna(0)
        
        # Reset Index for position-based lookup
        merged = merged.reset_index(drop=True)
        
        # Find T-0 index (the trading day on or immediately before announcement)
        try:
            t0_idx = merged[merged['date'] <= ann_date].index[-1]
        except IndexError: continue
        
        # Extract T-5 to T-0 (Anticipation / Front-Running Window)
        t5_idx = max(0, t0_idx - 5)
        window_pre = merged.iloc[t5_idx:t0_idx + 1]
        
        # Extract T+1 (Realization Window - the market reaction after announcement)
        # Note: If announcement is after-market, T+1 is the first chance to react.
        t1_idx = t0_idx + 1
        window_post = merged.iloc[t1_idx:t1_idx + 1] if t1_idx < len(merged) else pd.DataFrame()
        
        if len(window_pre) < 5: continue 
        
        # Metrics Calculation
        pre_alpha = (1 + window_pre['alpha']).prod() - 1
        post_alpha = window_post['alpha'].iloc[0] if not window_post.empty else 0
        
        total_bps = window_pre['bps_factor'].sum()
        
        results.append({
            'Stock': stock_id,
            'Growth%': growth,
            'NewHigh': 'YES' if is_high else 'no',
            'Pre-Alpha% (T-5 to T0)': pre_alpha * 100,
            'Post-Alpha% (T+1)': post_alpha * 100,
            'Total Smart BPS': total_bps,
            'AnnDate': ann_date.strftime('%Y-%m-%d')
        })

    # 4. Final Report
    if not results:
        print("No results found for the specified period.")
        return
        
    report_df = pd.DataFrame(results).sort_values('Pre-Alpha% (T-5 to T0)', ascending=False)
    
    print("\n--- Accurate Alpha Report (Jan 2025) ---")
    print(report_df.to_string(index=False))
    
    # Save Report
    report_df.to_csv('data/batch_alpha_report_refined_jan2025.csv', index=False)
    
    # Analysis Summary
    print("\n--- Strategy Performance Summary ---")
    print(f"Overall Avg Pre-Alpha: {report_df['Pre-Alpha% (T-5 to T0)'].mean():.2f}%")
    print(f"Overall Avg Post-Alpha: {report_df['Post-Alpha% (T+1)'].mean():.2f}%")
    
    smart_money = report_df[report_df['Total Smart BPS'] > 0]
    if not smart_money.empty:
        print(f"\n[Smart Money (BPS > 0)] Count: {len(smart_money)}")
        print(f"Avg Pre-Alpha: {smart_money['Pre-Alpha% (T-5 to T0)'].mean():.2f}%")
        print(f"Avg Post-Alpha: {smart_money['Post-Alpha% (T+1)'].mean():.2f}%")

if __name__ == "__main__":
    run_batch_alpha_analysis()
