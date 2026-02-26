import pandas as pd
import os
import glob
from datetime import datetime

def run_batch_alpha_analysis():
    print("--- 🚀 Refined Batch Alpha Analysis (Target: Jan 2025 Announcements) ---")
    
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
    
    # 3. Load Target Month Announcements (Jan 2025)
    ann_df = pd.read_csv('data/announcement.csv')
    ann_df['營收發布日'] = pd.to_datetime(ann_df['營收發布日'])
    
    # Target Jan 2025 Announcements (usually released around Jan 1-15)
    target_start = '2025-01-01'
    target_end = '2025-01-15'
    
    ann_df = ann_df[(ann_df['營收發布日'] >= target_start) & (ann_df['營收發布日'] <= target_end)]
    ann_df['stock_id'] = ann_df['公司'].str.split(' ').str[0]
    ann_df['is_high'] = ann_df['創新高/低(歷史)'].fillna('').astype(str).str.contains('H')
    
    results = []
    
    print(f"Scanning {len(stock_ids)} stocks for matching Jan 2025 announcements...")
    
    for stock_id in stock_ids:
        # Get matching announcement
        stock_ann = ann_df[ann_df['stock_id'] == stock_id].sort_values('營收發布日', ascending=False)
        if stock_ann.empty:
            continue
            
        ann_date = stock_ann.iloc[0]['營收發布日']
        is_high = stock_ann.iloc[0]['is_high']
        
        # Get Price Backbone
        stock_prices = all_prices[all_prices['stock_id'] == stock_id].sort_values('date').copy()
        if stock_prices.empty: continue
        
        stock_prices['stock_ret'] = stock_prices['close'].pct_change()
        
        # Load BPS
        bps_path = f'data/smart_bps_result_{stock_id}.csv'
        bps_df = pd.read_csv(bps_path)
        bps_df['date'] = pd.to_datetime(bps_df['date'])
        
        # Merge
        merged = pd.merge(stock_prices, bps_df[['date', 'smart_bps']], on='date', how='left')
        merged = pd.merge(merged, market_df, on='date', how='left')
        merged['alpha'] = (merged['stock_ret'] - merged['mkt_ret']).fillna(0)
        merged['smart_bps'] = merged['smart_bps'].fillna(0)
        
        # Find T-0 index (closest to or on announcement date)
        try:
            t0_idx = merged[merged['date'] <= ann_date].index[-1]
        except IndexError: continue
        
        # Extract T-5 to T-0
        t5_idx = max(0, t0_idx - 5)
        window = merged.iloc[t5_idx:t0_idx + 1]
        
        if len(window) < 5: continue # Skip if data is too sparse
        
        # Metrics
        cum_alpha = (1 + window['alpha']).prod() - 1
        # BPS Signal Intensity (Sum of buy signals in the T-5 period)
        total_bps = window['smart_bps'].sum()
        max_daily_alpha = window['alpha'].max()
        
        results.append({
            'Stock': stock_id,
            'Name': stock_ann.iloc[0]['公司'].split(' ')[1],
            'NewHigh': 'YES' if is_high else 'no',
            'T-5 to T-0 Alpha%': cum_alpha * 100,
            'Total Smart BPS': total_bps,
            'Max Daily Alpha%': max_daily_alpha * 100
        })

    # 4. Final Report
    report_df = pd.DataFrame(results).sort_values('T-5 to T-0 Alpha%', ascending=False)
    
    print("\n--- 📊 Refined Alpha Report (Jan 2025) ---")
    print(report_df.to_string(index=False))
    
    # Save Report
    report_df.to_csv('data/batch_alpha_report_jan2025.csv', index=False)
    
    # Analysis Summary
    print("\n--- Strategy Performance Summary ---")
    print(f"1. Overall Avg Alpha: {report_df['T-5 to T-0 Alpha%'].mean():.2f}%")
    
    high_revenue = report_df[report_df['NewHigh'] == 'YES']
    if not high_revenue.empty:
        print(f"2. Revenue New High Group Avg Alpha: {high_revenue['T-5 to T-0 Alpha%'].mean():.2f}% ({len(high_revenue)} stocks)")
        
    smart_money = report_df[report_df['Total Smart BPS'] > 0]
    if not smart_money.empty:
        print(f"3. Smart Money (BPS > 0) Group Avg Alpha: {smart_money['T-5 to T-0 Alpha%'].mean():.2f}% ({len(smart_money)} stocks)")
        
    combo = report_df[(report_df['NewHigh'] == 'YES') & (report_df['Total Smart BPS'] > 0)]
    if not combo.empty:
        print(f"4. COMBO (New High + Smart Money) Avg Alpha: {combo['T-5 to T-0 Alpha%'].mean():.2f}% ({len(combo)} stocks)")

if __name__ == "__main__":
    run_batch_alpha_analysis()
