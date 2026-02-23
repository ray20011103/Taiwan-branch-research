import pandas as pd
import os
from tqdm import tqdm
from smart_bps import run_smart_bps
from bps_strategy import load_data

# Configuration
# Original + New Top 10 Active Stocks
TARGET_STOCKS = [
    '1536', '3645', '3450', '6558', '3706', '4931',
    '3013', '2365', '1815', '8096', '2408', '1514', '6215', '2486', '4510', '6140'
]
REVENUE_DB = 'data/revenue_announcements.parquet'

def calculate_front_run_score(stock_id):
    # Load Data
    rev_df = pd.read_parquet(REVENUE_DB)
    rev_df = rev_df[rev_df['stock_id'] == stock_id].sort_values('announcement_date')
    
    # We now have data up to 2025-06-30
    rev_df = rev_df[rev_df['announcement_date'] < '2025-07-01'] 
    
    if rev_df.empty:
        return None

    # Load Smart BPS
    output_path = f'data/smart_bps_result_{stock_id}.csv'
    if not os.path.exists(output_path):
        run_smart_bps(stock_id)
        
    if not os.path.exists(output_path):
        return None
        
    bps_df = pd.read_csv(output_path)
    
    # 2. Analyze each event
    results = []
    for _, event in rev_df.iterrows():
        ann_date = event['announcement_date']
        growth = event['revenue_growth_pct']
        
        # Window: T-5 to T-1
        bps_window = bps_df[bps_df['date'] < ann_date].tail(5)
        
        if not bps_window.empty:
            pre_ann_buy = bps_window['smart_bps'].sum()
            score = pre_ann_buy * (1 if growth > 0 else -1)
            
            results.append({
                'stock_id': stock_id,
                'ann_date': ann_date,
                'growth_pct': growth,
                'smart_buy_t5': pre_ann_buy,
                'is_front_run': (pre_ann_buy > 10000 and growth > 10) or (pre_ann_buy < -10000 and growth < -10)
            })
            
    return results

def run_market_scan():
    print(f"Scanning {len(TARGET_STOCKS)} stocks for Front-Running signals...")
    
    all_results = []
    # Using tqdm for progress tracking
    for stock in tqdm(TARGET_STOCKS, desc="Scanning Events"):
        res = calculate_front_run_score(stock)
        if res:
            all_results.extend(res)
            
    if all_results:
        df_res = pd.DataFrame(all_results)
        print("\n\n" + "="*50)
        print("DETECTED FRONT-RUNNING EVENTS (2025 H1)")
        print("="*50)
        
        # Sort by absolute Smart Buy volume to see biggest actions
        df_res['abs_buy'] = df_res['smart_buy_t5'].abs()
        df_res = df_res.sort_values('abs_buy', ascending=False)
        
        print(df_res[['stock_id', 'ann_date', 'growth_pct', 'smart_buy_t5', 'is_front_run']].to_string(index=False))
        
        # Summary by Stock
        print("\n--- Summary by Stock ---")
        summary = df_res.groupby('stock_id').agg({
            'is_front_run': 'sum',
            'growth_pct': 'mean'
        }).rename(columns={'is_front_run': 'detected_events'})
        print(summary)

if __name__ == "__main__":
    run_market_scan()
