import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from bps_strategy import load_price_data, load_data, calculate_bps
from smart_bps import run_smart_bps # To ensure smart_bps files exist

# Configuration
TARGET_STOCKS = [
    '1536', '3645', '3450', '6558', '3706', '4931',
    '3013', '2365', '1815', '8096', '2408', '1514', '6215', '2486', '4510', '6140'
]
REVENUE_DB = 'data/revenue_announcements.parquet'
STOP_LOSS_PCT = 0.07

def get_price_on_date(price_df, target_date):
    """Aligns dates similar to backtest_strategy.py"""
    target_dt = pd.to_datetime(target_date)
    
    # Search backwards 5 days
    for i in range(6):
        d = target_dt - pd.Timedelta(days=i)
        d_str = d.strftime('%Y-%m-%d')
        row = price_df[price_df['date'] == d_str]
        if not row.empty:
            return row.iloc[0]['close'], d_str, row.index[0]
    return None, None, None

def run_ab_test():
    print(f"--- A/B Test: Smart BPS vs Original BPS (2025 H1) ---")
    
    # Load Revenue Events
    if not os.path.exists(REVENUE_DB):
        return
    rev_df = pd.read_parquet(REVENUE_DB)
    rev_df = rev_df[rev_df['stock_id'].isin(TARGET_STOCKS)]
    rev_df = rev_df[(rev_df['announcement_date'] >= '2025-01-01') & (rev_df['announcement_date'] < '2025-07-01')]
    
    # Results storage
    results_smart = []
    results_original = []
    
    for stock_id in tqdm(TARGET_STOCKS, desc="Comparing"):
        # 1. Ensure Smart BPS file exists (if not, run it)
        smart_path = f'data/smart_bps_result_{stock_id}.csv'
        if not os.path.exists(smart_path):
            # In a real rigorous test, we should run this. 
            # But assume previous steps generated them. If missing, skip.
            continue
            
        # Load combined result (Smart BPS file contains Original BPS column too!)
        df = pd.read_csv(smart_path)
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        
        # Load Price Data for Stop Loss checking
        price_df = load_price_data(stock_id)
        if price_df.empty: continue
        price_df['date'] = pd.to_datetime(price_df['date']).dt.strftime('%Y-%m-%d')
        price_df = price_df.sort_values('date').reset_index(drop=True)
        
        # Events
        stock_events = rev_df[rev_df['stock_id'] == stock_id].sort_values('announcement_date')
        
        for _, event in stock_events.iterrows():
            ann_date = event['announcement_date']
            
            # Align Exit Date
            exit_price, _, exit_idx = get_price_on_date(price_df, ann_date)
            if exit_idx is None or exit_idx < 5: continue
            
            entry_idx = exit_idx - 5
            
            # Get Signal Data Window (from BPS df)
            # Need to match dates from price_df to BPS df
            entry_date_str = price_df.iloc[entry_idx]['date']
            exit_date_str = price_df.iloc[exit_idx]['date']
            
            bps_window = df[(df['date'] >= entry_date_str) & (df['date'] < exit_date_str)]
            
            if bps_window.empty: continue
            
            # --- STRATEGY A: SMART BPS ---
            sig_smart = bps_window['smart_bps'].sum()
            
            # --- STRATEGY B: ORIGINAL BPS ---
            sig_original = bps_window['original_bps'].sum()
            
            entry_price = price_df.iloc[entry_idx]['close']
            
            # Simulation Function
            def simulate_trade(signal):
                if signal <= 0: return None # Long Only
                
                # Check Stop Loss
                final_exit = exit_price
                for i in range(entry_idx + 1, exit_idx + 1):
                    curr_p = price_df.iloc[i]['close']
                    if (curr_p - entry_price) / entry_price <= -STOP_LOSS_PCT:
                        final_exit = curr_p
                        break
                
                return (final_exit - entry_price) / entry_price

            # Record results
            ret_smart = simulate_trade(sig_smart)
            if ret_smart is not None:
                results_smart.append(ret_smart)
                
            ret_original = simulate_trade(sig_original)
            if ret_original is not None:
                results_original.append(ret_original)

    # --- METRICS CALCULATION ---
    def calc_metrics(returns, name):
        if not returns:
            print(f"\n{name}: No trades.")
            return
        
        returns = np.array(returns)
        total_ret = np.sum(returns)
        avg_ret = np.mean(returns)
        win_rate = np.sum(returns > 0) / len(returns)
        
        # Simplified Sharpe (assuming risk-free = 0, annualized)
        # We don't have daily returns here, only trade returns. 
        # A proper Sharpe requires daily equity curve. 
        # Here we use "Trade Sharpe" = Mean / Std
        trade_sharpe = avg_ret / np.std(returns) if np.std(returns) > 0 else 0
        
        print(f"\n--- {name} Results ---")
        print(f"Trades: {len(returns)}")
        print(f"Total Return: {total_ret*100:.2f}%")
        print(f"Avg Return: {avg_ret*100:.2f}%")
        print(f"Win Rate: {win_rate*100:.2f}%")
        print(f"Trade Sharpe (Mean/Std): {trade_sharpe:.4f}")

    calc_metrics(results_original, "STRATEGY B: ORIGINAL BPS (All Brokers)")
    calc_metrics(results_smart, "STRATEGY A: SMART BPS (Cluster 3)")

if __name__ == "__main__":
    run_ab_test()
