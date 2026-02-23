import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from bps_strategy import load_price_data 

# Configuration
# Top 50 Active Stocks
TARGET_STOCKS = [
    '3013', '2365', '3450', '6558', '1815', '8096', '2408', '4931', '1514', '6215', 
    '2486', '4510', '6140', '3047', '3312', '4909', '2615', '4979', '2359', '8054', 
    '3363', '4991', '3706', '3163', '8028', '2609', '6117', '1503', '2374', '4303', 
    '2543', '8064', '1540', '6148', '5426', '8111', '2363', '5443', '4562', '2464', 
    '2312', '3379', '5251', '3535', '1519', '3062', '6442', '6462', '2468', '3376'
]
REVENUE_DB = 'data/revenue_announcements.parquet'
MARKET_INDEX_DB = 'data/market_index.parquet'
STOP_LOSS_PCT = 0.07  # 7% Stop Loss
VALUE_THRESHOLD = 10_000_000 # Entry Signal: Smart Buy Value > 10 Million NTD

def get_price_on_date(bps_df, target_date):
    """
    Tries to find the price on target_date. 
    If not found, searches backwards for up to 5 days.
    Returns (price, valid_date, index) or (None, None, None).
    """
    target_date_str = str(target_date)
    
    # Try exact match first
    row = bps_df[bps_df['date'] == target_date_str]
    if not row.empty:
        return row.iloc[0]['price'], target_date_str, row.index[0]
        
    # Search backwards
    bps_dates = pd.to_datetime(bps_df['date'])
    target_dt = pd.to_datetime(target_date)
    
    for i in range(1, 6):
        prev_date = target_dt - pd.Timedelta(days=i)
        prev_date_str = prev_date.strftime('%Y-%m-%d')
        row = bps_df[bps_df['date'] == prev_date_str]
        if not row.empty:
            return row.iloc[0]['price'], prev_date_str, row.index[0]
            
    return None, None, None

def run_backtest():
    print("Starting Backtest: Smart BPS Revenue Ambush Strategy (2024-2025 Full Market).")
    print(f"Target: Top {len(TARGET_STOCKS)} Active Stocks.")
    print(f"Strategy: Long Only. Enter at T-5.")
    print(f"Filter: Smart Buy Value > {VALUE_THRESHOLD/1_000_000}M NTD.")
    print(f"Exit: Announcement Day. Stop Loss: {STOP_LOSS_PCT*100}%.\n")
    
    # Load Revenue Events
    if not os.path.exists(REVENUE_DB):
        print("Revenue DB not found.")
        return

    rev_df = pd.read_parquet(REVENUE_DB)
    rev_df = rev_df[rev_df['stock_id'].isin(TARGET_STOCKS)]
    # Extend period to 2024-2025
    rev_df = rev_df[(rev_df['announcement_date'] >= '2024-01-01') & (rev_df['announcement_date'] < '2025-07-01')]
    
    # [Load Real Market Index]
    print("Loading Real Market Index (TAIEX)...")
    if not os.path.exists(MARKET_INDEX_DB):
        print("Market Index DB not found. Alpha will be 0.")
        market_returns = {}
    else:
        market_df = pd.read_parquet(MARKET_INDEX_DB)
        # Create a dictionary for fast lookup: date_str -> daily_return
        # Ensure date format matches bps_df (YYYY-MM-DD string)
        market_df['date_str'] = market_df['date'].dt.strftime('%Y-%m-%d')
        market_returns = pd.Series(market_df.market_ret.values, index=market_df.date_str).to_dict()
        print(f"Market data loaded: {len(market_returns)} days.")
    
    trades = []
    
    for stock_id in tqdm(TARGET_STOCKS, desc="Backtesting Stocks"):
        bps_path = f'data/smart_bps_result_{stock_id}.csv'
        if not os.path.exists(bps_path):
            continue
            
        bps_df = pd.read_csv(bps_path)
        bps_df['date'] = bps_df['date'].astype(str)
        bps_df = bps_df.sort_values('date')
        
        stock_events = rev_df[rev_df['stock_id'] == stock_id].sort_values('announcement_date')
        
        for _, event in stock_events.iterrows():
            ann_date = event['announcement_date']
            
            # 1. Align Dates (Find Exit Date Index)
            exit_price, valid_exit_date, exit_idx = get_price_on_date(bps_df, ann_date)
            
            if exit_idx is None: continue
            if exit_idx < 5: continue
                
            # Define Window Indices
            entry_idx = exit_idx - 5
            window_indices = range(entry_idx, exit_idx) 
            window_df = bps_df.iloc[window_indices]
            
            # 2. Calculate Signal
            cumulative_smart_buy = window_df['smart_bps'].sum()
            if pd.isna(cumulative_smart_buy): cumulative_smart_buy = 0
            
            # 3. Execution Price
            entry_price = bps_df.iloc[entry_idx]['price']
            if entry_price == 0: continue

            # --- NEW FILTER: VALUE THRESHOLD ---
            smart_buy_value = cumulative_smart_buy * entry_price
            
            if smart_buy_value <= VALUE_THRESHOLD:
                continue # Skip trade, signal too weak
                
            # 4. Check for Stop Loss
            final_exit_price = exit_price
            exit_reason = 'Event'
            actual_exit_idx = exit_idx
            
            for i in range(entry_idx + 1, exit_idx + 1):
                current_price = bps_df.iloc[i]['price']
                current_date = bps_df.iloc[i]['date']
                
                pnl_pct = (current_price - entry_price) / entry_price
                
                if pnl_pct <= -STOP_LOSS_PCT:
                    final_exit_price = current_price
                    exit_reason = f'Stop Loss ({current_date})'
                    actual_exit_idx = i
                    break
            
            return_pct = (final_exit_price - entry_price) / entry_price
            
            # --- CALCULATE ALPHA (Geometric Compounding) ---
            # Holding Period: From Entry Date (Close) to Exit Date (Close)
            # Market return applies from Entry Date + 1 to Exit Date
            
            entry_date_str = bps_df.iloc[entry_idx]['date']
            exit_date_str = bps_df.iloc[actual_exit_idx]['date']
            
            entry_date_obj = pd.to_datetime(entry_date_str)
            exit_date_obj = pd.to_datetime(exit_date_str)
            
            market_compound_ret = 1.0
            curr_d = entry_date_obj + pd.Timedelta(days=1)
            
            while curr_d <= exit_date_obj:
                d_str = curr_d.strftime('%Y-%m-%d')
                if d_str in market_returns:
                    market_compound_ret *= (1 + market_returns[d_str])
                curr_d += pd.Timedelta(days=1)
            
            market_ret_period = market_compound_ret - 1.0
            alpha = return_pct - market_ret_period
            
            trades.append({
                'stock_id': stock_id,
                'ann_date': ann_date,
                'growth_pct': event['revenue_growth_pct'],
                'signal_qty': cumulative_smart_buy,
                'value_mn': smart_buy_value / 1_000_000,
                'entry_price': entry_price,
                'exit_price': final_exit_price,
                'exit_reason': exit_reason,
                'return_pct': return_pct * 100,
                'market_ret': market_ret_period * 100,
                'alpha': alpha * 100
            })

    # Analyze Results
    if not trades:
        print("No trades generated with current filter.")
        return

    trade_df = pd.DataFrame(trades)
    
    # Save full report to CSV
    report_path = 'data/full_trade_report.csv'
    trade_df.to_csv(report_path, index=False)
    
    print("\n" + "="*60)
    print(f"BACKTEST RESULTS (2024-2025 Full Market) - Alpha Analysis")
    print("="*60)
    
    print(f"Total Trades: {len(trade_df)}")
    wins = len(trade_df[trade_df['return_pct'] > 0])
    losses = len(trade_df[trade_df['return_pct'] <= 0])
    print(f"Win Rate: {wins / len(trade_df) * 100:.2f}% ({wins} W / {losses} L)")
    print(f"Avg Return: {trade_df['return_pct'].mean():.2f}%")
    print(f"Avg Market Return (Beta): {trade_df['market_ret'].mean():.2f}%")
    print(f"Avg Alpha (Excess Return): {trade_df['alpha'].mean():.2f}%")
    print(f"Total Alpha Generated: {trade_df['alpha'].sum():.2f}%")
    
    # Calculate Alpha Win Rate (How often do we beat the market?)
    alpha_wins = len(trade_df[trade_df['alpha'] > 0])
    print(f"Alpha Win Rate (Beat Market): {alpha_wins / len(trade_df) * 100:.2f}%")
    
    # Sharpe Ratio (Trade-based)
    returns = trade_df['return_pct']
    sharpe = returns.mean() / returns.std() if len(returns) > 1 else 0
    print(f"Trade Sharpe Ratio: {sharpe:.4f}")
    
    print(f"\n[OK] Full trade report saved to: {report_path}")
    print("\n--- Top 10 High Alpha Trades ---")
    print(trade_df.sort_values('alpha', ascending=False).head(10)[['stock_id', 'ann_date', 'return_pct', 'market_ret', 'alpha']].to_string(index=False))

if __name__ == "__main__":
    run_backtest()