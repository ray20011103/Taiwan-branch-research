import pandas as pd
import os

# Target Trades
TRADES = [
    # {'stock_id': '1503', 'ann_date': '2024-04-10'}, # 士電
    {'stock_id': '6215', 'ann_date': '2025-01-09'},  # 和椿

    ]

def analyze_trade_details():
    print("--- Deep Dive: Trade Forensics (Alpha Calculation) ---")
    
    # Load Price History for Backbone
    if not os.path.exists('data/stock_price_history.parquet'):
        print("Price history not found.")
        return
    
    all_prices = pd.read_parquet('data/stock_price_history.parquet')
    all_prices['date'] = pd.to_datetime(all_prices['date'])
    
    # Load Market Index for Alpha calculation
    if os.path.exists('data/market_index.parquet'):
        market_df = pd.read_parquet('data/market_index.parquet')
        market_df['date'] = pd.to_datetime(market_df['date'])
        # We only need the market return
        market_df = market_df[['date', 'market_ret']].rename(columns={'market_ret': 'mkt_ret'})
    else:
        print("Market index not found. Alpha calculation will be skipped.")
        market_df = None
    
    for trade in TRADES:
        stock_id = trade['stock_id']
        ann_date = trade['ann_date']
        
        print(f"\nAnalyzing {stock_id} around {ann_date}...")
        
        # 1. Get Price Backbone for this stock
        stock_prices = all_prices[all_prices['stock_id'] == stock_id].sort_values('date').copy()
        
        # Calculate daily returns
        stock_prices['stock_ret'] = stock_prices['close'].pct_change()
        
        # 2. Load Smart BPS Data
        bps_path = f'data/smart_bps_result_{stock_id}.csv'
        if not os.path.exists(bps_path):
            print(f"Data file not found: {bps_path}")
            continue
            
        bps_df = pd.read_csv(bps_path)
        bps_df['date'] = pd.to_datetime(bps_df['date'])
        
        # 3. Merge (Left Join on Price Data)
        merged_df = pd.merge(stock_prices, bps_df[['date', 'smart_bps']], on='date', how='left')
        
        # Merge with Market Data
        if market_df is not None:
            merged_df = pd.merge(merged_df, market_df, on='date', how='left')
            merged_df['alpha'] = merged_df['stock_ret'] - merged_df['mkt_ret']
            merged_df['cum_alpha'] = 0.0 # Placeholder
        
        # Fill NaN BPS with 0 (No Signal)
        merged_df['smart_bps'] = merged_df['smart_bps'].fillna(0)
        
        # 4. Find the target window
        target_dt = pd.to_datetime(ann_date)
        
        try:
            idx = merged_df[merged_df['date'] == target_dt].index[0]
        except IndexError:
            idx = merged_df[merged_df['date'] < target_dt].index[-1]
            print(f"(Note: Announcement {ann_date} was non-trading, using {merged_df.iloc[idx]['date'].date()})")

        # Calculate Cumulative Alpha starting from T-10
        start_calc_idx = max(0, idx - 10)
        merged_df.loc[start_calc_idx:idx+5, 'cum_alpha'] = merged_df.loc[start_calc_idx:idx+5, 'alpha'].cumsum()

        # Extract T-15 to T+5 window
        start_idx = max(0, idx - 15)
        end_idx = min(len(merged_df), idx + 5)
        
        window = merged_df.iloc[start_idx:end_idx].copy()
        
        # Display
        print(f"{'Date':<12} {'Price':<8} {'Ret%':<8} {'Mkt%':<8} {'Alpha%':<8} {'BPS':<10} {'Note'}")
        print("-" * 75)
        
        entry_idx = idx - 5
        
        for i, row in window.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            price = row['close']
            ret = row['stock_ret'] * 100 if not pd.isna(row['stock_ret']) else 0
            mkt = row['mkt_ret'] * 100 if not pd.isna(row['mkt_ret']) else 0
            alpha = row['alpha'] * 100 if not pd.isna(row['alpha']) else 0
            sbps = row['smart_bps']
            
            note = ""
            if i == entry_idx: note = "<-- T-5 (Entry)"
            if i == idx: note = "<-- Announce (Exit)"
            
            bps_display = f"{sbps:,.0f}" if sbps != 0 else "0"
            
            print(f"{date_str:<12} {price:<8.2f} {ret:<8.2f} {mkt:<8.2f} {alpha:<8.2f} {bps_display:<10} {note}")

if __name__ == "__main__":
    analyze_trade_details()
