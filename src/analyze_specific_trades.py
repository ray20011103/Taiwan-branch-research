import pandas as pd
import os

# Target Trades
TRADES = [
    # {'stock_id': '1503', 'ann_date': '2024-04-10'}, # 士電
    # {'stock_id': '6215', 'ann_date': '2025-01-09'},  # 
    {'stock_id': '8111', 'ann_date': '2025-01-09'}
    ]

def analyze_trade_details():
    print("--- Deep Dive: Trade Forensics (Full Calendar) ---")
    
    # Load Price History for Backbone
    if not os.path.exists('data/stock_price_history.parquet'):
        print("Price history not found.")
        return
    
    all_prices = pd.read_parquet('data/stock_price_history.parquet')
    all_prices['date'] = pd.to_datetime(all_prices['date'])
    
    for trade in TRADES:
        stock_id = trade['stock_id']
        ann_date = trade['ann_date']
        
        print(f"\nAnalyzing {stock_id} around {ann_date}...")
        
        # 1. Get Price Backbone for this stock
        stock_prices = all_prices[all_prices['stock_id'] == stock_id].sort_values('date').copy()
        
        # 2. Load Smart BPS Data
        bps_path = f'data/smart_bps_result_{stock_id}.csv'
        if not os.path.exists(bps_path):
            print(f"Data file not found: {bps_path}")
            continue
            
        bps_df = pd.read_csv(bps_path)
        bps_df['date'] = pd.to_datetime(bps_df['date'])
        
        # 3. Merge (Left Join on Price Data) to reveal missing BPS days
        merged_df = pd.merge(stock_prices, bps_df[['date', 'smart_bps']], on='date', how='left')
        
        # Fill NaN BPS with 0 (No Signal)
        merged_df['smart_bps'] = merged_df['smart_bps'].fillna(0)
        
        # 4. Find the target window
        target_dt = pd.to_datetime(ann_date)
        
        # Find index of target date in the continuous price series
        try:
            idx = merged_df[merged_df['date'] == target_dt].index[0]
        except IndexError:
            # If announcement date is not a trading day, find the previous trading day
            idx = merged_df[merged_df['date'] < target_dt].index[-1]
            print(f"(Note: Announcement {ann_date} was non-trading, using {merged_df.iloc[idx]['date'].date()})")

        # Extract T-15 to T+5 window
        start_idx = max(0, idx - 15)
        end_idx = min(len(merged_df), idx + 5)
        
        window = merged_df.iloc[start_idx:end_idx].copy()
        
        # Display
        print(f"{'Date':<12} {'Price':<10} {'Smart BPS':<15} {'Note':<20}")
        print("-" * 65)
        
        entry_idx = idx - 5
        
        for i, row in window.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            price = row['close']
            sbps = row['smart_bps']
            
            note = ""
            if i == entry_idx: note = "<-- T-5 (Entry)"
            if i == idx: note = "<-- Announce (Exit)"
            
            # Highlight missing data days
            bps_display = f"{sbps:,.0f}"
            if sbps == 0:
                bps_display = "0 (No Signal)"
            
            print(f"{date_str:<12} {price:<10.2f} {bps_display:<15} {note}")

if __name__ == "__main__":
    analyze_trade_details()
