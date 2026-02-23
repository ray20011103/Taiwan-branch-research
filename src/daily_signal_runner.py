import pandas as pd
import os
import json
from datetime import datetime, timedelta
from bps_strategy import load_data, load_price_data, calculate_bps
from smart_bps import run_smart_bps
from tqdm import tqdm

# Configuration
MOCK_TODAY = '2025-05-09'  # Simulate running this script on this date
TARGET_STOCKS = ['6215', '3706', '3450', '6558', '3013', '6140'] # Subset for demo
PORTFOLIO_FILE = 'data/live_portfolio.json' # Stores current holdings

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, 'r') as f:
            return json.load(f)
    return {} # Format: {'stock_id': {'entry_date': '...', 'entry_price': 100, 'qty': 1}}

def generate_daily_report(today_date):
    print(f"\n{'='*50}")
    print(f"📡 DAILY SIGNAL REPORT (Simulated Date: {today_date})")
    print(f"{'='*50}\n")
    
    portfolio = load_portfolio()
    action_plan = []
    
    # 1. SCAN FOR NEW ENTRIES
    print(">>> Scanning for New Entry Signals...")
    
    for stock_id in tqdm(TARGET_STOCKS, desc="Scanning"):
        # In a real scenario, we would load data up to today_date
        # Here we assume the parquet files already contain data for this date
        
        # We perform a quick check using the cached Smart BPS result to save time
        # In production, you would run_smart_bps(stock_id) daily
        bps_path = f'data/smart_bps_result_{stock_id}.csv'
        if not os.path.exists(bps_path):
             # Try to generate if missing (mocking the daily update)
             continue
        
        df = pd.read_csv(bps_path)
        
        # Get data for 'today' and previous days
        row_today = df[df['date'] == today_date]
        if row_today.empty:
            continue
            
        today_idx = row_today.index[0]
        if today_idx < 5: continue
        
        # Calculate trailing 5-day Smart BPS
        window = df.iloc[today_idx-4 : today_idx+1] # Include today
        cum_buy = window['smart_bps'].sum()
        price_today = row_today.iloc[0]['price']
        
        # Signal Logic: Strong Buy (> 500張 & Price Trend Up)
        # Simplified for demo
        if cum_buy > 500000: # 500張
            action_plan.append({
                'action': 'BUY',
                'stock_id': stock_id,
                'price': price_today,
                'reason': f'Smart BPS Accumulation ({int(cum_buy/1000)}K shares)'
            })

    # 2. MONITOR EXISTING POSITIONS
    print("\n>>> Monitoring Portfolio Positions...")
    # Mocking a position in 4931 (which had a stop loss event around this time)
    # or 3645
    mock_portfolio = {
        '4931': {'entry_date': '2025-04-15', 'entry_price': 75.3, 'qty': 1000},
        '6215': {'entry_date': '2025-05-02', 'entry_price': 101.0, 'qty': 1000} 
    }
    
    for stock_id, pos in mock_portfolio.items():
        # Get current price
        price_data = load_price_data(stock_id)
        if price_data.empty: continue
        
        row = price_data[price_data['date'] == today_date]
        if row.empty: continue
        current_price = row.iloc[0]['close']
        
        # Check Stop Loss (-7%)
        pnl_pct = (current_price - pos['entry_price']) / pos['entry_price']
        
        status = f"Unrealized PnL: {pnl_pct*100:.2f}%"
        
        if pnl_pct <= -0.07:
            action_plan.append({
                'action': 'SELL (STOP LOSS)',
                'stock_id': stock_id,
                'price': current_price,
                'reason': f'Hit Stop Loss (-7%). Entry: {pos["entry_price"]}'
            })
        elif pnl_pct > 0.2: # Take Profit Target
             action_plan.append({
                'action': 'SELL (TAKE PROFIT)',
                'stock_id': stock_id,
                'price': current_price,
                'reason': f'Hit Profit Target (>20%)'
            })
        else:
            print(f"  - {stock_id}: Hold. {status}")

    # 3. PRINT FINAL ACTION PLAN
    print("\n" + "="*50)
    print("📋 ACTION PLAN FOR TOMORROW")
    print("="*50)
    
    if not action_plan:
        print("No actions required.")
    else:
        for item in action_plan:
            icon = "🟢" if item['action'] == 'BUY' else "🔴"
            print(f"{icon} {item['action']} {item['stock_id']} @ {item['price']}")
            print(f"   Reason: {item['reason']}")
            print("-" * 30)

if __name__ == "__main__":
    generate_daily_report(MOCK_TODAY)
