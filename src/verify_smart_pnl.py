import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# Configuration
DATA_DIR = 'data/'
CORE_BROKERS = ['8960', '8890', '1360']
TARGET_STOCKS = ['1536', '3645', '3450', '6558', '3706', '4931', '3013', '2365', '1815', '8096', '2408', '1514', '6215', '2486', '4510', '6140']

def verify_broker_performance():
    print(f"Verifying real-world performance for Core Smart Brokers: {CORE_BROKERS}")
    
    # 1. Load Price and Announcement Data
    all_prices = pd.read_parquet('data/stock_price_history.parquet')
    all_prices['date'] = pd.to_datetime(all_prices['date'])
    
    ann_df = pd.read_csv('data/announcement.csv')
    ann_df['ann_date'] = pd.to_datetime(ann_df['營收發布日'])
    ann_df['stock_id'] = ann_df['公司'].str.split(' ').str[0]
    
    # 2. Load Transaction Data for Core Brokers
    file_path = os.path.join(DATA_DIR, 'StockBranch.parquet')
    results = []
    
    for stock_id in tqdm(TARGET_STOCKS, desc="Calculating PnL across stocks"):
        try:
            # Load trades for core brokers in this stock
            df = pd.read_parquet(file_path, filters=[
                ('CommodityId', '==', str(stock_id)),
                ('SecuritiesTraderId', 'in', CORE_BROKERS)
            ])
            if df.empty: continue
            
            df = df.rename(columns={'Date': 'date', 'CommodityId': 'stock_id', 'SecuritiesTraderId': 'broker_id', 'Buy': 'buy', 'Sell': 'sell', 'Price': 'price'})
            df['date'] = pd.to_datetime(df['date'])
            
            # Get Price for this stock
            prices = all_prices[all_prices['stock_id'] == stock_id].sort_values('date')
            
            # For each broker in this stock, find their first entry in the Jan 2025 window
            for broker in CORE_BROKERS:
                broker_trades = df[df['broker_id'] == broker].sort_values('date')
                if broker_trades.empty: continue
                
                # Focus on the Jan 2025 Revenue Event (T-15 to T-2 entry)
                match_ann = ann_df[(ann_df['stock_id'] == stock_id) & (ann_df['ann_date'] >= '2025-01-01') & (ann_df['ann_date'] <= '2025-01-20')]
                if match_ann.empty: continue
                
                ann_date = match_ann.iloc[0]['ann_date']
                
                # Find entry window (T-15 to T-2)
                entry_trades = broker_trades[(broker_trades['date'] >= ann_date - pd.Timedelta(days=15)) & (broker_trades['date'] <= ann_date - pd.Timedelta(days=2))]
                if entry_trades.empty: continue
                
                avg_entry_price = (entry_trades['buy'] * entry_trades['price']).sum() / entry_trades['buy'].sum()
                entry_date = entry_trades.iloc[0]['date']
                
                # Max Run-up (Highest price within 20 days after entry)
                future_prices = prices[prices['date'] >= entry_date].head(20)
                if future_prices.empty: continue
                
                max_price = future_prices['high'].max()
                max_ret = (max_price / avg_entry_price - 1)
                
                results.append({
                    'Broker': broker,
                    'Stock': stock_id,
                    'EntryDate': entry_date,
                    'AnnDate': ann_date,
                    'MaxRunUp%': max_ret * 100
                })
        except Exception as e:
            continue
            
    perf_df = pd.DataFrame(results)
    
    # 3. Aggregate Performance Stats
    summary = perf_df.groupby('Broker').agg({
        'MaxRunUp%': ['count', 'mean', 'max']
    }).reset_index()
    summary.columns = ['Broker', 'Trades', 'AvgMaxRunUp%', 'SingleBest%']
    
    print("\n--- Core Smart Broker Performance (Jan 2025 Window) ---")
    print(summary.to_string(index=False))
    
    # Save for report
    summary.to_csv('data/core_broker_performance.csv', index=False)
    perf_df.to_csv('data/core_broker_trade_details.csv', index=False)

if __name__ == "__main__":
    verify_broker_performance()
