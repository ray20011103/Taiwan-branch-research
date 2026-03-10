import pandas as pd
import os

def find_leveraged_smart_money(stock_id='2317'):
    print(f"--- Searching for Leveraged Smart Money in {stock_id} ---")
    
    # 1. Load Stock Transactions
    stock_path = 'data/StockBranch.parquet'
    df_stock = pd.read_parquet(stock_path, filters=[('CommodityId', '==', str(stock_id))])
    df_stock = df_stock.rename(columns={'Date': 'date', 'SecuritiesTraderId': 'broker_id', 'Buy': 'stock_buy', 'Sell': 'stock_sell'})
    df_stock['date'] = pd.to_datetime(df_stock['date'])
    df_stock['net_stock_buy'] = df_stock['stock_buy'] - df_stock['stock_sell']
    
    stock_agg = df_stock.groupby(['date', 'broker_id'])['net_stock_buy'].sum().reset_index()
    print(f"Total Stock Transactions for {stock_id}: {len(stock_agg)}")

    # 2. Load Warrant Transactions & Specs
    warrant_trades_path = 'data/分點進出.parquet'
    warrant_specs_path = 'data/權證條件.parquet'
    
    df_w_trades = pd.read_parquet(warrant_trades_path)
    df_w_specs = pd.read_parquet(warrant_specs_path)
    
    df_w_specs['標的代號'] = df_w_specs['標的代號'].astype(str)
    this_stock_warrants = df_w_specs[df_w_specs['標的代號'] == str(stock_id)]['權證代號'].unique()
    print(f"Found {len(this_stock_warrants)} warrants for {stock_id}")
    
    df_w_trades_filtered = df_w_trades[df_w_trades['權證代號'].isin(this_stock_warrants)].copy()
    df_w_trades_filtered['date'] = pd.to_datetime(df_w_trades_filtered['日期'])
    df_w_trades_filtered = df_w_trades_filtered.rename(columns={'分點代號': 'broker_id'})
    print(f"Total Warrant Trades for {stock_id}: {len(df_w_trades_filtered)}")
    
    # Merge with specs
    df_w_merged = pd.merge(
        df_w_trades_filtered,
        df_w_specs[['日期', '權證代號', '最新執行比例', 'IVDelta值']],
        left_on=['date', '權證代號'],
        right_on=['日期', '權證代號'],
        how='inner'
    )
    print(f"Warrant Trades with Delta Specs: {len(df_w_merged)}")
    
    df_w_merged['warrant_delta_buy'] = (
        (df_w_merged['買張'] - df_w_merged['賣張']) * 
        df_w_merged['最新執行比例'] * 
        df_w_merged['IVDelta值']
    )
    
    warrant_agg = df_w_merged.groupby(['date', 'broker_id'])['warrant_delta_buy'].sum().reset_index()
    print(f"Unique Broker-Date Warrant Pairs: {len(warrant_agg)}")

    # 3. Combine
    stock_agg['net_stock_buy_lots'] = stock_agg['net_stock_buy'] / 1000.0
    combined = pd.merge(stock_agg, warrant_agg, on=['date', 'broker_id'], how='inner')
    print(f"Overlapping Broker-Date Pairs: {len(combined)}")
    
    if not combined.empty:
        combined['total_pressure_lots'] = combined['net_stock_buy_lots'] + combined['warrant_delta_buy']
        
        # Filter for "Significant Double Buying"
        # At least 50 lots of combined pressure
        double_buy = combined[(combined['net_stock_buy_lots'] > 20) & (combined['warrant_delta_buy'] > 20)]
        
        print(f"\nFound {len(double_buy)} cases of Significant Leveraged Double-Buying (Stock > 20 Lots & Warrant Delta > 20 Lots)!")
        if not double_buy.empty:
            print(double_buy.sort_values('total_pressure_lots', ascending=False).head(10))

if __name__ == "__main__":
    find_leveraged_smart_money('2317')
