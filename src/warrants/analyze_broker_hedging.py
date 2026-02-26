import pandas as pd
import os

def analyze_warrant_hedging_impact():
    print("--- 🛡️ Warrant Hedging & Market Impact Analysis (Full Period 2025) ---")
    
    # 1. Load Data
    warrant_trades_path = 'data/分點進出.parquet'
    warrant_specs_path = 'data/權證條件.parquet'
    stock_history_path = 'data/stock_price_history.parquet'
    
    if not all(os.path.exists(p) for p in [warrant_trades_path, warrant_specs_path, stock_history_path]):
        print("Required data files not found.")
        return

    # Load Data
    df_trades = pd.read_parquet(warrant_trades_path)
    df_specs = pd.read_parquet(warrant_specs_path)
    df_prices = pd.read_parquet(stock_history_path)
    
    # Standardize Stock ID to String
    df_specs['標的代號'] = df_specs['標的代號'].astype(str)
    df_prices['stock_id'] = df_prices['stock_id'].astype(str)
    
    df_trades['日期'] = pd.to_datetime(df_trades['日期'])
    df_specs['日期'] = pd.to_datetime(df_specs['日期'])
    df_prices['date'] = pd.to_datetime(df_prices['date'])
    
    # Identify Top 50 Stocks by Volume Value in the Full Period (Jan - Aug 2025)
    # Increasing to 50 stocks to get a larger sample size
    valid_prices = df_prices[df_prices['date'].between('2025-01-01', '2025-08-31')]
    top_stocks = valid_prices.groupby('stock_id')['volume_value_1k'].sum().sort_values(ascending=False).head(50).index.tolist()
    
    print(f"Analyzing Top 50 Stocks over Jan-Aug 2025...")
    
    # 2. Process Warrant Data
    df_trades['net_buy_warrant'] = df_trades['買張'] - df_trades['賣張']
    df_specs_filtered = df_specs[df_specs['標的代號'].isin(top_stocks)]
    
    # Merge Trades with Specs
    merged = pd.merge(
        df_trades, 
        df_specs_filtered[['日期', '權證代號', '標的代號', '最新執行比例', 'IVDelta值', 'IVGamma值', '標的收盤價']], 
        on=['日期', '權證代號'], 
        how='inner'
    )
    
    if merged.empty:
        print("No matching warrant trades found.")
        return
        
    # 3. Calculate Implied Hedging Buy (in Lots/張)
    merged['implied_stock_buy_vol'] = (
        merged['net_buy_warrant'] * 
        merged['最新執行比例'] * 
        merged['IVDelta值']
    ) / 1000.0
    
    # 4. Aggregate across ALL branches per stock and date
    market_impact = merged.groupby(['日期', '標的代號']).agg({
        'implied_stock_buy_vol': 'sum',
        'net_buy_warrant': 'sum',
        '標的收盤價': 'mean',
        'IVGamma值': 'mean'
    }).reset_index()
    
    # Get daily returns
    df_prices_filtered = df_prices[df_prices['stock_id'].isin(top_stocks)].sort_values(['stock_id', 'date'])
    df_prices_filtered['stock_ret'] = df_prices_filtered.groupby('stock_id')['close'].pct_change()
    
    # Final Merge with Returns
    final_df = pd.merge(
        market_impact, 
        df_prices_filtered[['date', 'stock_id', 'stock_ret', 'volume_value_1k']], 
        left_on=['日期', '標的代號'], 
        right_on=['date', 'stock_id'], 
        how='inner'
    )
    
    # 5. Calculate Hedge Pressure %
    final_df['est_market_lots'] = final_df['volume_value_1k'] / final_df['標的收盤價']
    final_df['hedge_pressure_pct'] = (final_df['implied_stock_buy_vol'] / final_df['est_market_lots']) * 100
    
    # 6. Report Summary
    print(f"\nAnalysis Complete. Total sample size (n): {len(final_df)}")
    corr = final_df[['hedge_pressure_pct', 'stock_ret']].corr().iloc[0, 1]
    print(f"Overall Correlation (Full Period): {corr:.4f}")

    output_path = 'data/collective_warrant_impact_report.csv'
    final_df.to_csv(output_path, index=False)
    print(f"Report saved to {output_path}")

if __name__ == "__main__":
    analyze_warrant_hedging_impact()
