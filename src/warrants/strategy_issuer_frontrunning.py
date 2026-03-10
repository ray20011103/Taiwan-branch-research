import pandas as pd
import os
import numpy as np

def run_issuer_frontrunning_analysis():
    print("--- 🚀 Strategy: Issuer Front-running (Warrant Hedge Pressure) ---")
    
    # 1. Load Data
    warrant_trades_path = 'data/分點進出.parquet'
    warrant_specs_path = 'data/權證條件.parquet'
    stock_history_path = 'data/stock_price_history.parquet'
    
    if not all(os.path.exists(p) for p in [warrant_trades_path, warrant_specs_path, stock_history_path]):
        print("Required data files not found.")
        return

    df_w_trades = pd.read_parquet(warrant_trades_path)
    df_w_specs = pd.read_parquet(warrant_specs_path)
    df_prices = pd.read_parquet(stock_history_path)
    
    # 2. Preprocessing
    df_w_specs['標的代號'] = df_w_specs['標的代號'].astype(str)
    df_prices['stock_id'] = df_prices['stock_id'].astype(str)
    df_prices['date'] = pd.to_datetime(df_prices['date'])
    df_w_trades['date'] = pd.to_datetime(df_w_trades['日期'])
    
    # Estimate volume in lots (volume_value_1k is in 1000 TWD, so / close gives roughly number of lots)
    df_prices['est_lots'] = df_prices['volume_value_1k'] / df_prices['close']
    
    # Calculate 20-day Moving Average Volume for stocks
    df_prices = df_prices.sort_values(['stock_id', 'date'])
    df_prices['avg_lots_20d'] = df_prices.groupby('stock_id')['est_lots'].transform(lambda x: x.rolling(20).mean())
    
    # 3. Calculate Implied Hedge Demand per Warrant per Day
    # Merge trades with specs
    merged_w = pd.merge(
        df_w_trades,
        df_w_specs[['日期', '權證代號', '標的代號', '最新執行比例', 'IVDelta值']],
        left_on=['date', '權證代號'],
        right_on=['日期', '權證代號'],
        how='inner'
    )
    
    # Filter for CALL warrants (Delta > 0)
    # Put warrants would have Delta < 0
    merged_w = merged_w[merged_w['IVDelta值'] > 0]
    
    # Implied Stock Buy (Lots) = (Warrant Net Buy) * Ratio * Delta
    merged_w['hedge_lots'] = (merged_w['買張'] - merged_w['賣張']) * merged_w['最新執行比例'] * merged_w['IVDelta值']
    
    # 4. Aggregate by Stock and Date
    daily_hedge = merged_w.groupby(['date', '標的代號'])['hedge_lots'].sum().reset_index()
    
    # 5. Merge with Stock Market Data
    strategy_df = pd.merge(
        daily_hedge,
        df_prices[['date', 'stock_id', 'close', 'avg_lots_20d', 'est_lots']],
        left_on=['date', '標的代號'],
        right_on=['date', 'stock_id'],
        how='inner'
    )
    
    # 6. Calculate Hedge Pressure Ratio
    # How much of the stock's average daily volume is driven by warrant hedging?
    strategy_df['hedge_pressure_ratio'] = (strategy_df['hedge_lots'] / strategy_df['avg_lots_20d']) * 100
    
    # 7. Evaluate Next Day Performance
    strategy_df = strategy_df.sort_values(['stock_id', 'date'])
    strategy_df['next_day_ret'] = strategy_df.groupby('stock_id')['close'].pct_change().shift(-1)
    
    # Filter for significant pressure (e.g., Hedge Pressure > 5% of Avg Volume)
    high_pressure = strategy_df[strategy_df['hedge_pressure_ratio'] > 5].copy()
    
    print(f"\nAnalysis complete. Found {len(high_pressure)} high pressure events (> 5% of avg vol).")
    
    if not high_pressure.empty:
        print("\n--- Top 10 High Hedge Pressure Events ---")
        print(high_pressure.sort_values('hedge_pressure_ratio', ascending=False).head(10)[['date', 'stock_id', 'hedge_lots', 'hedge_pressure_ratio', 'next_day_ret']])
        
        # Summary Stats
        win_rate = (high_pressure['next_day_ret'] > 0).mean()
        avg_ret = high_pressure['next_day_ret'].mean()
        print(f"\nWin Rate (Next Day > 0): {win_rate*100:.2f}%")
        print(f"Average Next Day Return: {avg_ret*100:.4f}%")
        
        # Breakdown by Pressure Tiers
        high_pressure['tier'] = pd.cut(high_pressure['hedge_pressure_ratio'], bins=[5, 10, 20, 50, 1000], labels=['5-10%', '10-20%', '20-50%', '>50%'])
        print("\n--- Performance by Pressure Tier ---")
        tier_stats = high_pressure.groupby('tier', observed=True)['next_day_ret'].agg(['count', 'mean', lambda x: (x > 0).mean()]).rename(columns={'mean': 'avg_ret_pct', '<lambda_0>': 'win_rate_pct'})
        tier_stats['avg_ret_pct'] *= 100
        tier_stats['win_rate_pct'] *= 100
        print(tier_stats)

    output_path = 'data/issuer_frontrunning_results.csv'
    strategy_df.to_csv(output_path, index=False)
    print(f"\nFull results saved to {output_path}")

if __name__ == "__main__":
    run_issuer_frontrunning_analysis()
