import pandas as pd
import os

def analyze_intraday_vs_nextday():
    print("--- 📊 Analyzing Intraday vs. Next-Day Performance for Smart Warrant Signals ---")
    
    # 1. Load Smart Warrant Results
    smart_results_path = 'data/smart_warrant_results.csv'
    if not os.path.exists(smart_results_path):
        print("Smart warrant results not found. Run strategy_smart_warrant_pressure.py first.")
        return
        
    df_smart = pd.read_csv(smart_results_path)
    df_smart['date'] = pd.to_datetime(df_smart['date'])
    df_smart['stock_id'] = df_smart['stock_id'].astype(str)
    
    # Filter for the 35 high-confidence cases (Smart Pressure > 2%)
    signals = df_smart[df_smart['smart_pressure_ratio'] > 2].copy()
    print(f"Analyzing {len(signals)} Smart Warrant signals...")

    # 2. Load Price History to get OPEN prices
    price_history_path = 'data/stock_price_history.parquet'
    df_prices = pd.read_parquet(price_history_path)
    df_prices['date'] = pd.to_datetime(df_prices['date'])
    df_prices['stock_id'] = df_prices['stock_id'].astype(str)
    
    # 3. Merge to get T-Day Open/Close
    merged = pd.merge(
        signals,
        df_prices[['date', 'stock_id', 'open', 'close']],
        on=['date', 'stock_id'],
        how='inner',
        suffixes=('', '_actual')
    )
    
    # 4. Calculate Intraday Return (T-Day)
    # (Close - Open) / Open
    merged['intraday_ret'] = (merged['close_actual'] - merged['open']) / merged['open']
    
    # 5. Comparative Analysis
    print("\n" + "="*50)
    print(f"{'Metric':<25} | {'Value':<10}")
    print("-" * 50)
    print(f"{'T-Day Intraday Return':<25} | {merged['intraday_ret'].mean()*100:>8.2f}%")
    print(f"{'T-Day Win Rate (>0)':<25} | {(merged['intraday_ret'] > 0).mean()*100:>8.2f}%")
    print("-" * 50)
    print(f"{'T+1 Next-Day Return':<25} | {merged['next_day_ret'].mean()*100:>8.2f}%")
    print(f"{'T+1 Win Rate (>0)':<25} | {(merged['next_day_ret'] > 0).mean()*100:>8.2f}%")
    print("="*50)

    # 6. Detailed Look at Top 10 Cases
    print("\n--- Top 10 Smart Signal Performance Breakdown ---")
    top_10 = merged.sort_values('smart_pressure_ratio', ascending=False).head(10)
    print(top_10[['date', 'stock_id', 'smart_pressure_ratio', 'intraday_ret', 'next_day_ret']].to_string(index=False, formatters={
        'smart_pressure_ratio': '{:.2f}%'.format,
        'intraday_ret': '{:.2%}'.format,
        'next_day_ret': '{:.2%}'.format
    }))

    # Save detailed analysis
    merged.to_csv('data/smart_intraday_analysis.csv', index=False)
    print(f"Detailed report saved to data/smart_intraday_analysis.csv")

if __name__ == "__main__":
    analyze_intraday_vs_nextday()
