import pandas as pd
import os

def calculate_daily_hedging_flow():
    print("--- Starting Daily Issuer Hedging Flow Calculation ---")
    
    # 1. Load Warrant Specification Data
    specs_path = 'data/權證條件.parquet'
    if not os.path.exists(specs_path):
        print("Warrant specs file not found.")
        return

    df = pd.read_parquet(specs_path)
    
    # Standardize Data Types
    df['日期'] = pd.to_datetime(df['日期'])
    df['標的代號'] = df['標的代號'].astype(str)
    
    # 2. Calculate Individual Warrant Delta Exposure (in Shares)
    # Formula: Outstanding (1k units) * 1000 * Execution Ratio * Delta
    # This represents how many shares of stock the issuer needs to hold for ONE warrant ID
    df['warrant_delta_shares'] = (
        df['流通數量(千)'] * 1000 * 
        df['最新執行比例'] * 
        df['IVDelta值']
    )
    
    # 3. Aggregate by Stock and Date
    # Total hedge position required for each stock per day
    daily_pos = df.groupby(['日期', '標的代號']).agg({
        'warrant_delta_shares': 'sum',
        '標的收盤價': 'first'
    }).reset_index()
    
    # 4. Calculate Daily Flow (Difference from yesterday)
    daily_pos = daily_pos.sort_values(['標的代號', '日期'])
    
    # Daily Flow = Current Position - Previous Position
    # This represents the net buying/selling pressure from issuers
    daily_pos['hedging_flow_shares'] = daily_pos.groupby('標的代號')['warrant_delta_shares'].diff()
    
    # Convert to 'Lots' (張) for better readability in Taiwan market
    daily_pos['hedging_flow_lots'] = daily_pos['hedging_flow_shares'] / 1000.0
    
    # 5. Filter for Significant Flow and Report
    # Remove NaN from first day of data
    report_df = daily_pos.dropna(subset=['hedging_flow_lots'])
    
    print("\n--- Top 10 Stocks with Highest Positive Hedging Demand (Buy Pressure) ---")
    top_buy = report_df.sort_values('hedging_flow_lots', ascending=False).head(10)
    print(top_buy[['日期', '標的代號', 'hedging_flow_lots', '標的收盤價']].to_string(index=False))
    
    print("\n--- Top 10 Stocks with Highest Negative Hedging Demand (Sell Pressure) ---")
    top_sell = report_df.sort_values('hedging_flow_lots', ascending=True).head(10)
    print(top_sell[['日期', '標的代號', 'hedging_flow_lots', '標的收盤價']].to_string(index=False))
    
    # 6. Save Result
    output_path = 'data/issuer_hedging_flow_daily.csv'
    report_df.to_csv(output_path, index=False)
    print(f"\nCalculation complete. Daily flow report saved to {output_path}")

if __name__ == "__main__":
    calculate_daily_hedging_flow()
