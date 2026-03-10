import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

# Configuration
DATA_DIR = 'data/'
TARGET_STOCKS = [
    '1536', '3645', '3450', '6558', '3706', '4931',
    '3013', '2365', '1815', '8096', '2408', '1514', '6215', '2486', '4510', '6140'
]

def load_all_stock_data():
    """Loads and merges transaction data for all target stocks."""
    file_path = os.path.join(DATA_DIR, 'StockBranch.parquet')
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return pd.DataFrame()

    all_dfs = []
    print(f"Loading data for {len(TARGET_STOCKS)} stocks...")
    
    for stock_id in tqdm(TARGET_STOCKS):
        try:
            df = pd.read_parquet(file_path, filters=[('CommodityId', '==', str(stock_id))])
            if not df.empty:
                df = df.rename(columns={
                    'Date': 'date',
                    'CommodityId': 'stock_id',
                    'SecuritiesTraderId': 'broker_id',
                    'Buy': 'buy',
                    'Sell': 'sell'
                })
                all_dfs.append(df)
        except Exception as e:
            print(f"Error loading {stock_id}: {e}")
            
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

def analyze_consistency(df):
    """Calculates behavioral consistency for each broker across stocks."""
    print("Calculating broker behavior fingerprints...")
    
    # Calculate Daily Net Position per Broker per Stock
    df['net_buy'] = df['buy'] - df['sell']
    broker_stock_daily = df.groupby(['broker_id', 'stock_id', 'date'])['net_buy'].sum().reset_index()
    
    # Calculate Overnight Ratio (Proxy for Holding intent)
    # Simplified: % of days where net position is non-zero (active)
    # Refined: We reuse the concept from original clustering
    results = []
    
    for (broker, stock), group in tqdm(df.groupby(['broker_id', 'stock_id']), desc="Processing Broker-Stock pairs"):
        total_buy = group['buy'].sum()
        total_sell = group['sell'].sum()
        total_vol = total_buy + total_sell
        
        if total_vol == 0: continue
        
        # Overnight ratio = (Total Buy - Total Sell) / Total Volume (Absolute)
        # Higher absolute value means biased towards one side (Accumulating or Distributing)
        overnight_ratio = abs(total_buy - total_sell) / total_vol
        
        results.append({
            'broker_id': broker,
            'stock_id': stock,
            'overnight_ratio': overnight_ratio,
            'total_vol': total_vol
        })
        
    res_df = pd.DataFrame(results)
    
    # AGGREGATE ACROSS STOCKS
    consistency = res_df.groupby('broker_id').agg({
        'stock_id': 'count',
        'overnight_ratio': 'mean',
        'total_vol': 'sum'
    }).reset_index()
    
    consistency.columns = ['broker_id', 'stock_count', 'avg_overnight_ratio', 'total_vol_sum']
    
    return consistency, res_df

def plot_consistency_map(consistency):
    """Generates the Global Smart Money Map."""
    plt.figure(figsize=(12, 8))
    
    # Filter for active brokers to reduce noise
    plot_df = consistency[consistency['total_vol_sum'] > 100].copy()
    
    # Add some jitter to stock_count for better visualization
    plot_df['stock_count_jitter'] = plot_df['stock_count'] + np.random.normal(0, 0.1, size=len(plot_df))
    
    sns.scatterplot(
        data=plot_df, 
        x='stock_count_jitter', 
        y='avg_overnight_ratio', 
        size='total_vol_sum', 
        hue='avg_overnight_ratio',
        palette='viridis',
        alpha=0.6,
        sizes=(20, 500)
    )
    
    # Highlight the 'Global Smart' Quadrant
    plt.axvline(2.5, color='red', linestyle='--', alpha=0.5) # Brokers in > 2 stocks
    plt.axhline(0.6, color='red', linestyle='--', alpha=0.5) # High conviction
    
    plt.text(3, 0.8, "GLOBAL SMART MONEY\n(Consistently Accumulating)", color='red', fontweight='bold', fontsize=12)
    
    plt.title("Global Broker Consistency Map (Across 16 Stocks)", fontsize=16)
    plt.xlabel("Number of Stocks Participated", fontsize=12)
    plt.ylabel("Average Overnight Ratio (Behavioral Style)", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    output_path = 'docs/global_broker_consistency.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Global Consistency Map saved to {output_path}")

if __name__ == "__main__":
    raw_df = load_all_stock_data()
    if not raw_df.empty:
        consistency_df, raw_results = analyze_consistency(raw_df)
        
        # Display Top Consistent Brokers
        top_smart = consistency_df[consistency_df['stock_count'] >= 2].sort_values('avg_overnight_ratio', ascending=False)
        print("\n--- Top Global Smart Brokers (Active in 2+ Target Stocks) ---")
        print(top_smart.head(15).to_string(index=False))
        
        plot_consistency_map(consistency_df)
        
        # Save results for further report integration
        consistency_df.to_csv('data/global_broker_consistency.csv', index=False)
