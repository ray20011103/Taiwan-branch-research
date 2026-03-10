import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

# Configuration
DATA_DIR = 'data/'
TOP_BROKERS = ['8960', '8890', '1360'] # The consistent smart money identified
TARGET_STOCKS = ['1536', '3645', '3450', '6558', '3706', '4931', '3013', '2365', '1815', '8096', '2408', '1514', '6215', '2486', '4510', '6140']

def analyze_broker_timing():
    print(f"Generating timing fingerprints for top brokers: {TOP_BROKERS}")
    
    # 1. Load Price and Announcement Data
    ann_df = pd.read_csv('data/announcement.csv')
    ann_df['ann_date'] = pd.to_datetime(ann_df['營收發布日'])
    ann_df['stock_id'] = ann_df['公司'].str.split(' ').str[0]
    
    # 2. Load Transaction Data for Top Brokers
    file_path = os.path.join(DATA_DIR, 'StockBranch.parquet')
    all_trades = []
    
    for stock_id in tqdm(TARGET_STOCKS, desc="Analyzing timing across stocks"):
        try:
            df = pd.read_parquet(file_path, filters=[
                ('CommodityId', '==', str(stock_id)),
                ('SecuritiesTraderId', 'in', TOP_BROKERS)
            ])
            if not df.empty:
                df = df.rename(columns={'Date': 'date', 'CommodityId': 'stock_id', 'SecuritiesTraderId': 'broker_id', 'Buy': 'buy', 'Sell': 'sell'})
                df['date'] = pd.to_datetime(df['date'])
                all_trades.append(df)
        except Exception as e:
            print(f"Error: {e}")
            
    if not all_trades: return
    
    df_trades = pd.concat(all_trades)
    
    # 3. Calculate "Days Before Announcement" for each trade
    results = []
    for _, trade in df_trades.iterrows():
        # Get matching announcement for this stock in Jan 2025
        # (Assuming we are focusing on the Jan 2025 window for consistency)
        match = ann_df[(ann_df['stock_id'] == trade['stock_id']) & (ann_df['ann_date'] >= '2025-01-01') & (ann_df['ann_date'] <= '2025-01-20')]
        
        if not match.empty:
            ann_date = match.iloc[0]['ann_date']
            days_diff = (trade['date'] - ann_date).days
            
            # Focus on trades within T-15 to T+2
            if -15 <= days_diff <= 2:
                results.append({
                    'broker_id': trade['broker_id'],
                    'stock_id': trade['stock_id'],
                    'days_to_ann': days_diff,
                    'net_buy': trade['buy'] - trade['sell']
                })
                
    plot_df = pd.DataFrame(results)
    
    # 4. Plot Fingerprint (Heatmap or Strip Plot)
    plt.figure(figsize=(14, 8))
    sns.set_style("whitegrid")
    
    # Use stripplot to show individual buying events relative to T-0
    sns.stripplot(
        data=plot_df, 
        x='days_to_ann', 
        y='broker_id', 
        hue='broker_id',
        size=10, 
        alpha=0.5, 
        jitter=0.2,
        palette='Set1'
    )
    
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Revenue Announcement (T-0)')
    plt.title("Behavioral Fingerprints: Timing of Smart Money Entry", fontsize=16, fontweight='bold')
    plt.xlabel("Days Relative to Revenue Announcement (T-0)", fontsize=12)
    plt.ylabel("Core Smart Broker ID", fontsize=12)
    plt.legend()
    
    output_path = 'docs/broker_timing_fingerprint.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Behavioral Fingerprint saved to {output_path}")

if __name__ == "__main__":
    analyze_broker_timing()
