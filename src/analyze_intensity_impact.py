import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def load_all_events():
    # Load Price History
    prices = pd.read_parquet('data/stock_price_history.parquet')
    prices['date'] = pd.to_datetime(prices['date'])
    
    # Load Market Index
    market = pd.read_parquet('data/market_index.parquet')
    market['date'] = pd.to_datetime(market['date'])
    market = market[['date', 'market_ret']].rename(columns={'market_ret': 'mkt_ret'})
    
    # Load Announcements (2024-2025 H1)
    ann_df = pd.read_parquet('data/revenue_announcements.parquet')
    ann_df['announcement_date'] = pd.to_datetime(ann_df['announcement_date'])
    # Filter for 2024-01-01 to 2025-06-30
    ann_df = ann_df[(ann_df['announcement_date'] >= '2024-01-01') & (ann_df['announcement_date'] <= '2025-06-30')]
    
    bps_files = glob.glob('data/smart_bps_result_*.csv')
    stock_ids = [f.split('_')[-1].split('.')[0] for f in bps_files]
    
    event_results = []
    
    for stock_id in stock_ids:
        # Load BPS (both C1 and C3)
        bps_path = f'data/smart_bps_result_{stock_id}.csv'
        bps_df = pd.read_csv(bps_path)
        if 'bps_c3' not in bps_df.columns or 'bps_c1' not in bps_df.columns:
            continue
        bps_df['date'] = pd.to_datetime(bps_df['date'])
        
        # Price Data
        stock_prices = prices[prices['stock_id'] == stock_id].sort_values('date').copy()
        if stock_prices.empty:
            continue
        stock_prices['stock_ret'] = stock_prices['close'].pct_change()
        
        # Merge
        merged = pd.merge(stock_prices, bps_df[['date', 'bps_c1', 'bps_c3']], on='date', how='left')
        merged = pd.merge(merged, market, on='date', how='left')
        merged['alpha'] = merged['stock_ret'] - merged['mkt_ret']
        merged = merged.fillna(0).reset_index(drop=True)
        
        stock_anns = ann_df[ann_df['stock_id'] == stock_id]
        for _, ann in stock_anns.iterrows():
            try:
                # Find T0 (Announcement Date)
                t0_indices = merged[merged['date'] <= ann['announcement_date']].index
                if len(t0_indices) == 0: continue
                t0_idx = t0_indices[-1]
                
                # Window: T-5 to T+2
                start_idx = t0_idx - 5
                end_idx = t0_idx + 2
                if start_idx < 0 or end_idx >= len(merged):
                    continue
                
                window = merged.iloc[start_idx : end_idx + 1].copy()
                
                # Intensity (T-5 to T-1)
                c3_intensity = window.iloc[0:5]['bps_c3'].sum()
                c1_intensity = window.iloc[0:5]['bps_c1'].sum()
                
                # Pre-Alpha (T-5 to T0)
                pre_window = window.iloc[0:6]
                pre_alpha = (1 + pre_window['alpha']).prod() - 1
                
                # Post-Alpha (T0 to T+2)
                post_window = window.iloc[5:8] # T0, T+1, T+2
                post_alpha = (1 + post_window['alpha']).prod() - 1
                
                if c3_intensity > 0 or c1_intensity > 0:
                    event_results.append({
                        'stock_id': stock_id,
                        'ann_date': ann['announcement_date'],
                        'c3_intensity': c3_intensity,
                        'c1_intensity': c1_intensity,
                        'pre_alpha': pre_alpha * 100,
                        'post_alpha': post_alpha * 100,
                        'combo': 1 if (c3_intensity > 0 and c1_intensity > 0) else 0
                    })
            except Exception as e:
                continue
            
    return pd.DataFrame(event_results)

def run_analysis():
    print("--- Analyzing Multi-Cluster Buy Intensity Impact ---")
    df = load_all_events()
    if df.empty:
        print("No events found.")
        return
    
    # 1. C3 Intensity Analysis
    df['c3_quartile'] = pd.qcut(df['c3_intensity'].rank(method='first'), 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    c3_summary = df.groupby('c3_quartile', observed=False).agg({
        'pre_alpha': 'mean',
        'post_alpha': 'mean',
        'stock_id': 'count'
    }).round(2)
    
    print("\nImpact Summary (by C3 Quant Intensity):")
    print(c3_summary)
    
    # 2. Combo Effect (Story + Quant)
    combo_summary = df.groupby('combo', observed=False).agg({
        'pre_alpha': 'mean',
        'post_alpha': 'mean',
        'stock_id': 'count'
    }).round(2)
    
    print("\nImpact Summary (Combo vs Single Cluster):")
    print("Combo=0: Only one cluster buying | Combo=1: Both C1 & C3 buying")
    print(combo_summary)
    
    # 3. Correlation
    corr = df[['c3_intensity', 'c1_intensity', 'pre_alpha', 'post_alpha']].corr().round(3)
    print("\nCorrelation Matrix:")
    print(corr)
    
    # Visualization: Scatter C3 vs Pre-Alpha
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(df['c3_intensity'], df['pre_alpha'], alpha=0.6, c=df['combo'], cmap='coolwarm')
    plt.xscale('symlog')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.title('C3 Intensity vs Pre-Alpha (Color=Combo)')
    plt.xlabel('C3 Net Buy (Log)')
    plt.ylabel('Pre-Alpha %')
    
    plt.subplot(1, 2, 2)
    plt.scatter(df['c1_intensity'], df['pre_alpha'], alpha=0.6, c=df['combo'], cmap='viridis')
    plt.xscale('symlog')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.title('C1 Intensity vs Pre-Alpha')
    plt.xlabel('C1 Net Buy (Log)')
    plt.ylabel('Pre-Alpha %')
    
    plt.tight_layout()
    output_img = 'docs/intensity_impact_analysis.png'
    plt.savefig(output_img)
    print(f"\nAnalysis plot saved to {output_img}")
    
    # Save statistics
    df.to_csv('data/event_intensity_analysis.csv', index=False)

if __name__ == "__main__":
    run_analysis()
