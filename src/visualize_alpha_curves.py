import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def calculate_car_for_groups():
    # Load Basic Data
    prices = pd.read_parquet('data/stock_price_history.parquet')
    prices['date'] = pd.to_datetime(prices['date'])
    market = pd.read_parquet('data/market_index.parquet')
    market['date'] = pd.to_datetime(market['date'])
    market = market[['date', 'market_ret']].rename(columns={'market_ret': 'mkt_ret'})
    
    # Load Events
    events = pd.read_csv('data/event_intensity_analysis.csv')
    events['ann_date'] = pd.to_datetime(events['ann_date'])
    
    # Load Announcements (to get NewHigh info)
    ann_df = pd.read_parquet('data/revenue_announcements.parquet')
    ann_df['announcement_date'] = pd.to_datetime(ann_df['announcement_date'])
    ann_df['is_high'] = ann_df['創新高/低(歷史)'].fillna('').astype(str).str.contains('H')
    
    # Window: T-10 to T+10
    WINDOW_SIZE = 10
    all_car_paths = []
    
    print(f"Calculating CAR for {len(events)} events...")
    
    for _, event in events.iterrows():
        stock_id = str(event['stock_id'])
        ann_date = event['ann_date']
        
        # Get matching announcement for NewHigh info
        stock_ann = ann_df[(ann_df['stock_id'] == stock_id) & (ann_df['announcement_date'] == ann_date)]
        is_high = stock_ann.iloc[0]['is_high'] if not stock_ann.empty else False
        
        # Get prices
        stock_prices = prices[prices['stock_id'] == stock_id].sort_values('date').copy()
        if stock_prices.empty: continue
        
        stock_prices['stock_ret'] = stock_prices['close'].pct_change()
        merged = pd.merge(stock_prices, market, on='date', how='left')
        merged['alpha'] = (merged['stock_ret'] - merged['mkt_ret']).fillna(0)
        
        # Find T0 index
        t0_indices = merged[merged['date'] <= ann_date].index
        if len(t0_indices) == 0: continue
        t0_idx = t0_indices[-1]
        
        # Window: T-10 to T+10
        start_idx = t0_idx - WINDOW_SIZE
        end_idx = t0_idx + WINDOW_SIZE
        if start_idx < 0 or end_idx >= len(merged): continue
        
        window = merged.iloc[start_idx : end_idx + 1].copy()
        
        # Calculate CAR: (T-10 to current_T)
        window['car'] = (1 + window['alpha']).cumprod() - 1
        car_values = window['car'].values * 100 # Convert to %
        
        all_car_paths.append({
            'stock_id': stock_id,
            'ann_date': ann_date,
            'combo': event['combo'],
            'is_high': is_high,
            'car_path': car_values
        })
        
    results_df = pd.DataFrame(all_car_paths)
    if results_df.empty: return
    
    # --- VISUALIZATION ---
    plt.figure(figsize=(15, 6))
    x = np.arange(-WINDOW_SIZE, WINDOW_SIZE + 1)
    
    # Plot 1: Combo vs Single
    plt.subplot(1, 2, 1)
    combo_cars = np.vstack(results_df[results_df['combo'] == 1]['car_path'].values)
    single_cars = np.vstack(results_df[results_df['combo'] == 0]['car_path'].values)
    
    plt.plot(x, np.mean(combo_cars, axis=0), label=f'Combo (C1+C3) n={len(combo_cars)}', lw=3, color='red')
    plt.plot(x, np.mean(single_cars, axis=0), label=f'Single Cluster n={len(single_cars)}', lw=3, color='blue', alpha=0.6)
    
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.title('CAR: Combo vs. Single Cluster (T-10 to T+10)', fontsize=14)
    plt.xlabel('Days relative to Announcement (T0)', fontsize=12)
    plt.ylabel('Cumulative Alpha (%)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Revenue New High vs Normal
    plt.subplot(1, 2, 2)
    high_cars = np.vstack(results_df[results_df['is_high'] == True]['car_path'].values)
    normal_cars = np.vstack(results_df[results_df['is_high'] == False]['car_path'].values)
    
    plt.plot(x, np.mean(high_cars, axis=0), label=f'Revenue New High n={len(high_cars)}', lw=3, color='green')
    plt.plot(x, np.mean(normal_cars, axis=0), label=f'Normal Revenue n={len(normal_cars)}', lw=3, color='gray', alpha=0.6)
    
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.title('CAR: New High vs. Normal (T-10 to T+10)', fontsize=14)
    plt.xlabel('Days relative to Announcement (T0)', fontsize=12)
    plt.ylabel('Cumulative Alpha (%)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_img = 'docs/alpha_comparison_curves.png'
    plt.savefig(output_img)
    print(f"\nCAR Curves visualization saved to {output_img}")

if __name__ == "__main__":
    calculate_car_for_groups()
