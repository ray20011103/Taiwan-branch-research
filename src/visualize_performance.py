import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_performance_summary():
    # 1. Load Data
    events = pd.read_csv('data/event_intensity_analysis.csv')
    ann_df = pd.read_parquet('data/revenue_announcements.parquet')
    
    # Pre-process Announcement Data for New High info
    ann_df['announcement_date'] = pd.to_datetime(ann_df['announcement_date'])
    ann_df['is_high'] = ann_df['創新高/低(歷史)'].fillna('').astype(str).str.contains('H')
    
    # Merge events with New High info
    events['ann_date'] = pd.to_datetime(events['ann_date'])
    # Ensure stock_id is string in both DataFrames
    events['stock_id'] = events['stock_id'].astype(str)
    ann_df['stock_id'] = ann_df['stock_id'].astype(str)
    
    df = pd.merge(events, ann_df[['stock_id', 'announcement_date', 'is_high']], 
                  left_on=['stock_id', 'ann_date'], 
                  right_on=['stock_id', 'announcement_date'], 
                  how='left')
    
    # Calculate Win/Loss (Pre-Alpha > 0)
    df['win'] = (df['pre_alpha'] > 0).astype(int)
    
    # 2. Define Groups for Analysis
    group_configs = {
        'Combo Effect': 'combo',
        'Revenue Quality': 'is_high',
        'Intensity Quartile': 'c3_quartile'
    }
    
    summary_list = []
    
    # 3. Create Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (title, col) in enumerate(group_configs.items()):
        # Calculate stats
        stats = df.groupby(col, observed=False).agg({
            'pre_alpha': 'mean',
            'win': 'mean',
            'stock_id': 'count'
        }).reset_index()
        
        # Mapping labels for better readability
        if col == 'combo':
            stats[col] = stats[col].map({1: 'Combo (C1+C3)', 0: 'Single Cluster'})
        elif col == 'is_high':
            stats[col] = stats[col].map({True: 'New High', False: 'Normal'})
        
        # Plotting Mean Alpha (Bar)
        sns.barplot(x=col, y='pre_alpha', data=stats, ax=axes[i], palette='viridis', hue=col, legend=False)
        
        # Adding Win Rate as text labels on top of bars
        for idx, row in stats.iterrows():
            axes[i].text(idx, row['pre_alpha'] + 0.2, f"Win: {row['win']*100:.1f}%\nn={row['stock_id']}", 
                         ha='center', va='bottom', fontsize=10, fontweight='bold')
            
        axes[i].set_title(f'Impact of {title}', fontsize=14, fontweight='bold')
        axes[i].set_ylabel('Avg Pre-Alpha (T-5 to T0) %', fontsize=12)
        axes[i].set_xlabel('')
        axes[i].grid(axis='y', alpha=0.3)
        
        # Store for CSV output
        stats['category'] = title
        summary_list.append(stats)
        
    plt.suptitle('Strategy Performance Deep-Dive: 187 Revenue Events (2024-2025 H1)', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save Image
    output_img = 'docs/strategy_performance_visual.png'
    plt.savefig(output_img, bbox_inches='tight')
    print(f"Performance visualization saved to {output_img}")
    
    # Save Stats CSV
    final_stats = pd.concat(summary_list)
    final_stats.to_csv('data/strategy_performance_stats.csv', index=False)
    print("Detailed statistics saved to data/strategy_performance_stats.csv")

if __name__ == "__main__":
    run_performance_summary()
