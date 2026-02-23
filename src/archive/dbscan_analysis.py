import pandas as pd
import numpy as np
import os
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from broker_clustering import load_data, extract_features

# Configuration
STOCK_ID = '6215'

def run_dbscan_analysis(stock_id):
    print(f"--- DBSCAN Clustering Analysis for {stock_id} ---")
    
    # 1. Load and Extract Features (Reuse logic from broker_clustering)
    df_raw = load_data(stock_id)
    if df_raw.empty:
        return
    
    features_df = extract_features(df_raw)
    
    # Filter active brokers (at least 3 days of trading)
    active_brokers = features_df[features_df['transaction_days'] >= 3].copy()
    
    # 2. Prepare Features
    cluster_features = ['frequency', 'overnight_ratio', 'log_avg_daily_vol']
    X = active_brokers[cluster_features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. DBSCAN Clustering
    # eps: 鄰域半徑。這個值需要根據資料分佈調整，0.5 是一個常見的起點。
    # min_samples: 成為核心點所需的最少鄰居數。
    print("\nRunning DBSCAN algorithm...")
    dbscan = DBSCAN(eps=0.5, min_samples=5) 
    active_brokers['cluster'] = dbscan.fit_predict(X_scaled)
    
    # Calculate stats
    clusters = set(active_brokers['cluster'])
    n_clusters = len([c for c in clusters if c >= 0])
    n_noise = list(active_brokers['cluster']).count(-1)
    
    print(f"Estimated number of clusters: {n_clusters}")
    print(f"Estimated number of noise points: {n_noise} (out of {len(active_brokers)})")
    
    # 4. UMAP Visualization
    print("Running UMAP for visualization...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    active_brokers['umap_1'] = embedding[:, 0]
    active_brokers['umap_2'] = embedding[:, 1]
    
    # 5. Plotting
    plt.figure(figsize=(12, 8))
    
    # We use a specific color for noise (-1)
    palette = sns.color_palette("husl", n_clusters)
    color_map = {c: palette[i] if c >= 0 else (0.8, 0.8, 0.8) for i, c in enumerate(sorted([c for c in clusters if c >= 0]))}
    color_map[-1] = (0.5, 0.5, 0.5) # Gray for noise
    
    sns.scatterplot(
        data=active_brokers,
        x='umap_1',
        y='umap_2',
        hue='cluster',
        size='log_avg_daily_vol',
        palette=color_map,
        alpha=0.6,
        sizes=(20, 200)
    )
    
    # Annotate top brokers in non-noise clusters
    smart_candidates = active_brokers[active_brokers['cluster'] >= 0].sort_values('overnight_ratio', ascending=False).head(5)
    for _, row in smart_candidates.iterrows():
        plt.annotate(
            row['securities_trader_id'],
            (row['umap_1'], row['umap_2']),
            textcoords="offset points",
            xytext=(0,10),
            ha='center',
            fontsize=8,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.5)
        )

    plt.title(f'DBSCAN Broker Map - Stock {stock_id}\n(Clusters: {n_clusters}, Noise: {n_noise})', fontsize=15)
    plt.legend(title='Cluster ID (-1=Noise)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    os.makedirs('docs', exist_ok=True)
    output_img = f'docs/dbscan_map_{stock_id}.png'
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f"Visualization saved to {output_img}")
    
    # Cluster Analysis
    print("\n--- Cluster Behavioral Characteristics ---")
    if n_clusters > 0 or n_noise > 0:
        summary = active_brokers.groupby('cluster')[cluster_features].mean()
        summary['count'] = active_brokers['cluster'].value_counts()
        print(summary)
    
    return active_brokers

if __name__ == "__main__":
    run_dbscan_analysis(STOCK_ID)
