import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler

# Configuration
STOCK_ID = '6215' # The "Front-Running Champion"
DATA_PATH = f'data/broker_clusters_{STOCK_ID}.csv'

def visualize_broker_map():
    print(f"Loading cluster data for {STOCK_ID}...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found. Please run batch_clustering first.")
        return

    df = pd.read_csv(DATA_PATH)
    
    # 1. Prepare Features for UMAP
    # These are the features used in clustering
    features = ['frequency', 'overnight_ratio', 'log_avg_daily_vol']
    X = df[features]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Run UMAP for Dimensionality Reduction (to 2D)
    print("Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(X_scaled)
    
    df['umap_1'] = embedding[:, 0]
    df['umap_2'] = embedding[:, 1]
    
    # 3. Create Visualization
    print("Generating plot...")
    plt.figure(figsize=(12, 8))
    
    # Use seaborn for scatter plot
    scatter = sns.scatterplot(
        data=df,
        x='umap_1',
        y='umap_2',
        hue='cluster',
        size='log_avg_daily_vol',
        palette='viridis',
        alpha=0.6,
        sizes=(20, 200)
    )
    
    # Highlight specific interesting brokers (e.g., top volume in Cluster 3)
    # Usually Cluster 3 or 0 in our previous run was the "Accumulator"
    # Let's find the cluster with highest overnight ratio
    summary = df.groupby('cluster')['overnight_ratio'].mean()
    smart_cluster_id = summary.idxmax()
    
    smart_money = df[df['cluster'] == smart_cluster_id].sort_values('total_volume', ascending=False).head(5)
    
    for _, row in smart_money.iterrows():
        plt.annotate(
            row['securities_trader_id'],
            (row['umap_1'], row['umap_2']),
            textcoords="offset points",
            xytext=(0,10),
            ha='center',
            fontsize=9,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3)
        )

    plt.title(f'Broker Behavioral Map - Stock {STOCK_ID} (UMAP Projection)', fontsize=15)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(title='Cluster ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    output_img = f'docs/broker_map_{STOCK_ID}.png'
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f"Visualization saved to {output_img}")
    
    # Show cluster characteristics for reference
    print("\n--- Cluster Behavioral Characteristics ---")
    char = df.groupby('cluster')[features].mean()
    char['count'] = df['cluster'].value_counts()
    print(char)

if __name__ == "__main__":
    # Ensure docs directory exists
    os.makedirs('docs', exist_ok=True)
    visualize_broker_map()
