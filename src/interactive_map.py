import pandas as pd
import numpy as np
import os
import plotly.express as px
import umap
from sklearn.preprocessing import StandardScaler

# Configuration
STOCK_ID = '6215'
INPUT_FILE = f'data/broker_clusters_{STOCK_ID}.csv'
OUTPUT_FILE = f'docs/interactive_map_{STOCK_ID}.html'

def generate_interactive_map():
    # Input Validation
    if not os.path.exists(INPUT_FILE):
        print(f"[Error] Input file not found: {INPUT_FILE}")
        return

    print(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # Data Integrity Check
    required_cols = ['frequency', 'overnight_ratio', 'log_avg_daily_vol']
    if not all(col in df.columns for col in required_cols):
        print(f"[Error] Missing columns. Required: {required_cols}")
        return

    # Feature Preprocessing
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[required_cols])

    # UMAP Dimensionality Reduction
    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_jobs=1)
    embedding = reducer.fit_transform(X_scaled)
    
    df['umap_x'] = embedding[:, 0]
    df['umap_y'] = embedding[:, 1]
    
    # Identify "Smart Money" Cluster for highlighting
    # Logic: The cluster with highest overnight ratio
    summary = df.groupby('cluster')['overnight_ratio'].mean()
    smart_cluster_id = summary.idxmax()
    
    # Add a column for color-coding clearly
    df['Group'] = df['cluster'].apply(lambda x: 'Smart Money' if x == smart_cluster_id else f'Cluster {x}')

    # Visualization using Plotly
    print("Generating Plotly chart...")
    fig = px.scatter(
        df, 
        x='umap_x', 
        y='umap_y', 
        color='Group',
        symbol='Group', # Different shapes for different groups
        size='log_avg_daily_vol',
        hover_name='securities_trader_id',
        hover_data={
            'umap_x': False, 
            'umap_y': False,
            'cluster': True,
            'overnight_ratio': ':.2f',
            'frequency': ':.2f',
            'total_buy': ':,', 
            'total_sell': ':,' 
        },
        title=f'Interactive Broker Map: {STOCK_ID} (6215) - Smart Money Detection',
        template='plotly_dark',
        color_discrete_map={'Smart Money': '#00FF00'} # Highlight Smart Money in Green
    )
    
    # Enhance Layout
    fig.update_layout(
        legend_title_text='Broker Group',
        xaxis_title="Behavioral Dimension 1",
        yaxis_title="Behavioral Dimension 2"
    )

    # Save
    os.makedirs('docs', exist_ok=True)
    fig.write_html(OUTPUT_FILE)
    print(f"[Success] Interactive map saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_interactive_map()
