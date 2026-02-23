import pandas as pd
import glob
import os
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Configuration
STOCK_ID = '6215' # Changed to 6215 for better testing (denser data)
DATA_DIR = 'data/'
DEFAULT_LOOKBACK = 60 

def load_data(stock_id):
    """Loads transactions and applies an adaptive rolling window."""
    file_path = os.path.join(DATA_DIR, 'StockBranch.parquet')
    if not os.path.exists(file_path):
        return pd.DataFrame()

    try:
        df = pd.read_parquet(file_path, filters=[('CommodityId', '==', str(stock_id))])
        if df.empty: return pd.DataFrame()

        df = df.rename(columns={'Date': 'date', 'CommodityId': 'stock_id', 'SecuritiesTraderId': 'securities_trader_id', 'Price': 'price', 'Buy': 'buy', 'Sell': 'sell'})
        df['date'] = pd.to_datetime(df['date'])
        
        # [Optimization] Adaptive Window
        max_date = df['date'].max()
        window = DEFAULT_LOOKBACK
        
        # Check if 60 days is enough
        subset = df[df['date'] >= max_date - pd.Timedelta(days=window)]
        if subset['date'].nunique() < 5:
            print(f"Warning: Only {subset['date'].nunique()} trading days in last {window} days. Expanding window to 180 days...")
            window = 180
            subset = df[df['date'] >= max_date - pd.Timedelta(days=window)]
            
        subset = subset.copy()
        subset['date'] = subset['date'].dt.strftime('%Y-%m-%d')
        
        print(f"Loaded {len(subset)} transactions for {stock_id} (Adaptive Window: {window} days, {subset['date'].nunique()} trading days)")
        return subset
    except Exception as e:
        print(f"Error: {e}")
        return pd.DataFrame()

def extract_features(df):
    """Extracts behavioral features."""
    print("Extracting behavioral features...")
    broker_stats = df.groupby('securities_trader_id').apply(
        lambda x: pd.Series({
            'total_buy': x['buy'].sum(),
            'total_sell': x['sell'].sum(),
            'total_volume': x['buy'].sum() + x['sell'].sum(),
            'transaction_days': x['date'].nunique(),
            'total_days_in_period': df['date'].nunique(),
        }),
        include_groups=False # Silence future warning
    ).reset_index()
    
    broker_stats['frequency'] = broker_stats['transaction_days'] / broker_stats['total_days_in_period']
    broker_stats['net_volume'] = broker_stats['total_buy'] - broker_stats['total_sell']
    broker_stats['overnight_ratio'] = broker_stats['net_volume'].abs() / broker_stats['total_volume']
    broker_stats['avg_daily_vol'] = broker_stats['total_volume'] / broker_stats['transaction_days']
    broker_stats['log_avg_daily_vol'] = np.log1p(broker_stats['avg_daily_vol'])

    return broker_stats.fillna(0)

def perform_clustering(features_df, k=4):
    print(f"Performing Optimized K-Means (PCA + RobustScaler)...")
    active_brokers = features_df[features_df['transaction_days'] >= 2].copy() # Relaxed to 2 days
    
    if len(active_brokers) < k:
        print(f"Insufficient active brokers ({len(active_brokers)}). Skipping clustering.")
        return active_brokers
    
    cluster_features = ['frequency', 'overnight_ratio', 'log_avg_daily_vol']
    X = active_brokers[cluster_features]
    
    # Preprocessing Pipeline
    X_scaled = RobustScaler().fit_transform(X)
    X_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    
    # Clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    active_brokers['cluster'] = kmeans.fit_predict(X_pca)
    
    print("\n--- Optimized Cluster Summary ---")
    summary = active_brokers.groupby('cluster')[cluster_features].mean()
    summary['count'] = active_brokers['cluster'].value_counts()
    print(summary)
    
    return active_brokers

def run_analysis(stock_id):
    print(f"\n--- Model Update: {stock_id} ---")
    df = load_data(stock_id)
    if df.empty: return None
    
    features = extract_features(df)
    clustered = perform_clustering(features)
    
    if 'cluster' in clustered.columns:
        output_file = f'data/broker_clusters_{stock_id}.csv'
        clustered.to_csv(output_file, index=False)
        print(f"Updated results saved to {output_file}")
        return clustered
    return None

if __name__ == "__main__":
    run_analysis(STOCK_ID)