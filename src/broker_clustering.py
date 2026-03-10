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
    """Loads transactions and strictly filters for active stocks."""
    file_path = os.path.join(DATA_DIR, 'StockBranch.parquet')
    if not os.path.exists(file_path):
        return pd.DataFrame()

    try:
        # Use pyarrow filters
        df = pd.read_parquet(file_path, filters=[('CommodityId', '==', str(stock_id))])
        if df.empty: return pd.DataFrame()

        df = df.rename(columns={'Date': 'date', 'CommodityId': 'stock_id', 'SecuritiesTraderId': 'securities_trader_id', 'Price': 'price', 'Buy': 'buy', 'Sell': 'sell'})
        df['date'] = pd.to_datetime(df['date'])
        
        # [RELAXED] Frequent Trading Filter
        max_date = df['date'].max()
        window = DEFAULT_LOOKBACK # 60 days
        
        subset = df[df['date'] >= max_date - pd.Timedelta(days=window)]
        num_trading_days = subset['date'].nunique()
        
        # If trading days < 5 in the last 60 days, we consider it 'inactive'
        if num_trading_days < 5:
            print(f"Skipping {stock_id}: Infrequent trading ({num_trading_days} days in last {window} days)")
            return pd.DataFrame()
            
        subset = subset.copy()
        subset['date'] = subset['date'].dt.strftime('%Y-%m-%d')
        
        print(f"Loaded {len(subset)} transactions for {stock_id} (Active Window: {window} days, {num_trading_days} trading days)")
        return subset
    except Exception as e:
        print(f"Error loading data for {stock_id}: {e}")
        return pd.DataFrame()

def extract_features(df):
    """Extracts behavioral and performance features."""
    print("Extracting features (Behavioral + Performance)...")
    
    # Get current price (the latest close in the period)
    last_price = df.sort_values('date')['price'].iloc[-1]
    
    broker_stats = df.groupby('securities_trader_id').apply(
        lambda x: pd.Series({
            'total_buy': x['buy'].sum(),
            'total_sell': x['sell'].sum(),
            'total_buy_amt': (x['buy'] * x['price']).sum(),
            'total_sell_amt': (x['sell'] * x['price']).sum(),
            'total_volume': x['buy'].sum() + x['sell'].sum(),
            'transaction_days': x['date'].nunique(),
            'total_days_in_period': df['date'].nunique(),
        }),
        include_groups=False
    ).reset_index()
    
    # Calculate Estimated Profit: (Cash Flow from Sells + Value of Inventory) - Cash Flow for Buys
    broker_stats['inventory'] = broker_stats['total_buy'] - broker_stats['total_sell']
    broker_stats['est_profit'] = (broker_stats['total_sell_amt'] + broker_stats['inventory'] * last_price) - broker_stats['total_buy_amt']
    
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
    
    # [UPGRADED] Dynamic PCA to retain 95% variance
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Ensure at least 2 dimensions for KMeans logic consistency if needed
    if X_pca.shape[1] < 2:
        X_pca = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    
    print(f"PCA reduced dimensions to: {X_pca.shape[1]} (Explained Variance: {sum(pca.explained_variance_ratio_)*100:.1f}%)")
    
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