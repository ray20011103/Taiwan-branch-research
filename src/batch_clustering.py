import pandas as pd
import glob
import os
from tqdm import tqdm
from broker_clustering import extract_features, perform_clustering

# Configuration
DATA_DIR = 'data/'
# Original + New Top 10 Active Stocks
TARGET_STOCKS = [
    '1536', '3645', '3450', '6558', '3706', '4931',
    '3013', '2365', '1815', '8096', '2408', '1514', '6215', '2486', '4510', '6140'
]

def load_data(stock_id):
    """Loads transactions from consolidated StockBranch.parquet and filters for the specific stock."""
    file_path = os.path.join(DATA_DIR, 'StockBranch.parquet')
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return pd.DataFrame()

    try:
        # Use pyarrow filters
        df = pd.read_parquet(file_path, filters=[('CommodityId', '==', str(stock_id))])
        
        if df.empty:
            return pd.DataFrame()

        # Standardize column names
        df = df.rename(columns={
            'Date': 'date',
            'CommodityId': 'stock_id',
            'SecuritiesTraderId': 'securities_trader_id',
            'Price': 'price',
            'Buy': 'buy',
            'Sell': 'sell'
        })
        
        # Ensure date is string
        df['date'] = df['date'].astype(str)
        
        return df
    except Exception as e:
        print(f"Error loading StockBranch.parquet for {stock_id}: {e}")
        return pd.DataFrame()

def identify_accumulator_cluster(df_clustered):
    """
    Identifies which cluster is likely the 'Accumulator/Smart Money'.
    Criteria: High Overnight Ratio + High Frequency (optional) + Moderate/High Volume
    """
    if df_clustered is None or 'cluster' not in df_clustered.columns:
        return None, None
        
    summary = df_clustered.groupby('cluster')[['overnight_ratio', 'frequency', 'log_avg_daily_vol']].mean()
    summary['count'] = df_clustered['cluster'].value_counts()
    
    # Simple Heuristic: The cluster with the highest Overnight Ratio that has at least 3 members
    candidates = summary[summary['count'] >= 3]
    if candidates.empty:
        best_cluster = summary['overnight_ratio'].idxmax()
    else:
        best_cluster = candidates['overnight_ratio'].idxmax()
        
    return best_cluster, summary

def run_batch_clustering():
    print(f"Starting Batch Clustering for {len(TARGET_STOCKS)} stocks...")
    
    results = []
    
    # Using tqdm for progress tracking
    for stock in tqdm(TARGET_STOCKS, desc="Analyzing Stocks"):
        df = load_data(stock)
        
        if df.empty:
            continue
            
        features = extract_features(df)
        clustered = perform_clustering(features)
        
        best_cluster_id, summary = identify_accumulator_cluster(clustered)
        
        if best_cluster_id is not None:
            # Save results
            output_file = f'data/broker_clusters_{stock}.csv'
            clustered.to_csv(output_file, index=False)
            
            # Record Stats
            best_stats = summary.loc[best_cluster_id]
            results.append({
                'stock_id': stock,
                'accumulator_cluster': best_cluster_id,
                'num_brokers': best_stats['count'],
                'avg_overnight_ratio': best_stats['overnight_ratio'],
                'avg_frequency': best_stats['frequency']
            })
            
    if results:
        print("\n\n" + "="*40)
        print("BATCH CLUSTERING SUMMARY")
        print("="*40)
        print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    run_batch_clustering()
