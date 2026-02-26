import pandas as pd
import glob
import os

# Configuration
LOOKBACK_DAYS = 60 # Adjusted to 60 days as per user request
DATA_DIR = 'data/'
PRICE_DB_PATH = 'data/stock_price_history.parquet'

def load_price_data(stock_id):
    """Loads OHLCV data for the specific stock."""
    if not os.path.exists(PRICE_DB_PATH):
        print("Price DB not found.")
        return pd.DataFrame()
    
    try:
        df = pd.read_parquet(PRICE_DB_PATH)
        df = df[df['stock_id'] == str(stock_id)]
        return df
    except Exception as e:
        print(f"Error loading price data: {e}")
        return pd.DataFrame()

def load_data(stock_id):
    """Loads transactions from consolidated StockBranch.parquet and filters for the specific stock."""
    file_path = os.path.join(DATA_DIR, 'StockBranch.parquet')
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return pd.DataFrame()

    print(f"Reading from {file_path} for {stock_id}...")
    try:
        df = pd.read_parquet(file_path, filters=[('CommodityId', '==', str(stock_id))])
        
        if df.empty:
            print(f"No data found for stock {stock_id} in StockBranch.parquet")
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
        
        # Ensure date is string and sorted
        df['date'] = df['date'].astype(str)
        df = df.sort_values(['date', 'securities_trader_id'])
        
        print(f"Loaded {len(df)} transactions for {stock_id}")
        return df
    except Exception as e:
        print(f"Error loading StockBranch.parquet: {e}")
        return pd.DataFrame()

def calculate_bps(df, price_df):
    """
    Calculates Broker Profitability Score (BPS).
    """
    if df.empty: return pd.DataFrame()
    
    # Sort by date
    df = df.sort_values('date')
    dates = sorted(df['date'].unique())
    stock_id = df['stock_id'].iloc[0]
    
    # Create a map for fast price lookup: date -> close_price
    price_map = {}
    if not price_df.empty:
        price_map = price_df.set_index('date')['close'].to_dict()
        # Convert index dates to string for matching if necessary
        price_map = {str(k).split('T')[0]: v for k, v in price_map.items()}
    
    # State tracking
    broker_state = {}
    results = []

    print(f"Simulating BPS for {stock_id} over {len(dates)} days...")

    for current_date in dates:
        daily_tx = df[df['date'] == current_date]
        
        # 1. Determine Current Price for Valuation
        curr_date_str = str(current_date).split(' ')[0]
        if curr_date_str in price_map:
            current_price = price_map[curr_date_str]
        else:
            total_vol = daily_tx['buy'].sum() + daily_tx['sell'].sum()
            total_amt = (daily_tx['price'] * (daily_tx['buy'] + daily_tx['sell'])).sum()
            current_price = total_amt / total_vol if total_vol > 0 else 0
        
        # 2. Calculate Daily Net Buy
        daily_summary = daily_tx.groupby('securities_trader_id')[['buy', 'sell', 'price']].apply(
            lambda x: pd.Series({
                'net_buy_qty': x['buy'].sum() - x['sell'].sum(),
                'avg_price': (x['price'] * (x['buy'] + x['sell'])).sum() / (x['buy'] + x['sell']).sum() if (x['buy'] + x['sell']).sum() > 0 else 0
            }),
            include_groups=False
        ).reset_index()
        
        # 3. Calculate Historical Performance (BEFORE today)
        broker_pnls = []
        for bid, state in broker_state.items():
            unrealized = (current_price - state['avg_cost']) * state['qty']
            total_pnl = state['realized_pnl'] + unrealized
            broker_pnls.append({'securities_trader_id': bid, 'total_pnl': total_pnl})
        
        df_pnl = pd.DataFrame(broker_pnls)
        top_winners_net_buy = 0
        
        if not df_pnl.empty:
            df_pnl = df_pnl.sort_values('total_pnl', ascending=False)
            top_5_ids = df_pnl.head(5)['securities_trader_id'].tolist()
            winners_today = daily_summary[daily_summary['securities_trader_id'].isin(top_5_ids)]
            top_winners_net_buy = winners_today['net_buy_qty'].sum()
            
        results.append({
            'date': current_date,
            'price': current_price,
            'bps_factor': top_winners_net_buy
        })

        # 4. Update State
        for idx, row in daily_summary.iterrows():
            bid = row['securities_trader_id']
            net_qty = row['net_buy_qty']
            avg_p = row['avg_price'] 
            
            if bid not in broker_state:
                broker_state[bid] = {'qty': 0, 'avg_cost': 0, 'realized_pnl': 0}
            
            st = broker_state[bid]
            if net_qty > 0:
                new_cost = (st['qty'] * st['avg_cost'] + net_qty * avg_p) / (st['qty'] + net_qty)
                st['qty'] += net_qty
                st['avg_cost'] = new_cost
            elif net_qty < 0:
                sell_qty = abs(net_qty)
                if st['qty'] > 0:
                    profit = (avg_p - st['avg_cost']) * min(sell_qty, st['qty'])
                    st['realized_pnl'] += profit
                    st['qty'] = max(0, st['qty'] - sell_qty)

    return pd.DataFrame(results)

    return pd.DataFrame(results)

if __name__ == "__main__":
    print("Loading price data...")
    price_df = load_price_data(STOCK_ID)
    print(f"Loaded {len(price_df)} price records.")

    print("Loading daily report data...")
    df = load_data(STOCK_ID)
    
    if df.empty:
        print("No transaction data found.")
    else:
        print(f"Data loaded: {df.shape}")
        bps_df = calculate_bps(df, price_df)
        print("\n--- BPS Analysis Result (Last 10 Days) ---")
        print(bps_df.tail(10).to_string(index=False))
        
        # Simple Signal Check
        # BPS > 0 -> Bullish (Winners are buying)
        # BPS < 0 -> Bearish (Winners are selling)
        print("\nSummary:")
        print(bps_df[['date', 'price', 'bps_factor']].tail(5))