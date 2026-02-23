import pandas as pd
import glob
import os

# Configuration
STOCK_ID = '1536'  # Using the most active stock found: 1536 (和大?)
LOOKBACK_DAYS = 7 # Shortened for recent data analysis
DATA_DIR = 'data/'
PRICE_DB_PATH = 'data/stock_price_history.parquet'

def load_price_data(stock_id):
    """Loads OHLCV data for the specific stock."""
    if not os.path.exists(PRICE_DB_PATH):
        print("Price DB not found.")
        return pd.DataFrame()
    
    try:
        df = pd.read_parquet(PRICE_DB_PATH)
        df = df[df['stock_id'] == stock_id]
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

    print(f"Reading from {file_path}...")
    try:
        # Optimization: Use pyarrow filters to load only what we need if possible,
        # but for simplicity and since the file is 239MB, loading it is okay.
        # However, let's use the columns mapping from the Parquet file to our standard names.
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
    
    # Sort by date
    df = df.sort_values('date')
    dates = sorted(df['date'].unique())
    
    # Create a map for fast price lookup: date -> close_price
    price_map = {}
    if not price_df.empty:
        price_map = price_df.set_index('date')['close'].to_dict()
    
    # State tracking
    # broker_id -> {'qty': 0, 'avg_cost': 0, 'realized_pnl': 0}
    broker_state = {}
    
    results = []

    print(f"Simulating BPS for {STOCK_ID} over {len(dates)} days (Lookback={LOOKBACK_DAYS})...")

    for current_date in dates:
        daily_tx = df[df['date'] == current_date]
        
        # 1. Determine Current Price for Valuation
        # Prefer OHLCV close, fallback to weighted avg from transactions
        if current_date in price_map:
            current_price = price_map[current_date]
        else:
            total_vol = daily_tx['buy'].sum() + daily_tx['sell'].sum()
            total_amt = (daily_tx['price'] * (daily_tx['buy'] + daily_tx['sell'])).sum()
            current_price = total_amt / total_vol if total_vol > 0 else 0
        
        # 2. Calculate Daily Net Buy for all brokers (to identify winners' actions)
        daily_summary = daily_tx.groupby('securities_trader_id')[['buy', 'sell', 'price']].apply(
            lambda x: pd.Series({
                'net_buy_qty': x['buy'].sum() - x['sell'].sum(),
                # Approx avg price for this broker today
                'avg_price': (x['price'] * (x['buy'] + x['sell'])).sum() / (x['buy'] + x['sell']).sum() if (x['buy'] + x['sell']).sum() > 0 else 0
            })
        ).reset_index()
        
        # 3. Calculate Historical Performance (State BEFORE today's action)
        # Identify who WAS winning up to yesterday.
        
        broker_pnls = []
        for bid, state in broker_state.items():
            # Mark-to-Market PnL
            unrealized = (current_price - state['avg_cost']) * state['qty']
            total_pnl = state['realized_pnl'] + unrealized
            broker_pnls.append({'securities_trader_id': bid, 'total_pnl': total_pnl})
        
        df_pnl = pd.DataFrame(broker_pnls)
        
        top_winners_net_buy = 0
        top_winners = []
        
        if not df_pnl.empty:
            # Rank and Pick Top 5
            df_pnl = df_pnl.sort_values('total_pnl', ascending=False)
            top_5_ids = df_pnl.head(5)['securities_trader_id'].tolist()
            top_winners = top_5_ids
            
            # Calculate their action TODAY
            winners_today = daily_summary[daily_summary['securities_trader_id'].isin(top_5_ids)]
            top_winners_net_buy = winners_today['net_buy_qty'].sum()
            
        results.append({
            'date': current_date,
            'price': current_price,
            'bps_factor': top_winners_net_buy,
            'top_winners': top_winners
        })

        # 4. Update State for Tomorrow (incorporating today's trades)
        for idx, row in daily_summary.iterrows():
            bid = row['securities_trader_id']
            net_qty = row['net_buy_qty']
            avg_p = row['avg_price'] 
            
            if bid not in broker_state:
                broker_state[bid] = {'qty': 0, 'avg_cost': 0, 'realized_pnl': 0}
            
            st = broker_state[bid]
            
            # Update logic (Weighted Avg Cost)
            if net_qty > 0:
                # Buying: Update avg cost
                new_cost = (st['qty'] * st['avg_cost'] + net_qty * avg_p) / (st['qty'] + net_qty)
                st['qty'] += net_qty
                st['avg_cost'] = new_cost
            elif net_qty < 0:
                # Selling: Realize PnL
                sell_qty = abs(net_qty)
                if st['qty'] > 0:
                    # Profit = (Sell Price - Avg Cost) * Qty
                    profit = (avg_p - st['avg_cost']) * min(sell_qty, st['qty'])
                    st['realized_pnl'] += profit
                    st['qty'] = max(0, st['qty'] - sell_qty)
                else:
                    # Short selling (simplified: ignore realized pnl or treat as 0 cost)
                    pass
                
                # Decay Realized PnL?
                # The "Lookback" concept usually implies we only care about PnL made in the last N days.
                # Implementing a strict "Rolling Window PnL" is complex because we need to track PnL per day.
                # For this MVP, we accumulate PnL indefinitely but rely on the fact that 
                # we started from zero in Jan 2025, so it's naturally "recent".
                # To strictly follow "Lookback 60 days", we would need a queue of daily PnLs.
                # Given we only have ~30 days of data, simple accumulation is fine.

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