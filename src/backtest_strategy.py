import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def run_realistic_backtest():
    print("--- Starting Conservative Backtest (Equal Weight, No Compounding) ---")
    
    # 1. Load Data
    prices = pd.read_parquet('data/stock_price_history.parquet')
    prices['date'] = pd.to_datetime(prices['date'])
    market = pd.read_parquet('data/market_index.parquet')
    market['date'] = pd.to_datetime(market['date'])
    market = market[['date', 'market_ret']].rename(columns={'market_ret': 'mkt_ret'})
    
    # Load Events
    events = pd.read_csv('data/event_intensity_analysis.csv')
    events['ann_date'] = pd.to_datetime(events['ann_date'])
    events['stock_id'] = events['stock_id'].astype(str)
    
    # Create a daily timeline for the backtest period
    start_date = '2024-01-01'
    end_date = '2025-06-30'
    all_dates = pd.to_datetime(market[(market['date'] >= start_date) & (market['date'] <= end_date)]['date'].unique())
    all_dates = sorted(all_dates)
    
    # 2. Map Events to Daily Returns
    # We will pre-calculate the daily returns for each trade's holding period
    trade_daily_returns = []
    
    for _, event in events.iterrows():
        stock_id = event['stock_id']
        ann_date = event['ann_date']
        
        stock_df = prices[prices['stock_id'] == stock_id].sort_values('date').reset_index(drop=True)
        if stock_df.empty: continue
        
        t0_matches = stock_df[stock_df['date'] <= ann_date]
        if t0_matches.empty: continue
        t0_idx = t0_matches.index[-1]
        
        # T-5 to T+1
        entry_idx = t0_idx - 5
        exit_idx = t0_idx + 1
        if entry_idx < 0 or exit_idx >= len(stock_df): continue
        
        trade_period = stock_df.iloc[entry_idx : exit_idx + 1].copy()
        trade_period['daily_ret'] = trade_period['close'].pct_change()
        
        # The first day (T-5) has no return because we buy at close
        # Returns start from T-4
        for i in range(1, len(trade_period)):
            trade_daily_returns.append({
                'date': trade_period.iloc[i]['date'],
                'stock_id': stock_id,
                'ret': trade_period.iloc[i]['daily_ret'],
                'intensity': event['c3_intensity']
            })
            
    trade_df = pd.DataFrame(trade_daily_returns)
    
    # 3. Portfolio Simulation (Daily)
    MAX_SLOTS = 10 # Assume we divide capital into 10 slots
    portfolio_history = []
    current_simple_return = 0.0
    
    for d in all_dates:
        # Get active trades on this day
        daily_trades = trade_df[trade_df['date'] == d]
        
        if daily_trades.empty:
            daily_port_ret = 0
            num_active = 0
        else:
            # If more than MAX_SLOTS, pick those with highest intensity (or just top N)
            active_today = daily_trades.sort_values('intensity', ascending=False).head(MAX_SLOTS)
            num_active = len(active_today)
            # Daily return = Sum of (Trade Ret * 1/MAX_SLOTS)
            # This is "No Compounding" + "Equal Weight with Fixed Slots"
            daily_port_ret = active_today['ret'].sum() / MAX_SLOTS
            
        current_simple_return += daily_port_ret
        
        # Market Benchark (Simple sum of daily mkt_ret)
        mkt_today = market[market['date'] == d]['mkt_ret'].values
        mkt_ret = mkt_today[0] if len(mkt_today) > 0 else 0
        
        portfolio_history.append({
            'date': d,
            'daily_ret': daily_port_ret,
            'cum_simple_ret': current_simple_return,
            'num_trades': num_active,
            'mkt_ret': mkt_ret
        })
        
    port_df = pd.DataFrame(portfolio_history)
    port_df['cum_mkt_simple'] = port_df['mkt_ret'].cumsum()
    
    # 4. Metrics
    total_ret = port_df['cum_simple_ret'].iloc[-1]
    annual_ret = total_ret / 1.5 # 1.5 years
    vol = port_df['daily_ret'].std() * np.sqrt(252)
    sharpe = annual_ret / vol if vol > 0 else 0
    
    # Calculate MDD (on Simple Return curve)
    # For simple return, MDD is (Peak - Current Value)
    peak = port_df['cum_simple_ret'].cummax()
    drawdown = peak - port_df['cum_simple_ret']
    mdd = drawdown.max()
    
    # Win Rate from Trade Log (pre-calculated or derived)
    # Let's derive it from the individual trade returns calculated earlier
    # We need to re-access the individual trade returns for win rate
    
    print(f"\nCONSERVATIVE Backtest Results (10 Slots, No Compounding):")
    print(f"Total Simple Return: {total_ret*100:.2f}%")
    print(f"Annualized Return: {annual_ret*100:.2f}%")
    print(f"Annualized Volatility: {vol*100:.2f}%")
    print(f"Max Drawdown (MDD): {mdd*100:.2f}%")
    print(f"Simple Sharpe Ratio: {sharpe:.2f}")
    
    # 5. Plot
    plt.figure(figsize=(12, 6))
    plt.plot(port_df['date'], port_df['cum_simple_ret'] * 100, label='Strategy (Conservative)', lw=2, color='blue')
    plt.plot(port_df['date'], port_df['cum_mkt_simple'] * 100, label='Market (Simple Sum)', lw=2, color='gray', alpha=0.6)
    plt.title('Conservative Equity Curve: 10-Slot Equal Weight, No Compounding', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Simple Return (%)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_img = 'docs/realistic_portfolio_curve.png'
    plt.savefig(output_img)
    print(f"\nUpdated conservative curve saved to {output_img}")
    
    # Save log
    port_df.to_csv('data/conservative_backtest_log.csv', index=False)

if __name__ == "__main__":
    run_realistic_backtest()
