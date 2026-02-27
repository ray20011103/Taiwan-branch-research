import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def run_comprehensive_visualization():
    print("--- Generating Comprehensive Strategy Dashboard ---")
    
    # 1. Load Data
    trade_report_path = 'data/full_trade_report.csv'
    market_path = 'data/market_index.parquet'
    
    if not os.path.exists(trade_report_path):
        print("Trade report not found.")
        return
        
    trades = pd.read_csv(trade_report_path)
    trades['ann_date'] = pd.to_datetime(trades['ann_date'])
    
    # 2. Portfolio Simulation (5-Slot Logic)
    mkt = pd.read_parquet(market_path)
    calendar = sorted(pd.to_datetime(mkt['date']).unique())
    cal_series = pd.Series(range(len(calendar)), index=calendar)
    
    processed_trades = []
    for _, row in trades.iterrows():
        try:
            exit_date = row['ann_date']
            if "Stop Loss" in str(row['exit_reason']):
                date_str = str(row['exit_reason']).split('(')[1].split(')')[0]
                exit_date = pd.to_datetime(date_str)
            
            ann_idx = cal_series.get(row['ann_date'])
            if ann_idx is None: continue
            
            entry_idx = max(0, ann_idx - 5)
            entry_date = cal_series.index[entry_idx]
            
            processed_trades.append({
                'stock_id': row['stock_id'],
                'entry_date': entry_date,
                'exit_date': exit_date,
                'ret': row['return_pct'] / 100.0,
                'market_ret': row['market_ret'],
                'exit_reason': row['exit_reason'],
                'value_mn': row['value_mn']
            })
        except: continue
            
    df_trades = pd.DataFrame(processed_trades).sort_values('entry_date')
    
    # Portfolio Loop
    initial_capital = 1000000
    current_equity = initial_capital
    num_slots = 5
    slots = [None] * num_slots
    history = []
    
    sim_dates = sorted(list(df_trades['entry_date'].unique()) + list(df_trades['exit_date'].unique()))
    sim_calendar = [d for d in calendar if d >= sim_dates[0] and d <= sim_dates[-1]]
    
    for today in sim_calendar:
        for i in range(num_slots):
            if slots[i] and today >= slots[i]['exit_date']:
                current_equity += slots[i]['invested'] * slots[i]['ret']
                slots[i] = None
        
        todays_signals = df_trades[df_trades['entry_date'] == today]
        for _, sig in todays_signals.iterrows():
            for i in range(num_slots):
                if slots[i] is None:
                    slots[i] = {'exit_date': sig['exit_date'], 'invested': initial_capital/num_slots, 'ret': sig['ret']}
                    break
        history.append({'date': today, 'equity': current_equity})
    
    df_hist = pd.DataFrame(history)
    df_hist['equity_idx'] = df_hist['equity'] / initial_capital

    # 3. Plotting the 4-Grid Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    # A. Equity Curve (Professional 5-Slot)
    axes[0, 0].plot(df_hist['date'], df_hist['equity_idx'], color='navy', linewidth=2.5, label='Strategy (5-Slots)')
    axes[0, 0].axhline(1.0, color='red', linestyle='--', alpha=0.5)
    axes[0, 0].set_title('Realistic Equity Curve (Portfolio Simulation)', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Equity Multiplier')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # B. Individual Returns
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in df_trades['ret']]
    axes[0, 1].bar(range(len(df_trades)), df_trades['ret']*100, color=colors, alpha=0.8)
    axes[0, 1].set_title('Individual Trade Performance (%)', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Return %')
    axes[0, 1].axhline(0, color='black', linewidth=0.8)

    # C. Alpha Forensics (Scatter)
    sns.scatterplot(data=df_trades, x='market_ret', y=df_trades['ret']*100, size='value_mn', hue='exit_reason', palette='viridis', ax=axes[1, 0])
    axes[1, 0].plot([-20, 20], [-20, 20], ls="--", c=".3", label='Beta=1 Line')
    axes[1, 0].set_title('Alpha Forensics: Strategy vs Market', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Market Return %')
    axes[1, 0].set_ylabel('Stock Return %')
    axes[1, 0].legend(loc='lower right', fontsize='small')

    # D. Return Distribution
    sns.histplot(df_trades['ret']*100, bins=12, kde=True, ax=axes[1, 1], color='darkorchid')
    axes[1, 1].set_title('Strategy Return Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Return %')

    # Add Summary Text
    final_ret = (df_hist['equity_idx'].iloc[-1] - 1) * 100
    win_rate = (df_trades['ret'] > 0).mean() * 100
    profit_factor = (df_trades[df_trades['ret']>0]['ret'].sum()) / abs(df_trades[df_trades['ret']<0]['ret'].sum())
    
    summary_text = f"Summary:\nTotal Ret: {final_ret:.1f}%\nWin Rate: {win_rate:.1f}%\nProfit Factor: {profit_factor:.2f}"
    plt.gcf().text(0.02, 0.02, summary_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    output_path = 'docs/strategy_performance_visual.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Dashboard updated and saved to {output_path}")

if __name__ == "__main__":
    run_comprehensive_visualization()
