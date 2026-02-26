import pandas as pd
import os
import glob
from scipy import stats
import numpy as np

def run_combined_analysis():
    print("--- 🧠 Combined Strategy: Smart BPS + Warrant Hedging (Significance Test) ---")
    
    # 1. Load Data
    warrant_report_path = 'data/collective_warrant_impact_report.csv'
    market_index_path = 'data/market_index.parquet'
    
    if not os.path.exists(warrant_report_path) or not os.path.exists(market_index_path):
        print("Required data files not found.")
        return
        
    w_df = pd.read_csv(warrant_report_path)
    w_df['date'] = pd.to_datetime(w_df['date'])
    w_df['stock_id'] = w_df['stock_id'].astype(str)
    
    # Load Market Data for Alpha calculation
    mkt_df = pd.read_parquet(market_index_path)
    mkt_df['date'] = pd.to_datetime(mkt_df['date'])
    mkt_df = mkt_df[['date', 'market_ret']].rename(columns={'market_ret': 'mkt_ret'})
    
    stocks = w_df['stock_id'].unique()
    bps_data_list = []
    
    for sid in stocks:
        bps_path = f'data/smart_bps_result_{sid}.csv'
        if os.path.exists(bps_path):
            tmp = pd.read_csv(bps_path)
            tmp['date'] = pd.to_datetime(tmp['date'])
            tmp['stock_id'] = str(sid)
            if 'bps_factor' in tmp.columns:
                tmp = tmp.rename(columns={'bps_factor': 'smart_bps'})
            tmp = tmp.drop_duplicates('date')
            bps_data_list.append(tmp[['date', 'stock_id', 'smart_bps']])
    
    if not bps_data_list:
        print("No Smart BPS data found.")
        return
        
    bps_all = pd.concat(bps_data_list)
    
    # Merge Warrant + BPS
    combined = pd.merge(w_df, bps_all, on=['date', 'stock_id'], how='inner')
    # Merge Market Index
    combined = pd.merge(combined, mkt_df, on='date', how='left')
    
    # 2. Define Signals & Target (ALPHA BASED)
    combined['alpha_ret'] = combined['stock_ret'] - combined['mkt_ret']
    
    combined['is_smart_buy'] = combined['smart_bps'] > 0
    combined['is_hedge_push'] = combined['hedge_pressure_pct'] > 0
    
    combined = combined.sort_values(['stock_id', 'date'])
    # Predicting NEXT DAY Alpha
    combined['next_alpha'] = combined.groupby('stock_id')['alpha_ret'].shift(-1)
    combined = combined.dropna(subset=['next_alpha'])
    
    # 3. Statistical Groups
    group_both = combined[combined['is_smart_buy'] & combined['is_hedge_push']]['next_alpha']
    group_none = combined[(~combined['is_smart_buy']) & (~combined['is_hedge_push'])]['next_alpha']
    group_hedge_only = combined[(~combined['is_smart_buy']) & combined['is_hedge_push']]['next_alpha']
    group_smart_only = combined[combined['is_smart_buy'] & (~combined['is_hedge_push'])]['next_alpha']

    # 4. Performance Attribution
    def get_stats(series, label):
        mean = series.mean()
        std = series.std()
        return {
            'Signal': label,
            'Count': len(series),
            'Mean_Alpha%': mean * 100,
            'Std_Dev%': std * 100,
            'Win_Rate%': (series > 0).mean() * 100,
            'Alpha_Sharpe': (mean / std * np.sqrt(252)) if (not pd.isna(std) and std != 0) else 0
        }

    results = [
        get_stats(group_none, 'None'),
        get_stats(group_smart_only, 'Smart Only'),
        get_stats(group_hedge_only, 'Hedge Only'),
        get_stats(group_both, 'Both (Synergy)')
    ]
    
    print("\n--- 📊 Performance Summary (Market Alpha Based) ---")
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))
    
    # 5. T-Tests for Significance
    print("\n--- ⚖️ Statistical Significance Tests (P-Values) ---")
    if len(group_both) > 5 and len(group_none) > 5:
        t_stat, p_val = stats.ttest_ind(group_both, group_none, equal_var=False)
        print(f"1. Both vs None (Alpha):    P-Value = {p_val:.4f} {'(SIGNIFICANT)' if p_val < 0.05 else '(Not Significant)'}")
        
        t_stat, p_val_sh = stats.ttest_ind(group_both, group_smart_only, equal_var=False)
        print(f"2. Both vs Smart Only:      P-Value = {p_val_sh:.4f} {'(SIGNIFICANT)' if p_val_sh < 0.05 else '(Not Significant)'}")
    else:
        print("Sample size still too low for reliable T-Test.")

    # 6. Conclusion
    print("\n--- 📝 Final Verdict ---")
    if not summary_df.empty:
        best_signal = summary_df.loc[summary_df['Alpha_Sharpe'].idxmax(), 'Signal']
        print(f"Best Risk-Adjusted Signal: {best_signal}")
    
    combined.to_csv('data/combined_bps_warrant_analysis.csv', index=False)
    print("\nFull analysis saved to data/combined_bps_warrant_analysis.csv")

if __name__ == "__main__":
    run_combined_analysis()
