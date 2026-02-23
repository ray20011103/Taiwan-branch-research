import pandas as pd
import numpy as np

def analyze_insider_hypothesis():
    print("--- Insider Trading Hypothesis Verification ---")
    
    # Load trade report
    df = pd.read_csv('data/full_trade_report.csv')
    
    print(f"Total Trades Analyzed: {len(df)}")
    
    # 1. Growth Distribution
    print("\n[Revenue Growth of Selected Stocks]")
    pos_growth = df[df['growth_pct'] > 0]
    neg_growth = df[df['growth_pct'] <= 0]
    
    print(f"Positive Growth: {len(pos_growth)} ({len(pos_growth)/len(df)*100:.1f}%)")
    print(f"Negative Growth: {len(neg_growth)} ({len(neg_growth)/len(df)*100:.1f}%)")
    
    print(f"Avg Growth of Selected: {df['growth_pct'].mean():.2f}%")
    print(f"Median Growth of Selected: {df['growth_pct'].median():.2f}%")
    
    # 2. Correlation: Buy Strength vs Growth
    # Does 'Buying More' mean 'Better Revenue'?
    # We use 'value_mn' (Buy Value) as proxy for conviction
    correlation = df['value_mn'].corr(df['growth_pct'])
    print(f"\nCorrelation (Buy Value vs. Revenue Growth): {correlation:.4f}")
    
    # 3. Correlation: Growth vs Return
    # Does 'Better Revenue' actually lead to 'Higher Return'?
    impact_corr = df['growth_pct'].corr(df['return_pct'])
    print(f"Correlation (Revenue Growth vs. Price Return): {impact_corr:.4f}")
    
    # 4. The "Smart" Filter Test
    # Group by Signal Strength
    high_conviction = df[df['value_mn'] > 50] # > 50M NTD
    low_conviction = df[df['value_mn'] <= 50]
    
    print("\n[Conviction Test]")
    print(f"High Conviction (>50M) Avg Growth: {high_conviction['growth_pct'].mean():.2f}% (N={len(high_conviction)})")
    print(f"Low Conviction (<=50M) Avg Growth: {low_conviction['growth_pct'].mean():.2f}% (N={len(low_conviction)})")

    # 5. Conclusion
    print("\n[Preliminary Conclusion]")
    if len(pos_growth)/len(df) > 0.6:
        print(">> Strong Evidence: Smart Money predominantly picks growing companies.")
    else:
        print(">> Weak Evidence: Smart Money picks are not strictly tied to revenue growth.")
        
    if correlation > 0.3:
        print(">> Buying pressure scales with revenue quality.")

if __name__ == "__main__":
    analyze_insider_hypothesis()
