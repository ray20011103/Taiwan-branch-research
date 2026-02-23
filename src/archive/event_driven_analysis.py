import pandas as pd
import os
from smart_bps import run_smart_bps

# Configuration
STOCK_ID = '3706'
REVENUE_DB = 'data/revenue_announcements.parquet'
SMART_BPS_RESULT = f'data/smart_bps_result_{STOCK_ID}.csv'

def analyze_event_correlation(stock_id):
    print(f"\n--- Event-Driven Analysis for Stock: {stock_id} ---")
    
    # 1. Load Revenue Announcements
    if not os.path.exists(REVENUE_DB):
        print("Revenue DB not found.")
        return
    
    rev_df = pd.read_parquet(REVENUE_DB)
    rev_df = rev_df[rev_df['stock_id'] == stock_id].sort_values('announcement_date')
    
    if rev_df.empty:
        print(f"No revenue data found for {stock_id}.")
        return
    
    print("\nRecent Announcements:")
    print(rev_df.tail(5).to_string(index=False))

    # 2. Load Smart BPS Results
    # If not exists, run it first
    if not os.path.exists(SMART_BPS_RESULT):
        print("Smart BPS result not found. Running calculation...")
        run_smart_bps(stock_id)
        
    bps_df = pd.read_csv(SMART_BPS_RESULT)
    
    # 3. Correlation Analysis
    # For each announcement date, look at the BPS factor in the 5 days PRIOR
    for _, event in rev_df.iterrows():
        ann_date = event['announcement_date']
        growth = event['revenue_growth_pct']
        is_high = event['創新高/低(歷史)'] == 'H'
        
        # Get window: T-5 to T
        bps_window = bps_df[bps_df['date'] <= ann_date].tail(6) # Including the day of
        
        if not bps_window.empty:
            pre_ann_buy = bps_window['smart_bps'].iloc[:-1].sum() # T-5 to T-1
            day_of_buy = bps_window['smart_bps'].iloc[-1]
            
            print(f"\n📅 Announcement Date: {ann_date}")
            print(f"   Growth: {growth}% | New High: {'Yes' if is_high else 'No'}")
            print(f"   Smart BPS (5 days prior sum): {pre_ann_buy:,.0f}")
            print(f"   Smart BPS (Day of): {day_of_buy:,.0f}")
            
            if pre_ann_buy > 50000 and growth > 10:
                print("   🔥 SIGNAL: Front-running detected! Smart brokers bought heavily before positive news.")
            elif pre_ann_buy < -50000 and growth < 0:
                print("   ❄️ SIGNAL: Inside Exit detected! Smart brokers sold before negative news.")

if __name__ == "__main__":
    analyze_event_correlation(STOCK_ID)
