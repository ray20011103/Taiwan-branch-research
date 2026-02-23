import pandas as pd
import os

INPUT_CSV = 'data/announcement.csv'
OUTPUT_PARQUET = 'data/revenue_announcements.parquet'

def clean_announcement_data():
    print(f"Reading {INPUT_CSV}...")
    
    # Read CSV
    # Columns: 公司,年月,營收發布日,單月營收成長率％,創新高/低(歷史),創新高/低(近一年)
    try:
        df = pd.read_csv(INPUT_CSV, encoding='utf-8')
        
        # 1. Parse Stock ID
        df['stock_id'] = df['公司'].astype(str).apply(lambda x: x.split()[0].strip())
        
        # 2. Parse Dates (Handle formats like YYYY/M/D)
        df['announcement_date'] = pd.to_datetime(df['營收發布日'], errors='coerce').dt.strftime('%Y-%m-%d')
        df['report_month'] = pd.to_datetime(df['年月'], errors='coerce').dt.strftime('%Y-%m')
        
        # 3. Clean numeric column
        df['revenue_growth_pct'] = pd.to_numeric(df['單月營收成長率％'].astype(str).str.replace(',', ''), errors='coerce')
        
        # 4. Keep relevant columns
        df = df[['stock_id', 'announcement_date', 'report_month', 'revenue_growth_pct', '創新高/低(歷史)', '創新高/低(近一年)']]
        
        # Filter for 2025 data to match our transaction data
        df_2025 = df[df['announcement_date'].str.startswith('2025', na=False)]
        
        print(f"Found {len(df_2025)} announcements in 2025.")
        
        # Save
        df_2025.to_parquet(OUTPUT_PARQUET, index=False)
        print(f"Saved to {OUTPUT_PARQUET}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    clean_announcement_data()
