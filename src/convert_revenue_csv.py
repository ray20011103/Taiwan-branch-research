import pandas as pd
import os

# Define file paths
CSV_PATH = 'data/announcement.csv'
PARQUET_PATH = 'data/revenue_announcements.parquet'

def process_revenue_data():
    print(f"Reading from {CSV_PATH}...")
    
    try:
        # Read the CSV file
        df = pd.read_csv(CSV_PATH)
        
        # Rename columns to standard English names
        # 公司, 年月, 營收發布日, 單月營收成長率％, 創新高/低(歷史), 創新高/低(近一年)
        # Note: '公司' column contains both ID and Name, e.g., "1101 台泥"
        df = df.rename(columns={
            '年月': 'report_month',
            '營收發布日': 'announcement_date',
            '單月營收成長率％': 'revenue_growth_pct'
        })
        
        # Extract stock_id from the '公司' column
        # Assuming format "1101 台泥", split by space and take the first part
        df['stock_id'] = df['公司'].apply(lambda x: str(x).split(' ')[0])
        
        # Convert date columns to datetime objects
        df['announcement_date'] = pd.to_datetime(df['announcement_date'], errors='coerce')
        
        # Filter for valid dates (e.g., non-NaT)
        df = df.dropna(subset=['announcement_date'])
        
        # Filter for 2024 and 2025 data
        # We want announcement_date in 2024 or 2025
        # Actually, let's keep all data, but ensure format is correct YYYY-MM-DD string for consistency with other scripts
        df = df[df['announcement_date'].dt.year.isin([2024, 2025])]
        
        # Format dates as strings YYYY-MM-DD
        df['announcement_date'] = df['announcement_date'].dt.strftime('%Y-%m-%d')
        
        # Select relevant columns
        cols_to_keep = ['stock_id', 'announcement_date', 'report_month', 'revenue_growth_pct', '創新高/低(歷史)', '創新高/低(近一年)']
        df = df[cols_to_keep]
        
        # Clean numeric columns if necessary
        # revenue_growth_pct might contain commas or other characters? 
        # CSV snippet shows: "-8.98", "4,366.63". Need to remove commas.
        if df['revenue_growth_pct'].dtype == 'object':
             df['revenue_growth_pct'] = df['revenue_growth_pct'].astype(str).str.replace(',', '').astype(float)
        
        print(f"Processed {len(df)} records for 2024-2025.")
        print(df.head())
        
        # Save to Parquet
        print(f"Saving to {PARQUET_PATH}...")
        df.to_parquet(PARQUET_PATH, index=False)
        print("Done.")
        
    except Exception as e:
        print(f"Error processing CSV: {e}")

if __name__ == "__main__":
    process_revenue_data()
