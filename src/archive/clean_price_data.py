import pandas as pd
import os

# Configuration
INPUT_CSV = 'data/price.csv'
OUTPUT_PARQUET = 'data/stock_price_history.parquet'

def clean_price_data():
    print(f"Reading {INPUT_CSV}...")
    
    # Reading in chunks to handle large file size
    chunk_size = 100000
    chunks = []
    
    # Define column mapping based on the header we saw
    # 證券代碼,年月日,開盤價(元),最高價(元),最低價(元),收盤價(元),成交值(千元),...
    use_cols = ['證券代碼', '年月日', '開盤價(元)', '最高價(元)', '最低價(元)', '收盤價(元)', '成交值(千元)']
    
    try:
        # Specify dtype to avoid mixed type warnings and errors
        dtype_spec = {'證券代碼': str, '年月日': str}
        
        for chunk in pd.read_csv(INPUT_CSV, usecols=use_cols, chunksize=chunk_size, encoding='utf-8', on_bad_lines='skip', dtype=dtype_spec):
            
            # 1. Parse Stock ID safely
            # "1101 台泥" -> "1101"
            # Handle potential NaN or non-string values
            chunk['stock_id'] = chunk['證券代碼'].astype(str).apply(lambda x: x.split()[0].strip() if pd.notnull(x) and isinstance(x, str) else '')
            
            # Remove empty stock_ids
            chunk = chunk[chunk['stock_id'] != 'nan']
            chunk = chunk[chunk['stock_id'] != '']
            
            # 2. Parse Date
            # 20150105 -> 2015-01-05
            chunk['date'] = pd.to_datetime(chunk['年月日'], format='%Y%m%d', errors='coerce').dt.strftime('%Y-%m-%d')
            
            # 3. Rename columns
            chunk = chunk.rename(columns={
                '開盤價(元)': 'open',
                '最高價(元)': 'high',
                '最低價(元)': 'low',
                '收盤價(元)': 'close',
                '成交值(千元)': 'volume_value_1k' # Keeping raw value for now
            })
            
            # 4. Clean numeric columns (remove commas if any, handle conversion)
            for col in ['open', 'high', 'low', 'close', 'volume_value_1k']:
                 chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
            
            # Drop rows with invalid dates or prices
            chunk = chunk.dropna(subset=['date', 'stock_id', 'close'])
            
            # Keep only necessary columns
            final_chunk = chunk[['date', 'stock_id', 'open', 'high', 'low', 'close', 'volume_value_1k']]
            chunks.append(final_chunk)
            
        print("Concatenating chunks...")
        df_all = pd.concat(chunks, ignore_index=True)
        
        # Sort
        df_all = df_all.sort_values(['stock_id', 'date'])
        
        print(f"Writing to {OUTPUT_PARQUET}...")
        df_all.to_parquet(OUTPUT_PARQUET, index=False)
        print("Done.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    clean_price_data()
