import pandas as pd
from FinMind.data import DataLoader
import sys

def main():
    print("--- FinMind 權證資料抓取工具 ---")

    # 1. 登入驗證
    # 改用 input 方便使用者貼上並確認內容
    api_token = input("請貼上您的 FinMind Token: ").strip()
    if not api_token:
        print("錯誤: Token 不能為空")
        return

    dl = DataLoader()
    try:
        dl.login_by_token(api_token)
        print("成功登入 FinMind")
    except Exception as e:
        print(f"登入失敗: {e}")
        return

    # 2. 搜尋標的所有權證
    underlying_id = input("請輸入標的股號 (預設 2330): ") or "2330"
    print(f"正在搜尋與 {underlying_id} 相關的權證列表...")

    try:
        # 獲取權證基本資訊
        warrant_info = dl.taiwan_warrant_info()
        # 篩選特定標的的權證
        related_warrants = warrant_info[warrant_info['underlying_id'] == underlying_id]

        if related_warrants.empty:
            print(f"未找到與 {underlying_id} 相關的權證資料")
            return

        print(f"找到 {len(related_warrants)} 檔權證，前 5 檔如下:")
        print(related_warrants[['warrant_id', 'warrant_name', 'exercise_price', 'extradition_date']].head())

    except Exception as e:
        print(f"獲取權證列表失敗: {e}")
        return

    # 3. 抓取指定權證的行情資料
    target_warrant = input("\n請輸入欲查詢的權證代號 (例如 030001): ")
    if not target_warrant:
        print("未輸入權證代號，程式結束")
        return

    start_date = input("請輸入開始日期 (YYYY-MM-DD，預設 2024-03-01): ") or "2024-03-01"

    print(f"正在抓取權證 {target_warrant} 從 {start_date} 起的行情資料...")

    try:
        # 獲取權證每日成交行情
        df_quotes = dl.taiwan_warrant_daily_trading_quotes(
            warrant_id=target_warrant,
            start_date=start_date
        )

        if df_quotes.empty:
            print("該時段無成交行情資料")
        else:
            print(f"\n--- {target_warrant} 行情資料 (前 5 筆) ---")
            # 選取關鍵欄位顯示
            cols = ['date', 'close', 'volume', 'buy_iv', 'sell_iv', 'delta']
            existing_cols = [c for c in cols if c in df_quotes.columns]
            print(df_quotes[existing_cols].head())

            # 4. 儲存檔案
            save_csv = input("\n是否儲存為 CSV 檔案? (y/n): ").lower()
            if save_csv == 'y':
                file_name = f"warrant_{target_warrant}_{start_date}.csv"
                df_quotes.to_csv(file_name, index=False)
                print(f"檔案已儲存: {file_name}")

    except Exception as e:
        print(f"抓取行情失敗: {e}")

if __name__ == "__main__":
    main()
