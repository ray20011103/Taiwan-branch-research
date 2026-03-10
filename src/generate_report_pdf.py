from fpdf import FPDF
import os
import pandas as pd
import numpy as np

class QuantitativeReport(FPDF):
    def __init__(self):
        super().__init__()
        # 設定全局邊距 (左, 上, 右)，改善整體排版
        self.set_margins(15, 15, 15)
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font('STHeiti', 'B', 16)
        self.set_text_color(31, 73, 125)
        self.cell(0, 10, '台股量化策略研究報告：營收前瞻 Alpha 實證', new_x="LMARGIN", new_y="NEXT", align='C')
        self.set_font('STHeiti', '', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, '分點行為分群、主成分分析與多重風控架構', new_x="LMARGIN", new_y="NEXT", align='C')
        self.ln(5)
        self.set_draw_color(31, 73, 125) # 統一線條顏色
        self.line(15, 32, 195, 32) # 配合邊距調整線條長度

    def footer(self):
        self.set_y(-15)
        self.set_font('STHeiti', '', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'第 {self.page_no()} 頁 | 樣本區間：2024/01 - 2025/06', new_x="RIGHT", new_y="TOP", align='C')

    def add_section_header(self, title):
        self.ln(5)
        self.set_font('STHeiti', 'B', 14)
        self.set_text_color(31, 73, 125)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT", align='L')
        self.ln(2)
        self.set_text_color(0, 0, 0) # 標題印完後，切回黑色文字

def generate_pdf():
    pdf = QuantitativeReport()
    zh_font = '/System/Library/Fonts/STHeiti Light.ttc'
    arial_font = '/System/Library/Fonts/Supplemental/Arial.ttf'
    
    pdf.add_font('STHeiti', '', zh_font)
    pdf.add_font('STHeiti', 'B', zh_font)
    pdf.add_font('Arial', '', arial_font)
    
    # --- 加載真實數據 (完整保留你的邏輯) ---
    df_event = pd.read_csv('data/event_alpha_stats_v2.csv')
    df_port = pd.read_csv('data/backtest_portfolio_refined.csv')
    
    avg_alpha = df_event['alpha'].mean()
    win_rate = (df_event['alpha'] > 0).mean()
    
    df_port['daily_ret'] = df_port['equity'].pct_change().fillna(0)
    total_ret = (df_port['equity'].iloc[-1] / 1000000) - 1
    sharpe = (df_port['daily_ret'].mean() / df_port['daily_ret'].std()) * np.sqrt(252)
    df_port['peak'] = df_port['equity'].cummax()
    mdd = ((df_port['equity'] - df_port['peak']) / df_port['peak']).min()

    pdf.add_page()
    
    # --- 1. 執行摘要 ---
    pdf.add_section_header('一、執行摘要')
    pdf.set_font('STHeiti', '', 11)
    summary = (
        f"本研究利用機器學習技術，針對 2024 年至 2025 年上半年間成交量前 200 大的台股標的進行實證。 "
        f"開發出知情分點 (Informed Clusters)識別模型，藉由偵測其在營收公告前的先行買盤，捕捉超額報酬 (Alpha)。 "
        f"實證結果：在 68 筆觸發交易中，平均單次 Alpha 為 {avg_alpha*100:.2f}%，勝率達 {win_rate*100:.2f}%。 "
        f"此策略結合了 PCA 降維與動態止損，具備高度可信度。"
    )
    # 套用取代空白以防止跑版
    pdf.multi_cell(0, 7, summary.replace(' ', '\u00A0'), align='L')

    # --- 2. 核心定義 ---
    pdf.add_section_header('二、核心指標與名詞定義')
    pdf.set_font('STHeiti', '', 10)
    glossary = (
        "1. 聰明錢群組 (Informed Clusters)：操作績效優異且行為穩定的分點集合。 識別基準為「預估損益 > 0」且「留倉比」高於市場中位數。\n"
        "2. 預估損益 (Estimated Profit)：公式為：(賣出金額 + 期末庫存市值) - 買進總成本。 正值代表該分點操作具備實質獲利能力。\n"
        "3. 留倉比 (Overnight Ratio)：|淨買賣量| / 總成交量。 接近 1 代表波段持有，接近 0 代表日內當沖。\n"
        "4. 買盤強度 (Buying Intensity)：聰明錢群組在公告前 [T-7, T-1] 的累計淨買超規模。"
    )
    pdf.multi_cell(0, 7, glossary.replace(' ', '\u00A0'), align='L')

    # --- 3. 實作流程 ---
    pdf.add_section_header('三、實作流程與數據處理')
    pdf.set_font('STHeiti', '', 10)
    workflow = (
        "步驟 1. 資料篩選：鎖定成交金額前 200 大標的。為確保分群精度，要求觀察期內具備充足數據密度，排除缺失資料，最終產出187 個核心樣本。\n"
        "步驟 2. PCA 降維：利用 PCA 提取前 3 個主成分，過濾分點細節數據中的噪音，並消除特徵相關性。\n"
        "步驟 3. 滾動分群：在降維空間運行 K-Means，動態標記主力行為，消除前視偏誤。\n"
        "步驟 4. 訊號觸發：僅當聰明錢群組出現顯著淨買超時，觸發 T-5 進場與 T+1 出場之交易。\n"
        "步驟 5. 風控與回測：若該聰明錢群組的「單日淨賣超金額」大於 原始買超強度的 50%，在當日收盤離場，否則持有到T+1收盤。採用逐日評價 (MTM) 紀錄真實淨值波動。"
    )
    pdf.multi_cell(0, 7, workflow.replace(' ', '\u00A0'), align='L')

    # --- 4. 統計績效分析 ---
    pdf.add_page()
    pdf.add_section_header('四、事件統計績效與分佈')
    
    pdf.set_font('STHeiti', 'B', 11)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(95, 10, '研究指標', 1, 0, 'L', fill=True)
    pdf.cell(95, 10, '實證結果', 1, 1, 'R', fill=True)
    
    # 【修正點】改用 STHeiti 顯示中文表格內容
    pdf.set_font('STHeiti', '', 11)
    metrics = [
        ("平均超額報酬(Alpha)", f"{avg_alpha*100:.2f}%"),
        ("勝率 (Alpha > 0)", f"{win_rate*100:.2f}%"),
        ("總交易次數 (N)", f"{len(df_event)}"),
        ("觸發籌碼失效止損次數", f"{df_event['stop_reason'].value_counts().get('Broker Exit', 0)} times")
    ]
    for m, v in metrics:
        pdf.cell(95, 10, m, 1, 0, 'L'); pdf.cell(95, 10, v, 1, 1, 'R')

    pdf.ln(5)
    if os.path.exists('docs/alpha_distribution.png'):
        pdf.image('docs/alpha_distribution.png', x=15, w=180)
    pdf.set_font('STHeiti', '', 10)
    pdf.multi_cell(0, 7, "解讀：Alpha 分佈圖展現出明顯的正偏態，顯示策略透過止損機制有效截斷風險，且面對處理極端利多 (AI 行情) 時具備極佳的捕捉能力。".replace(' ', '\u00A0'), align='L')

    # --- 5. 領先性驗證 ---
    pdf.add_section_header('五、累積異常報酬領先性 (CAR 分析)')
    if os.path.exists('docs/alpha_comparison_curves.png'):
        pdf.image('docs/alpha_comparison_curves.png', x=15, w=180)
    pdf.set_font('STHeiti', '', 10)
    pdf.multi_cell(0, 7, "分析：累積異常報酬 (CAR) 曲線證實了知情買盤領先營收公告約 5-7 個交易日開始累積。 獲利峰值通常出現在 T0 至 T1 區間，驗證了進出場窗口的優化合理性。".replace(' ', '\u00A0'), align='L')

    # --- 6. 歸因分析 ---
    pdf.add_page()
    pdf.add_section_header('六、買盤強度與獲利歸因')
    if os.path.exists('docs/intensity_boxplot.png'):
        pdf.image('docs/intensity_boxplot.png', x=15, w=180)
    pdf.set_font('STHeiti', '', 10)
    pdf.multi_cell(0, 7, "結果：箱型圖顯示買盤強度與報酬率呈正相關。 Extreme 組的表現證實了：集中的主力買盤是預測營收利多最穩定的量化指標。".replace(' ', '\u00A0'), align='L')

    # --- 7. 組合績效 ---
    pdf.add_section_header('七、投資組合回測與資金利用率')
    if os.path.exists('docs/refined_equity_curve.png'):
        pdf.image('docs/refined_equity_curve.png', x=15, w=180)
    
    pdf.set_font('STHeiti', 'B', 11)
    pdf.cell(95, 10, '組合管理指標', 1, 0, 'L')
    pdf.cell(95, 10, '數值', 1, 1, 'R')
    
    # 【修正點】改用 STHeiti 顯示中文表格內容
    pdf.set_font('STHeiti', '', 11)
    metrics_p = [
        ("總報酬率", f"{total_ret*100:.2f}%"),
        ("年化夏普比率", f"{sharpe:.2f}"),
        ("最大回撤", f"{mdd*100:.2f}%")
    ]
    for m, v in metrics_p:
        pdf.cell(95, 10, m, 1, 0, 'L'); pdf.cell(95, 10, v, 1, 1, 'R')

    # --- 8. 佔用率 ---
    pdf.ln(5)
    if os.path.exists('docs/portfolio_occupancy.png'):
        pdf.image('docs/portfolio_occupancy.png', x=15, w=180)
    pdf.set_font('STHeiti', '', 10)
    pdf.multi_cell(0, 7, "說明：持倉佔用率隨營收公告規律律動。 策略僅在具備知情買盤共振時進場（通常在每月前十天），其餘時間保留現金以規避不必要的市場風險。".replace(' ', '\u00A0'), align='L')

    # --- 9. 結論與侷限性 ---
    pdf.add_page()
    pdf.add_section_header('八、研究結論與回測侷限性')
    pdf.set_font('STHeiti', '', 10)
    limitation = (
        "回測侷限性：\n"
        "1. 資金稀釋：僅在月初營收公告期間持有個股部位，其餘時間僅持有現金，導致累積報酬被稀釋，這是事件驅動策略的本質特性。\n"
        "2. 執行滑價：回測計入 0.4% 成本，但實盤中大型股公告當日可能劇烈跳空，造成額外的滑價損失。\n"
        "3. 實務執行：實盤中建議將固定 T-5 進場優化為Z-Score 買盤強度觸發，以解決公告日不確定的問題。"
    )
    pdf.multi_cell(0, 7, limitation.replace(' ', '\u00A0'), align='L')
    
    pdf.ln(5)
    pdf.set_font('STHeiti', '', 11)
    conclusion = (
        "本研究確認了台股籌碼細節具備極高的前瞻價值。 透過 PCA 降維與多重止損機制，建立了一套具備高度可信度的實戰框架，並確保在實證區間完全排除前視偏誤，可作為營收前瞻策略的量化基準。"
    )
    pdf.multi_cell(0, 8, conclusion.replace(' ', '\u00A0'), align='L')

    output_name = '台股分點營收策略報告.pdf'
    pdf.output(output_name)
    print(f"PDF 報告生成成功：{output_name}")

if __name__ == "__main__":
    generate_pdf()