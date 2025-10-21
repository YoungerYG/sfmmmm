#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime
import numpy as np
import pandas as pd
import talib
import yfinance as yf


def fetch_ohlcv(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    下载指定股票的OHLCV数据（开盘价、最高价、最低价、收盘价和成交量）。
    """
    df = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        group_by="ticker",
        progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"下载数据为空：{ticker}")
    return df


def pick_series_from_df(df: pd.DataFrame, ticker: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """自动识别多层列方向"""
    if isinstance(df.columns, pd.MultiIndex):
        level_names = list(df.columns.names)
        print("DEBUG MultiIndex levels:", level_names)

        if level_names == ["Price", "Ticker"] or "Price" in level_names[0]:
            close_s = df["Close"][ticker].dropna()
            high_s = df["High"][ticker].dropna()
            low_s = df["Low"][ticker].dropna()
        elif level_names == ["Ticker", "Price"] or "Ticker" in level_names[0]:
            close_s = df[ticker]["Close"].dropna()
            high_s = df[ticker]["High"].dropna()
            low_s = df[ticker]["Low"].dropna()
        else:
            raise ValueError(f"无法识别 MultiIndex 层级结构: {df.columns.names}")
    else:
        close_s = df["Close"].dropna()
        high_s = df["High"].dropna()
        low_s = df["Low"].dropna()

    return close_s.astype(float), high_s.astype(float), low_s.astype(float)


def compute_indicators(close_np: np.ndarray, high_np: np.ndarray, low_np: np.ndarray) -> dict:
    """
    计算常见的技术指标，包括 RSI、随机指标（Stochastic）、Williams %R 和 200日均线。
    """
    rsi = talib.RSI(close_np, timeperiod=14)
    slowk, slowd = talib.STOCH(high_np, low_np, close_np, fastk_period=9, slowk_period=3, slowd_period=3)
    williams = talib.WILLR(high_np, low_np, close_np, timeperiod=14)
    sma_200 = talib.SMA(close_np, timeperiod=200)
    return {"rsi": rsi, "slowk": slowk, "slowd": slowd, "williams": williams, "sma_200": sma_200}


def interpret_rsi(rsi: float) -> str:
    """
    解读 RSI 指标：超买区、超卖区、偏强区、偏弱区。
    """
    if rsi > 70:
        return "超买区（市场偏热，可能面临短期回调风险）"
    elif rsi < 30:
        return "超卖区（市场偏冷，存在反弹潜力）"
    elif 50 <= rsi <= 70:
        return "偏强区（上升趋势中，市场情绪乐观）"
    elif 30 < rsi < 50:
        return "偏弱区（下降趋势中，情绪谨慎）"
    return "中性"


def interpret_stoch(k: float, d: float) -> str:
    """
    解读随机指标 Stochastic：超买区、超卖区、K线与D线关系。
    """
    if k > 80 and d > 80:
        return "超买区（短线过热）"
    elif k < 20 and d < 20:
        return "超卖区（短线过冷）"
    elif k > d:
        return "K线上穿D线（短线反弹信号）"
    elif k < d:
        return "K线下穿D线（短线走弱信号）"
    return "中性"


def interpret_williams(wr: float) -> str:
    """
    解读 Williams %R 指标：超买区、超卖区。
    """
    if wr > -20:
        return "超买区（市场情绪过热）"
    elif wr < -80:
        return "超卖区（市场情绪悲观）"
    return "中性区（震荡或平衡）"


def interpret_deviation(dev: float) -> str:
    """
    解读股价相对于 200 日均线的乖离：是否过热或过冷。
    """
    if np.isnan(dev):
        return "数据不足"
    elif dev > 10:
        return "股价高于200日均线超过10%，趋势强但短线可能拉伸过度"
    elif dev < -10:
        return "股价低于200日均线超过10%，趋势偏弱或可能超跌"
    else:
        return "接近长期均衡状态"


def score_market(last_vals: dict, deviation_pct: float) -> tuple[int, list[str]]:
    """
    根据各项技术指标计算市场情绪得分，并返回详细解读。
    """
    score = 0
    notes = []

    rsi_last = float(last_vals["rsi"])
    k_last = float(last_vals["slowk"])
    d_last = float(last_vals["slowd"])
    wr_last = float(last_vals["williams"])

    # RSI
    if rsi_last > 70:
        score += 1
        notes.append(f"RSI {rsi_last:.2f} > 70（偏热 +1）")
    elif rsi_last < 30:
        score -= 1
        notes.append(f"RSI {rsi_last:.2f} < 30（偏冷 -1）")

    # Stochastic
    if k_last > 80 and d_last > 80:
        score += 1
        notes.append(f"Stoch K/D {k_last:.2f}/{d_last:.2f} > 80（偏热 +1）")
    elif k_last < 20 and d_last < 20:
        score -= 1
        notes.append(f"Stoch K/D {k_last:.2f}/{d_last:.2f} < 20（偏冷 -1）")

    # Williams %R
    if wr_last > -20:
        score += 1
        notes.append(f"Williams %R {wr_last:.2f} > -20（超买/偏热 +1）")
    elif wr_last < -80:
        score -= 1
        notes.append(f"Williams %R {wr_last:.2f} < -80（超卖/偏冷 -1）")

    # 200日均线乖离
    if not np.isnan(deviation_pct):
        if deviation_pct > 10:
            score += 1
            notes.append(f"对200日乖离 {deviation_pct:.2f}% > 10%（拉伸/偏热 +1）")
        elif deviation_pct < -10:
            score -= 1
            notes.append(f"对200日乖离 {deviation_pct:.2f}% < -10%（偏冷 -1）")
    else:
        notes.append("SMA200 数据不足，未计入评分")

    return score, notes


def interpret_score(score: int) -> str:
    """
    根据市场情绪得分返回对应的解读。
    """
    if score <= -2:
        return f"当前分数: {score} 📉 市场明显偏冷，短期情绪低迷或恐慌，存在超卖反弹机会。"
    elif -1 <= score <= 1:
        return f"当前分数: {score} 😐 市场中性，技术面平衡或震荡，适合观望或轻仓操作。"
    elif 2 <= score <= 3:
        return f"当前分数: {score} ⚠️ 市场偏热，短线过度乐观，需谨慎追高或考虑获利了结。"
    elif score >= 4:
        return f"当前分数: {score} 🔥 市场极度过热，存在显著回调风险，防范情绪化行情。"
    else:
        return f"当前分数: {score} ❓ 无法判断。"


def main():
    parser = argparse.ArgumentParser(description="基于常见技术指标给市场打分并输出解读")
    parser.add_argument("--ticker", type=str, default="^IXIC", help="标的代码，默认 ^IXIC，可选 QQQ 等")
    parser.add_argument("--years", type=float, default=1.0, help="向前取多少年历史，默认 1 年")
    args = parser.parse_args()

    end_date = datetime.datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.datetime.today() - pd.DateOffset(years=args.years)).strftime("%Y-%m-%d")
    print(f"[INFO] Ticker={args.ticker}, Start={start_date}, End={end_date}")

    df = fetch_ohlcv(args.ticker, start_date, end_date)
    close_s, high_s, low_s = pick_series_from_df(df, args.ticker)

    close_np = close_s.to_numpy().astype(float)
    high_np = high_s.to_numpy().astype(float)
    low_np = low_s.to_numpy().astype(float)

    ind = compute_indicators(close_np, high_np, low_np)

    last_close = float(close_s.iloc[-1])
    sma_200 = ind["sma_200"]
    last_sma_200 = float(sma_200[~np.isnan(sma_200)][-1]) if np.any(~np.isnan(sma_200)) else np.nan
    deviation_pct = (last_close - last_sma_200) / last_sma_200 * 100.0 if not np.isnan(last_sma_200) else np.nan

    rsi_last = float(ind["rsi"][-1])
    k_last = float(ind["slowk"][-1])
    d_last = float(ind["slowd"][-1])
    wr_last = float(ind["williams"][-1])

    print("\n=== 指标当前值与解读 ===")
    print(f"RSI: {rsi_last:.2f} → {interpret_rsi(rsi_last)}")
    print(f"随机指标 Stoch K/D: {k_last:.2f}/{d_last:.2f} → {interpret_stoch(k_last, d_last)}")
    print(f"Williams %R: {wr_last:.2f} → {interpret_williams(wr_last)}")
    print(f"200日均线乖离: {deviation_pct:.2f}% → {interpret_deviation(deviation_pct)}")

    last_vals = {"rsi": rsi_last, "slowk": k_last, "slowd": d_last, "williams": wr_last}
    score, notes = score_market(last_vals, deviation_pct)

    print("\n=== 评分细节 ===")
    for n in notes:
        print(" -", n)

    print("\n=== 最终市场热度分数 ===")
    print(f"当前分数: {score}")
    print(interpret_score(score))
    
    indicators = f"""
    ✅ ^IXIC
    === 指标当前值与解读 ===
    RSI: {rsi_last:.2f} → {interpret_rsi(rsi_last)}
    随机指标 Stoch K/D: {k_last:.2f}/{d_last:.2f} → {interpret_stoch(k_last, d_last)}
    Williams %R: {wr_last:.2f} → {interpret_williams(wr_last)}
    200日均线乖离: {deviation_pct:.2f}% → {interpret_deviation(deviation_pct)}
    """

    return interpret_score(score), indicators


if __name__ == "__main__":
    main()

