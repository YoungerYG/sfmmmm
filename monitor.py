#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整稳定版 monitor.py（含邮件发送）
功能：
- 从 FRED 获取宏观数据
- GSCPI 自动回退到纽约联储 Excel（多引擎容错、自动选表）
- 自动修复 state.db 中无效日期
- 无新数据也执行一次计算并输出
- 推送到 Slack（可选 @ 提醒）与邮箱（SMTP）
"""

import os, sqlite3, io, textwrap, smtplib
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ========== 初始化环境变量 ==========
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY", "").strip()
#SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "").strip()
ALERT_MENTION = os.getenv("ALERT_MENTION", "").strip()  # 例：<!here> 或 @your-handle

# 邮件
SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "").strip()
SMTP_PASS = os.getenv("SMTP_PASS", "").strip()
EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USER).strip()
EMAIL_TO = os.getenv("EMAIL_TO", SMTP_USER).strip()

DB_PATH = "state.db"

# ========== 指标配置 ==========
SERIES = {
    "cpi_headline": "CPIAUCSL",
    "pce_core": "PCEPILFE",
    "unrate": "UNRATE",
    "ahe_prod": "CES0500000003",
    "real_gdp": "GDPC1",
    "wti": "DCOILWTICO",
    "gscpi": "GSCPI",  # fallback NY Fed
    "umich": "UMCSENT",
}

# ========== 数据提取函数 ==========
def fred_observations(series_id: str, limit=18) -> pd.DataFrame:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "series_id": series_id,
        "sort_order": "desc",
        "limit": limit,
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json().get("observations", [])
    if not data:
        return pd.DataFrame(columns=["date", "value"]).assign(value=np.nan)
    df = pd.DataFrame(data)[["date", "value"]]
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"].replace(".", np.nan), errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def get_gscpi_from_nyfed() -> pd.DataFrame:
    """自动读取纽约联储 GSCPI Excel 数据（支持不同表名与引擎容错）"""
    url = "https://www.newyorkfed.org/medialibrary/research/interactives/gscpi/downloads/gscpi_data.xlsx"
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    engines = [None, "openpyxl", "xlrd"]
    last_err = None
    xls = None
    for eng in engines:
        try:
            xls = pd.ExcelFile(io.BytesIO(r.content), engine=eng)
            break
        except Exception as e:
            last_err = e
            xls = None
    if xls is None:
        raise RuntimeError(f"无法打开 GSCPI Excel：{last_err}")

    df_candidate = None
    for sheet in xls.sheet_names:
        try:
            df0 = pd.read_excel(xls, sheet_name=sheet)
            if df0.shape[1] >= 2:
                df0 = df0.rename(columns={df0.columns[0]: "date", df0.columns[1]: "value"})
                df0 = df0[["date", "value"]]
                df0["date"] = pd.to_datetime(df0["date"], errors="coerce")
                df0["value"] = pd.to_numeric(df0["value"], errors="coerce")
                df0 = df0.dropna().reset_index(drop=True)
                if len(df0) > 0:
                    df_candidate = df0
                    print(f"✅ 选定工作表 {sheet}，共 {len(df0)} 条记录")
                    break
        except Exception:
            continue

    if df_candidate is None:
        raise RuntimeError("未能从 GSCPI 文件中读取有效数据")
    return df_candidate.sort_values("date").reset_index(drop=True)


# ========== 计算函数 ==========
def last_yoy(df):
    if len(df) < 13: return None
    a, b = df["value"].iloc[-1], df["value"].iloc[-13]
    if pd.isna(a) or pd.isna(b) or b == 0: return None
    return (a / b - 1) * 100

def last_3mma_annualized(df):
    if len(df) < 4: return None
    p0, p1 = df["value"].iloc[-4], df["value"].iloc[-1]
    if pd.isna(p0) or pd.isna(p1) or p0 == 0: return None
    return ((p1 / p0) ** 4 - 1) * 100

def sahm_rule_gap(df):
    if len(df) < 15: return None
    x = df["value"].rolling(3).mean()
    rolling_min = x.rolling(12).min()
    return float((x - rolling_min).iloc[-1])

def gdp_qoq(df):
    if len(df) < 2: return None
    p0, p1 = df["value"].iloc[-2], df["value"].iloc[-1]
    if pd.isna(p0) or pd.isna(p1) or p0 == 0: return None
    return (p1 / p0 - 1) * 100

def last_month_value(df):
    if df.empty: return None, None
    return df["date"].iloc[-1].to_pydatetime(), float(df["value"].iloc[-1])


# ========== 数据库操作 ==========
def ensure_db():
    conn = sqlite3.connect(DB_PATH)
    with conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS last_seen (series_id TEXT PRIMARY KEY, last_date TEXT)"
        )
    conn.close()

def is_new_data(series_id: str, last_date: datetime) -> bool:
    """检查是否有新数据并自动修复 NaT"""
    conn = sqlite3.connect(DB_PATH)
    with conn:
        row = conn.execute(
            "SELECT last_date FROM last_seen WHERE series_id=?", (series_id,)
        ).fetchone()

        if row is None or not row[0] or row[0] in ("NaT", "nan", "None"):
            conn.execute(
                "INSERT OR REPLACE INTO last_seen(series_id, last_date) VALUES (?, ?)",
                (series_id, last_date.date().isoformat()),
            )
            return True

        try:
            prev = datetime.fromisoformat(row[0])
        except Exception:
            conn.execute(
                "UPDATE last_seen SET last_date=? WHERE series_id=?",
                (last_date.date().isoformat(), series_id),
            )
            return True

        if last_date.date() > prev.date():
            conn.execute(
                "UPDATE last_seen SET last_date=? WHERE series_id=?",
                (last_date.date().isoformat(), series_id),
            )
            return True
    return False


# ========== 评分与格式化 ==========
def severity_scale(value, lo, hi):
    if value is None: return 0.0
    return float(max(0.0, min(1.0, (value - lo) / (hi - lo))))

def fmt(v, digits=2):
    """安全格式化：None/NaN -> NA"""
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "NA"
        return f"{float(v):.{digits}f}"
    except Exception:
        return "NA"

def compute_risk_and_text(metrics):
    infl1 = severity_scale((metrics.get("core_pce_yoy") or 0) - 2.0, 0.0, 2.0)
    infl2 = severity_scale((metrics.get("core_pce_3mma_ann") or 0) - 2.0, 0.0, 2.0)
    infl = 0.6 * infl1 + 0.4 * infl2
    sahm = severity_scale(metrics.get("sahm_gap") or 0, 0.0, 0.5)
    growth = severity_scale(-(metrics.get("gdp_qoq") or 0), 0.0, 2.0)
    energy = severity_scale(metrics.get("wti_yoy") or 0, 0.0, 25.0)
    supply = severity_scale(metrics.get("gscpi") or 0, 0.0, 1.0) * 0.2
    sentiment = severity_scale(85 - (metrics.get("umich") or 85), 0.0, 20.0) * 0.2
    score = 100 * (0.4 * infl + 0.25 * sahm + 0.25 * growth + 0.1 * energy + 0.05 * supply + 0.05 * sentiment)

    bucket = "低" if score < 30 else "关注" if score < 60 else "显著" if score < 80 else "高"

    parts = [
        f"当前滞胀风险评分 **{round(score)}/100（{bucket}）**",
        f"- 核心PCE同比 {fmt(metrics.get('core_pce_yoy'))}%；3个月年化 {fmt(metrics.get('core_pce_3mma_ann'))}%",
        f"- 失业率 Sahm gap {fmt(metrics.get('sahm_gap'))}pct（≥0.5 为衰退信号）",
        f"- 实际GDP q/q {fmt(metrics.get('gdp_qoq'))}%",
        f"- WTI 原油同比 {fmt(metrics.get('wti_yoy'))}%",
        f"- GSCPI {fmt(metrics.get('gscpi'))}",
        f"- 密歇根信心 {fmt(metrics.get('umich'), 1)}",
    ]
    return int(round(score)), "\n".join(parts)


# ========== 通知（Slack & 邮件） ==========
#def send_slack(text: str, title="滞胀风险更新", touched_series=None):
#    if not SLACK_WEBHOOK_URL:
#        print("\n[Slack未配置] 输出结果：\n", text)
#        return
#    # 对重要指标出现时触发 @ 提醒（可选）
#    important = {"cpi_headline", "pce_core", "unrate", "real_gdp"}
#    need_ping = bool(touched_series and important.intersection(set(touched_series)) and ALERT_MENTION)
#    prefix = f"{ALERT_MENTION} " if need_ping else ""
#    payload = {
#        "text": f"{prefix}*{title}*\n{text}",
#        "mrkdwn": True,
#        "link_names": 1
#    }
#    try:
#        r = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=15)
#        r.raise_for_status()
#        print("✅ Slack 消息已发送" + ("（含@提醒）" if need_ping else ""))
#    except Exception as e:
#        print(f"❌ 发送 Slack 失败: {e}")

def send_email(subject: str, body: str, html: bool=False):
    """通过 SMTP 发送邮件：
       1) 先尝试 STARTTLS (默认587)
       2) 失败后自动回退为 SSL 直连 (465)
    """
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASS:
        print("⚠️ 邮件通知未配置，跳过发送。")
        return

    recipients = [a.strip() for a in EMAIL_TO.split(",") if a.strip()]
    msg = MIMEMultipart()
    msg["From"] = EMAIL_FROM or SMTP_USER
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    subtype = "html" if html else "plain"
    msg.attach(MIMEText(body, subtype, "utf-8"))

    timeout = 30
    use_ssl_env = os.getenv("SMTP_USE_SSL", "").lower() in ("1", "true", "yes")

    def _send_starttls():
        import smtplib
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=timeout) as server:
            # 可按需打开调试：将 SMTP_DEBUG=1 写入 .env
            if os.getenv("SMTP_DEBUG", "") == "1":
                server.set_debuglevel(1)
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(msg["From"], recipients, msg.as_string())

    def _send_ssl():
        import smtplib, ssl
        port = 465  # Gmail/大多数服务的 SSL 端口
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_HOST, port, context=context, timeout=timeout) as server:
            if os.getenv("SMTP_DEBUG", "") == "1":
                server.set_debuglevel(1)
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(msg["From"], recipients, msg.as_string())

    try:
        if use_ssl_env:
            _send_ssl()
        else:
            try:
                _send_starttls()
            except Exception as e1:
                print(f"⚠️ STARTTLS 失败：{e1}，改用 SSL(465) 重试…")
                _send_ssl()
        print(f"📧 邮件已发送到: {', '.join(recipients)}")
    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")



# ========== 主流程 ==========
def main():
    if not FRED_API_KEY:
        raise SystemExit("❌ 缺少 FRED_API_KEY，请在 .env 中配置")

    ensure_db()
    series_latest, new_flags = {}, []

    for k, sid in SERIES.items():
        try:
            if k == "gscpi":
                try:
                    df = fred_observations(sid, limit=24)
                except Exception:
                    print("⚠️ FRED 无 GSCPI，改用纽约联储数据")
                    df = get_gscpi_from_nyfed()
            else:
                df = fred_observations(sid, limit=24 if k != "real_gdp" else 8)
        except Exception as e:
            print(f"❌ 拉取 {k} ({sid}) 失败：{e}")
            df = pd.DataFrame(columns=["date", "value"]).assign(value=np.nan)

        dt, val = last_month_value(df)
        series_latest[k] = {"date": dt, "value": val, "df": df}
        if dt and is_new_data(sid, dt):
            new_flags.append(k)

    # 无新数据也执行一次
    if not new_flags:
        print("⚙️ 无新数据，但仍执行完整计算。")

    metrics = {
        "core_pce_yoy": last_yoy(series_latest["pce_core"]["df"]),
        "core_pce_3mma_ann": last_3mma_annualized(series_latest["pce_core"]["df"]),
        "sahm_gap": sahm_rule_gap(series_latest["unrate"]["df"]),
        "gdp_qoq": gdp_qoq(series_latest["real_gdp"]["df"]),
        "wti_yoy": last_yoy(series_latest["wti"]["df"]),
        "gscpi": series_latest.get("gscpi", {}).get("value"),
        "umich": series_latest.get("umich", {}).get("value"),
    }

    score, body = compute_risk_and_text(metrics)
    title = f"滞胀关键数据更新：{', '.join(sorted(new_flags))}" if new_flags else "手动运行：最新滞胀风险评估"
    message = textwrap.dedent(f"""
    更新时间（UTC）: {datetime.now(timezone.utc):%Y-%m-%d %H:%M}
    新数据：{', '.join(new_flags) if new_flags else '无'}

    {body}
    """).strip()

    print(message)
#    send_slack(message, title, touched_series=new_flags)
    # 纯文本邮件（如需HTML，把 html=True，并传入HTML模板字符串）
    send_email(title, message, html=False)


if __name__ == "__main__":
    main()
