# stagflation-monitor

一个可直接部署的**美国滞胀风险自动监控**最小可行项目（MVP）。
- 数据源：**FRED API**（圣路易斯联储），仅需一个 `FRED_API_KEY`。
- 指标覆盖：CPI（CPIAUCSL）、核心PCE（PCEPILFE）、失业率（UNRATE）、WTI油价（DCOILWTICO）、真实GDP（GDPC1）、密歇根消费者信心（UMCSENT）、全球供应链压力（GSCPI）。
- 模型：简单的**滞胀风险评分 S(0-100)** = 通胀压力(40%) + 劳动力/失业(25%) + 增长放缓(25%) + 能源(10%)。
- 通知：默认 SMTP 邮件。

## 本地运行
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export FRED_API_KEY=你的key
python monitor.py
```

## GitHub Actions（推荐，免服务器）
1. 将本仓库推送到 GitHub。
2. 在 **Settings → Secrets and variables → Actions** 新建：`FRED_API_KEY`。
3. Actions 会在工作日 **13:00 UTC** 运行（可在 `.github/workflows/monitor.yml` 中修改 `cron`）。

## 配置
- `.env.example` 给出可用的环境变量。
- `monitor.py` 会创建 `state.db`（SQLite）记录已处理的数据时间戳，避免重复发通知。

## 警报逻辑（可在 `monitor.py` 中调整阈值）
- **通胀压力**：核心PCE同比高于 2% 的超目标幅度（3个月年化趋势上行加权）。
- **劳动力走弱**：**Sahm Rule** 间接度量（3个月均值相对过去12个月低点的抬升，0.5pct=强信号）。
- **增长放缓**：近两个季度 **Real GDP (GDPC1)** 的环比增速（近似）若转负，加分。
- **能源推升**：WTI油价同比涨幅（>25% 计满分）。
- **总分阈值**：<30 低；30–60 关注；60–80 显著；>80 高。

# sf-moni
