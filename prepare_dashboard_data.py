# prepare_dashboard_data.py
# Reads the raw CSVs, merges them, computes stats, and writes dashboard.html
# Run: python data/prepare_dashboard_data.py

import pandas as pd
import numpy as np
import json, os

DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(DIR)

# ── load data ──────────────────────────────────────────────
print("Loading CSVs...")
fgi = pd.read_csv(os.path.join(DIR, "fear_greed_index.csv"))
trades = pd.read_csv(os.path.join(DIR, "historical_data.csv"))

# parse dates
fgi['date'] = pd.to_datetime(fgi['date'])
fgi['day'] = fgi['date'].dt.date

trades['dt'] = pd.to_datetime(trades['Timestamp IST'], format='%d-%m-%Y %H:%M', errors='coerce')
trades['day'] = trades['dt'].dt.date

# merge on date
df = pd.merge(trades, fgi, on='day', how='inner')
print(f"Merged: {len(df)} rows, {df['day'].nunique()} unique days")

# simplify direction
def dir_bucket(d):
    d = str(d).lower()
    if 'long' in d or 'buy' in d: return 'Bullish'
    if 'short' in d or 'sell' in d: return 'Bearish'
    return 'Other'

df['dir'] = df['Direction'].apply(dir_bucket)
df['win'] = np.where(df['Closed PnL'] > 0, 1, np.where(df['Closed PnL'] < 0, 0, np.nan))

SENTIMENTS = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']

def win_rate(series):
    valid = series.dropna()
    return float(valid.mean()) if len(valid) > 0 else 0.0

# ── 1. overall KPIs ───────────────────────────────────────
overall = {
    "totalTrades": int(len(df)),
    "totalPnL": float(df['Closed PnL'].sum()),
    "totalVolume": float(df['Size USD'].sum()),
    "winRate": win_rate(df['win']),
    "totalFee": float(df['Fee'].sum()),
}

# ── 2. per-sentiment stats ────────────────────────────────
sent_stats = {}
for s in SENTIMENTS:
    sub = df[df['classification'] == s]
    n = len(sub)
    dirs = sub['dir'].value_counts()
    sent_stats[s] = {
        "trades": n,
        "totalPnL": float(sub['Closed PnL'].sum()),
        "meanPnL": float(sub['Closed PnL'].mean()) if n else 0,
        "volume": float(sub['Size USD'].sum()),
        "winRate": win_rate(sub['win']),
        "bullishCount": int(dirs.get('Bullish', 0)),
        "bearishCount": int(dirs.get('Bearish', 0)),
        "otherCount": int(dirs.get('Other', 0)),
    }

# ── 3. per-account profiles ──────────────────────────────
accounts = []
for acc, grp in df.groupby('Account'):
    n = len(grp)
    pnl = grp['Closed PnL'].sum()
    vol = grp['Size USD'].sum()
    wr = win_rate(grp['win'])
    fee = grp['Fee'].sum()

    # per-sentiment breakdown for this account
    acc_sent = {}
    for s in SENTIMENTS:
        sg = grp[grp['classification'] == s]
        sn = len(sg)
        dc = sg['dir'].value_counts()
        bc = int(dc.get('Bullish', 0))
        sc = int(dc.get('Bearish', 0))
        acc_sent[s] = {
            "trades": sn,
            "totalPnL": float(sg['Closed PnL'].sum()),
            "meanPnL": float(sg['Closed PnL'].mean()) if sn else 0,
            "winRate": win_rate(sg['win']),
            "volume": float(sg['Size USD'].sum()),
            "bullishPct": bc / sn if sn else 0,
            "bearishPct": sc / sn if sn else 0,
        }

    # classify trading style based on contrarian behavior
    fear_n = sum(acc_sent[r]['trades'] for r in ['Extreme Fear', 'Fear'])
    fear_bull = sum(acc_sent[r]['bullishPct'] * acc_sent[r]['trades'] for r in ['Extreme Fear', 'Fear'])
    fb_pct = fear_bull / fear_n if fear_n else 0

    greed_n = sum(acc_sent[r]['trades'] for r in ['Greed', 'Extreme Greed'])
    greed_bear = sum(acc_sent[r]['bearishPct'] * acc_sent[r]['trades'] for r in ['Greed', 'Extreme Greed'])
    gb_pct = greed_bear / greed_n if greed_n else 0

    if pnl > 0:
        if fb_pct > 0.55 and gb_pct > 0.55:
            style = "Contrarian Master"
            desc = "Buys during fear, shorts during greed — classic contrarian approach that works."
        elif fb_pct > 0.60:
            style = "Fear Buyer"
            desc = "Specializes in buying dips during panic/fear periods."
        elif gb_pct > 0.60:
            style = "Greed Seller"
            desc = "Profits by shorting overextended rallies during euphoria."
        elif fb_pct < 0.45 and gb_pct < 0.45:
            style = "Trend Follower"
            desc = "Rides momentum — buys strength, shorts weakness."
        else:
            style = "Balanced Scalper"
            desc = "Takes quick profits in both directions across all sentiment regimes."
    else:
        if fb_pct < 0.35 and gb_pct < 0.35:
            style = "Panic Chaser"
            desc = "Chases momentum at the worst times — buys tops, shorts bottoms."
        elif fb_pct < 0.35:
            style = "Bottom Shorter"
            desc = "Shorts during capitulation, gets squeezed on bounces."
        elif gb_pct < 0.35:
            style = "FOMO Buyer"
            desc = "Buys blow-off tops during extreme greed, eats the correction."
        else:
            style = "Over-Leveraged"
            desc = "High activity but poor risk management across sentiment shifts."

    accounts.append({
        "account": str(acc),
        "trades": int(n),
        "totalPnL": float(pnl),
        "meanPnL": float(grp['Closed PnL'].mean()),
        "winRate": wr,
        "volume": float(vol),
        "fee": float(fee),
        "profileType": style,
        "description": desc,
        "sentimentStats": acc_sent,
    })

accounts.sort(key=lambda x: x['totalPnL'], reverse=True)

# ── 4. top coins breakdown ───────────────────────────────
TOP_COINS = ['HYPE', '@107', 'BTC', 'ETH', 'SOL']
coins = {}
for c in TOP_COINS:
    cg = df[df['Coin'] == c]
    cn = len(cg)
    cs = {}
    for s in SENTIMENTS:
        sg = cg[cg['classification'] == s]
        sn = len(sg)
        cs[s] = {
            "trades": sn,
            "totalPnL": float(sg['Closed PnL'].sum()),
            "meanPnL": float(sg['Closed PnL'].mean()) if sn else 0,
            "winRate": win_rate(sg['win']),
            "volume": float(sg['Size USD'].sum()),
        }
    coins[c] = {
        "trades": cn,
        "totalPnL": float(cg['Closed PnL'].sum()),
        "volume": float(cg['Size USD'].sum()),
        "winRate": win_rate(cg['win']),
        "sentimentStats": cs,
    }

# ── 5. daily aggregation for scatter plot ─────────────────
daily = df.groupby(['day', 'value', 'classification']).agg(
    pnl=('Closed PnL', 'sum'),
    vol=('Size USD', 'sum'),
    cnt=('Trade ID', 'count'),
).reset_index().sort_values('day')

daily_list = [
    {"date": str(r['day']), "fgiValue": int(r['value']),
     "fgiClassification": str(r['classification']),
     "dailyPnL": float(r['pnl']), "dailyVolume": float(r['vol']),
     "dailyTradeCount": int(r['cnt'])}
    for _, r in daily.iterrows()
]

# ── assemble and inject into HTML ─────────────────────────
payload = {
    "overallMetrics": overall,
    "sentimentStats": sent_stats,
    "traders": accounts,
    "coins": coins,
    "dailyTrading": daily_list,
}

json_str = json.dumps(payload, indent=None)  # compact to save space

# read template, inject data, write final dashboard
tmpl_path = os.path.join(ROOT, "dashboard.html")

# read dashboard.html, inject the data, write it back
with open(tmpl_path, 'r', encoding='utf-8') as f:
    html = f.read()

injection = f'const dashboardData = {json_str};'

if '// DATA_PLACEHOLDER' in html:
    # first-time inject from template
    html = html.replace('// DATA_PLACEHOLDER', injection)
elif 'const dashboardData = ' in html:
    # re-inject: find the start and end of the old data block
    start = html.index('const dashboardData = ')
    # find the closing ';' — scan for '};' after the opening '{'
    brace_start = html.index('{', start)
    depth = 0
    i = brace_start
    while i < len(html):
        if html[i] == '{': depth += 1
        elif html[i] == '}': depth -= 1
        if depth == 0: break
        i += 1
    # i is now at the closing '}', find the ';' after it
    end = html.index(';', i) + 1
    html = html[:start] + injection + html[end:]
else:
    print("WARNING: couldn't find injection point in dashboard.html")

with open(tmpl_path, 'w', encoding='utf-8') as f:
    f.write(html)

size_kb = os.path.getsize(tmpl_path) / 1024
print(f"Done — dashboard.html updated ({size_kb:.0f} KB)")

