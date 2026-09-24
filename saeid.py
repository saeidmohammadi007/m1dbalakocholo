import os
import pandas as pd
import numpy as np
import yfinance as yf
from tqdm import tqdm
import time
import requests
import ccxt

# --- الگوی مرجع: تک‌کندل روزانه ETH ---
PATTERN_SYMBOL = 'ETH-USD'
PATTERN_DATE   = '2025-07-04'
SHOW_N         = 10

# ---------- توابع ----------
def get_lbank_futures_symbols():
    exchange = ccxt.lbank({'options': {'defaultType': 'future'}})
    try:
        markets = exchange.load_markets()
    except Exception as e:
        print(f"❌ خطا در اتصال به LBank: {e}")
        return []

    base_list = []
    for symbol, market in markets.items():
        if not market.get('swap'):
            continue
        base = market.get('base')
        if not base or base.isdigit():
            continue
        base_list.append(base.upper())

    seen, unique_bases = set(), []
    for b in base_list:
        if b not in seen:
            seen.add(b)
            unique_bases.append(b)
    print(f"✅ تعداد ارزهای پایه‌ی منحصربه‌فرد فیوچرز LBank: {len(unique_bases)}")
    return unique_bases

def get_daily_data(ticker, start='2024-01-01'):
    df = yf.download(ticker, start=start, interval='1d',
                     progress=False, auto_adjust=False)
    if df.empty:
        return None
    df = df[['Open', 'High', 'Low', 'Close']].copy()
    df.index = pd.to_datetime(df.index)
    df.columns = ['open', 'high', 'low', 'close']
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df

def candle_vector(o, h, l, c):
    """بردار ۴بعدی نرمال‌شده‌ی کندل نسبت به دامنه High−Low"""
    rng = float(h - l)
    if rng <= 0 or not np.isfinite(rng):
        return None
    return np.array([
        (float(o) - float(l)) / rng,
        1.0,
        0.0,
        (float(c) - float(l)) / rng,
    ])

def compute_macd(close_series, fast=12, slow=26, signal=9):
    """محاسبه‌ی MACD و Signal روی سری Close"""
    ema_fast = close_series.ewm(span=fast, adjust=False).mean()
    ema_slow = close_series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def send_telegram_message(text):
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        print("❌ توکن یا chat_id تنظیم نشده است.")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': text, 'parse_mode': 'HTML'}
    try:
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code != 200:
            print(f"⚠️ خطا در ارسال پیام: {r.text}")
        else:
            print("✅ پیام با موفقیت به تلگرام ارسال شد.")
    except Exception as e:
        print(f"❌ خطا در ارسال به تلگرام: {e}")

# ---------- اجرای اصلی ----------
print(f"🔍 استخراج کندل مرجع {PATTERN_SYMBOL} در تاریخ {PATTERN_DATE} (روزانه) ...")
ref_daily = get_daily_data(PATTERN_SYMBOL, start='2024-01-01')
if ref_daily is None:
    print(f"❌ خطا در دریافت داده‌های {PATTERN_SYMBOL}")
    exit()

target_date = pd.to_datetime(PATTERN_DATE).date()
mask = ref_daily.index.date == target_date
if not mask.any():
    print(f"❌ کندلی برای تاریخ {PATTERN_DATE} یافت نشد.")
    exit()

candle = ref_daily[mask].iloc[0]
o, h, l, c = candle['open'], candle['high'], candle['low'], candle['close']
pattern_vec = candle_vector(o, h, l, c)
if pattern_vec is None:
    print("❌ کندل مرجع نامعتبر است.")
    exit()

print(f"📌 کندل مرجع (روزانه): O={o:.4f}  H={h:.4f}  L={l:.4f}  C={c:.4f}")
print(f"📐 بردار الگو: O={pattern_vec[0]:.3f} | H={pattern_vec[1]:.3f} | "
      f"L={pattern_vec[2]:.3f} | C={pattern_vec[3]:.3f}")

print("\n📊 دریافت نمادهای فیوچرز LBank ...")
symbols = get_lbank_futures_symbols()
if not symbols:
    print("❌ هیچ نمادی برای اسکن وجود ندارد!")
    exit()

results = []
for sym in tqdm(symbols, desc="اسکن کندل روزانه (قبلی)"):
    try:
        df_1d = get_daily_data(f"{sym}-USD", start='2024-01-01')
        if df_1d is None or len(df_1d) < 35:   # حداقل داده برای MACD
            continue

        # ---------- ✅ شرط MACD: خط MACD بالای خط Signal باشد ----------
        macd_line, signal_line = compute_macd(df_1d['close'])
        if not (macd_line.iloc[-2] > signal_line.iloc[-2]):
            continue
        # --------------------------------------------------------------

        prev = df_1d.iloc[-2]
        vec = candle_vector(prev['open'], prev['high'],
                            prev['low'],  prev['close'])
        if vec is None:
            continue

        dist = float(np.linalg.norm(pattern_vec - vec))

        results.append({
            'symbol':   sym,
            'dist':     dist,
            'last_1d':  df_1d.index[-2].strftime('%Y-%m-%d'),
            'o': float(prev['open']),
            'h': float(prev['high']),
            'l': float(prev['low']),
            'c': float(prev['close']),
            'macd': float(macd_line.iloc[-2]),
            'signal': float(signal_line.iloc[-2]),
        })
        time.sleep(0.3)
    except Exception:
        continue

if results:
    df_res = pd.DataFrame(results).sort_values('dist').head(SHOW_N)

    lines = []
    lines.append(f"🏆 <b>کندل‌های روزانه (قبلی) مشابه کندل روزانه {PATTERN_SYMBOL} ({PATTERN_DATE})</b>\n")
    lines.append(f"الگو (روزانه): O={o:.6g} | H={h:.6g} | L={l:.6g} | C={c:.6g}")
    lines.append("🔎 <i>فیلتر فعال: MACD > Signal</i>\n")
    for _, row in df_res.iterrows():
        lines.append(
            f"🔸 <b>{row['symbol']}</b>  (فاصله: {row['dist']:.4f})\n"
            f"   تاریخ: {row['last_1d']} | "
            f"O={row['o']:.6g} H={row['h']:.6g} "
            f"L={row['l']:.6g} C={row['c']:.6g}\n"
            f"   MACD={row['macd']:.4g} > Signal={row['signal']:.4g}"
        )
    lines.append(f"\n📅 تعداد ارزهای اسکن‌شده: {len(symbols)}")
    message = "\n".join(lines)

    send_telegram_message(message)
    print("\n" + message)
else:
    print("\n❌ نتیجه‌ای یافت نشد.")
    send_telegram_message("❌ در اسکن امروز هیچ نتیجه‌ای یافت نشد.")

print("\n✅ اسکن کامل شد!")
