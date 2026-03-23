import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import talib

# ==============================
# 0) GLOBAL CONFIG
# ==============================
SYMBOL = "XAUUSDm"
TIMEFRAME = mt5.TIMEFRAME_15
MAGIC_NUMBER = 151515
MODEL_NUM = 999
SL_MULTIPLIER = 1.5
TP_MULTIPLIER = 2

MODEL_PATH = "models/XGBOOST_Model_999.pkl"
SCALER_PATH = "scalers/XGBOOST_scaler_999.pkl"
HISTORY_BARS = 1000 # ดึงย้อนหลังเผื่อคำนวณ Indicator (เช่น EMA400)
LOT_SIZE = 0.01     # ⚠️ คุณสามารถเปลี่ยน Lot Size ตรงนี้ได้

INDICATOR_CONFIG = {'atr': 48, 'rsi': 28, 'bb': 50, 'roc': 168, 'cci': 72, 'macd': [28, 96, 150], 'ema': [10, 25, 200], 'sma': [10, 25, 200]}
LAGS_CONFIG = {'number': 5, 'stride': 1}

# ==============================
# 1) FEATURE ENGINEERING (จำลองกระบวนการเดียวกับตอน Train เป๊ะๆ)
# ==============================
def prepare_features(df: pd.DataFrame, config: dict, lag_periods: int) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values('time')

    out['body_size'] = (out['close'] - out['open']).abs()
    out['upper_wick'] = out['high'] - out['close'].combine(out['open'], max)
    out['lower_wick'] = out['close'].combine(out['open'], min) - out['low']
    out['rel_body_size'] = out['body_size'] / (out['high'] - out['low']).replace(0, np.nan)

    for span in config.get('ema', [20, 50, 400]):
        ema_name = f"EMA_{span}"
        out[ema_name] = out['close'].ewm(span=span, adjust=False).mean()
        out[f"dist_{ema_name}"] = (out['close'] - out[ema_name]) / out[ema_name]

    for w in config.get('sma', [20, 50, 400]):
        sma_name = f"SMA_{w}"
        out[sma_name] = out['close'].rolling(window=w).mean()
        out[f"dist_{sma_name}"] = (out['close'] - out[sma_name]) / out[sma_name]

    p = int(config.get('atr', 48))
    tr = pd.concat([out['high']-out['low'], (out['high']-out['close'].shift(1)).abs(), (out['low']-out['close'].shift(1)).abs()], axis=1).max(axis=1)
    out[f"ATR_{p}"] = tr.ewm(span=p, adjust=False).mean()
    
    upper, mid, lower = talib.BBANDS(out['close'].values.astype(float), timeperiod=int(config.get('bb', 50)))
    out['BB_width'] = (upper - lower) / mid

    rsi_p = int(config.get('rsi', 28))
    delta = out['close'].diff()
    gain = delta.clip(lower=0); loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/rsi_p, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_p, adjust=False).mean()
    out['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss.replace(0, np.nan))))

    out['ROC'] = out['close'].pct_change(periods=int(config.get('roc', 168))) * 100
    f, s, sig = config.get('macd', [28, 96, 150])
    ema_f = out['close'].ewm(span=int(f), adjust=False).mean()
    ema_s = out['close'].ewm(span=int(s), adjust=False).mean()
    out['MACD_line'] = ema_f - ema_s
    out['MACD_hist'] = out['MACD_line'] - out['MACD_line'].ewm(span=int(sig), adjust=False).mean()
    
    tp = (out['high'] + out['low'] + out['close']) / 3
    cci_p = int(config.get('cci', 72))
    out['CCI'] = (tp - tp.rolling(cci_p).mean()) / (0.015 * tp.rolling(cci_p).apply(lambda x: np.mean(np.abs(x - np.mean(x)))))

    out['hour'] = out['time'].dt.hour
    out['day_of_week'] = out['time'].dt.dayofweek

    key_features = ['close', 'RSI', 'MACD_hist', f"dist_SMA_{config.get('sma',[400])[-1]}", 'body_size']
    for col in key_features:
        if col in out.columns:
            for i in range(1, lag_periods + 1):
                out[f"{col}_lag_{i}"] = out[col].shift(i)

    return out

def add_lags_custom(df: pd.DataFrame, columns: list, lags: int = 3, stride: int = 1) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            for i in range(1, lags + 1):
                actual_lag = i * stride
                out[f"{col}_lag_{actual_lag}"] = out[col].shift(actual_lag)
    return out

# ==============================
# 2) MT5 ORDER EXECUTION
# ==============================
def open_trade(action, price, sl, tp):
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOT_SIZE,
        "type": action,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "XGBoost Bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Order Send Failed: {result.comment}")
    else:
        print(f"✅ Trade Opened Successfully! Ticket: {result.order}")

# ==============================
# 3) MAIN BOT LOGIC
# ==============================
def main():
    if not mt5.initialize():
        print("❌ MT5 Initialize failed")
        return
        
    print(f"🚀 Running Trade Bot for {SYMBOL}...")
    
    # 3.1) โหลดโมเดล
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("✅ Model & Scaler loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        mt5.shutdown()
        return

    # 3.2) ดึงข้อมูลล่าสุด
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, HISTORY_BARS)
    if rates is None or len(rates) == 0:
        print("❌ Failed to fetch data.")
        mt5.shutdown()
        return
        
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # 3.3) สร้าง Feature (กระบวนการเดียวกับตอน Train)
    df_feat = prepare_features(df, INDICATOR_CONFIG, LAGS_CONFIG['number'])
    atr_col = f"ATR_{INDICATOR_CONFIG.get('atr', 48)}"
    if atr_col in df_feat.columns:
        df_feat = add_lags_custom(df_feat, [atr_col], LAGS_CONFIG['number'], LAGS_CONFIG['stride'])
    
    df_feat.dropna(inplace=True)
    if len(df_feat) == 0:
        print("❌ Not enough data to calculate features.")
        mt5.shutdown()
        return

    # 3.4) ดึงข้อมูลแท่งล่าสุดเตรียมเข้าโมเดล
    last_bar = df_feat.iloc[-1:]
    current_price = mt5.symbol_info_tick(SYMBOL).ask
    current_atr = last_bar[atr_col].values[0]

    X_live = last_bar.drop(columns=['time'], errors='ignore')
    
    try:
        X_scaled = scaler.transform(X_live)
    except Exception as e:
        print(f"❌ Feature Mismatch Error: {e}")
        mt5.shutdown()
        return

    # 3.5) ทำนายผล!
    prediction = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1] # หาความน่าจะเป็นที่จะขึ้น (Uptrend)

    print(f"📊 Prediction: {'UPTREND (1)' if prediction == 1 else 'DOWNTREND (0)'} | Confidence: {prob*100:.2f}%")

    # 3.6) ระบบจัดการออเดอร์ (ง่ายๆ คือเปิดทีละ 1 ไม้)
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None or len(positions) > 0:
        print("⏳ Already have an open position. Waiting for it to close...")
        mt5.shutdown()
        return

    # ลอจิกเปิดออเดอร์ (ตัวอย่าง: ถ้าคาดว่าจะขึ้น และมั่นใจ > 60% ให้เปิด BUY)
    if prediction == 1 and prob > 0.60:
        sl = current_price - (current_atr * SL_MULTIPLIER)
        tp = current_price + (current_atr * TP_MULTIPLIER)
        print(f"📈 Executing BUY | Price: {current_price}, SL: {sl:.3f}, TP: {tp:.3f}")
        open_trade(mt5.ORDER_TYPE_BUY, current_price, sl, tp)
    else:
        print("📉 Signal is neutral/downtrend or confidence too low. No trade taken.")

    mt5.shutdown()

if __name__ == "__main__":
    main()
