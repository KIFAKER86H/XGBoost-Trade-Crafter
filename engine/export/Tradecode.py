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
TIMEFRAME = mt5.TIMEFRAME_M15
MAGIC_NUMBER = 150369
MODEL_NUM = 999
SL_MULTIPLIER = 1.5
TP_MULTIPLIER = 2

# Money Management
EQUITY_PER_LOT = 300 

MODEL_PATH = "models/XGBOOST_Model_999.pkl"
SCALER_PATH = "scalers/XGBOOST_scaler_999.pkl"
HISTORY_BARS = 1000 

INDICATOR_CONFIG = {'atr': 24, 'rsi': 7, 'bb': 48, 'roc': 48, 'cci': 72, 'macd': [14, 48, 75], 'ema': [7, 21, 160], 'sma': [10, 25, 200]}
LAGS_CONFIG = {'number': 6, 'stride': 2}

# ==============================
# 1) FEATURE ENGINEERING
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
def calculate_dynamic_lot(symbol, equity_per_lot=EQUITY_PER_LOT, lot_step_size=0.01):
    account_info = mt5.account_info()
    if account_info is None:
        return 0.01
    equity = account_info.equity
    calculated_lot = (equity / equity_per_lot) * lot_step_size

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        return 0.01

    min_lot = symbol_info.volume_min
    max_lot = symbol_info.volume_max
    lot_step = symbol_info.volume_step

    final_lot = round(calculated_lot / lot_step) * lot_step
    final_lot = max(min_lot, min(final_lot, max_lot))
    return round(final_lot, 2)

def open_trade(action, price, sl, tp):
    dynamic_lot = calculate_dynamic_lot(SYMBOL)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": dynamic_lot,
        "type": action,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "XGForge AI",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[ERROR] Order Send Failed: {result.comment}")
    else:
        print(f"[SUCCESS] Trade Opened! Ticket: {result.order} | Lot: {dynamic_lot}")

def close_position(position):
    tick = mt5.symbol_info_tick(position.symbol)
    close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    close_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": position.ticket,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": close_type,
        "price": close_price,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "AI Reversal Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"[CLOSED] Position {position.ticket} Closed Successfully.")
        return True
    else:
        print(f"[ERROR] Failed to close position: {result.comment}")
        return False

# ==============================
# 3) MAIN BOT LOGIC
# ==============================
def main():
    if not mt5.initialize():
        print("[ERROR] MT5 Initialize failed")
        return
        
    print(f"[START] Running Trade Bot for {SYMBOL}...")
    
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        mt5.shutdown()
        return

    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, HISTORY_BARS)
    if rates is None or len(rates) == 0:
        mt5.shutdown()
        return
        
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    df_feat = prepare_features(df, INDICATOR_CONFIG, LAGS_CONFIG['number'])
    atr_col = f"ATR_{INDICATOR_CONFIG.get('atr', 48)}"
    if atr_col in df_feat.columns:
        df_feat = add_lags_custom(df_feat, [atr_col], LAGS_CONFIG['number'], LAGS_CONFIG['stride'])
    
    df_feat.dropna(inplace=True)
    if len(df_feat) == 0:
        mt5.shutdown()
        return

    last_bar = df_feat.iloc[-1:]
    current_tick = mt5.symbol_info_tick(SYMBOL)
    current_atr = last_bar[atr_col].values[0]

    X_live = last_bar.drop(columns=['time'], errors='ignore')
    X_scaled = scaler.transform(X_live)

    prediction = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    print(f"[PREDICT] ML Prediction: {'UPTREND (BUY)' if prediction == 1 else 'DOWNTREND (SELL)'} | Confidence: {prob*100:.2f}%")

    positions = mt5.positions_get(symbol=SYMBOL)
    has_buy = False
    has_sell = False

    if positions:
        for pos in positions:
            if pos.magic != MAGIC_NUMBER:
                continue

            if pos.type == mt5.ORDER_TYPE_BUY:
                if prob < 0.55: 
                    print(f"[ALERT] Confidence dropped to {prob*100:.2f}%: Closing existing BUY position...")
                    close_position(pos)
                else:
                    has_buy = True 
            elif pos.type == mt5.ORDER_TYPE_SELL:
                if (1 - prob) < 0.55: 
                    print(f"[ALERT] Confidence dropped to {(1-prob)*100:.2f}%: Closing existing SELL position...")
                    close_position(pos)
                else:
                    has_sell = True 

    if not has_buy and not has_sell:
        if prediction == 1 and prob > 0.60:
            sl = current_tick.ask - (current_atr * SL_MULTIPLIER)
            tp = current_tick.ask + (current_atr * TP_MULTIPLIER)
            print(f"[BUY] Signal: STRONG BUY | High Confidence: {prob*100:.2f}%")
            open_trade(mt5.ORDER_TYPE_BUY, current_tick.ask, sl, tp)
            
        elif prediction == 0 and prob < 0.40: 
            sl = current_tick.bid + (current_atr * SL_MULTIPLIER)
            tp = current_tick.bid - (current_atr * TP_MULTIPLIER)
            print(f"[SELL] Signal: STRONG SELL | High Confidence: {(1-prob)*100:.2f}%")
            open_trade(mt5.ORDER_TYPE_SELL, current_tick.bid, sl, tp)
            
        else:
            print("[HOLD] Waiting... Confidence is in the neutral zone (between 40% - 60%).")
    else:
        print("[WAIT] Already holding the correct position. Waiting for next signal.")

    mt5.shutdown()

if __name__ == "__main__":
    main()
