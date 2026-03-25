import os
import joblib
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import talib
import ta
from datetime import datetime, timedelta, timezone

# Sklearn & XGBoost
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline


# PyTorch (สำหรับ LSTM)
import torch
import torch.nn as nn

# ==========================================
# SECTION 1: DATA FETCHING (ดึงข้อมูลจาก MT5)
# ==========================================
def get_mt5_timeframe(tf_string: str):
    tf_map = {
        "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1, "MN1": mt5.TIMEFRAME_MN1
    }
    return tf_map.get(tf_string.upper(), mt5.TIMEFRAME_H1)

def fetch_mt5_data(symbol: str, timeframe_str: str, history_config):
    """ดึงข้อมูล MT5 ตามการตั้งค่าจาก UI (จำกัด MAX ที่ 100,000 แท่ง)"""
    if not mt5.initialize():
        raise Exception("MT5 Initialization failed.")

    if not mt5.symbol_select(symbol, True):
        mt5.shutdown()
        raise Exception(f"Symbol '{symbol}' not found. Please check your Market Watch.")

    tf = get_mt5_timeframe(timeframe_str)
    
    rates = None
    warning_msg = None

    if history_config == "MAX":
        # 💡 กำหนดลิมิตดึงข้อมูลสูงสุดที่ 100,000 แท่ง เพื่อความเสถียรและตรงกับ Default ของ MT5
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, 99999)
    else:
        # ถ้าเป็น Custom Date ดึงตามช่วงเวลาปกติ
        start_date = pd.to_datetime(history_config['start']).to_pydatetime()
        end_date = pd.to_datetime(history_config['end']).to_pydatetime()
        rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)

    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise Exception("No data fetched from MT5. Please check symbol or date range.")

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # เช็คว่าถ้าเป็น Custom Date ข้อมูลที่ได้มา เก่าถึงตามที่เราขอไปไหม
    if history_config != "MAX":
        actual_start = df.iloc[0]['time']
        requested_start = pd.to_datetime(history_config['start'])
        if actual_start.date() > requested_start.date():
            warning_msg = f"Data available only from {actual_start.date()} (Requested: {requested_start.date()})"

    df['open'] = df['open'].astype(np.float64)
    df['high'] = df['high'].astype(np.float64)
    df['low'] = df['low'].astype(np.float64)
    df['close'] = df['close'].astype(np.float64)
    df['tick_volume'] = df['tick_volume'].astype(np.float64)
    
    # กำจัดค่า Infinity และ NaN ก่อนส่งให้ XGBoost
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    return df, warning_msg

# ==========================================
# SECTION 2: FEATURE ENGINEERING (สร้าง Indicator & Lag)
# ==========================================
def prepare_features(df: pd.DataFrame, config: dict, lag_periods: int = 3) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values('time')

    # 1. Price Structure
    out['body_size'] = (out['close'] - out['open']).abs()
    out['upper_wick'] = out['high'] - out['close'].combine(out['open'], max)
    out['lower_wick'] = out['close'].combine(out['open'], min) - out['low']
    out['rel_body_size'] = out['body_size'] / (out['high'] - out['low']).replace(0, np.nan)

    # 2. EMA / SMA
    for span in config.get('ema', [20, 50, 400]): # ปรับให้รับ key 'ema' จาก UI
        ema_name = f"EMA_{span}"
        out[ema_name] = out['close'].ewm(span=span, adjust=False).mean()
        out[f"dist_{ema_name}"] = (out['close'] - out[ema_name]) / out[ema_name]

    for w in config.get('sma', [20, 50, 400]): # ปรับให้รับ key 'sma' จาก UI
        sma_name = f"SMA_{w}"
        out[sma_name] = out['close'].rolling(window=w).mean()
        out[f"dist_{sma_name}"] = (out['close'] - out[sma_name]) / out[sma_name]

    # 3. ATR, BB, RSI, ROC, MACD, CCI
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

    # 4. Time-based
    out['hour'] = out['time'].dt.hour
    out['day_of_week'] = out['time'].dt.dayofweek

    # 5. Lags สำหรับ Feature หลัก
    key_features = ['close', 'RSI', 'MACD_hist', f"dist_SMA_{config.get('sma',[400])[-1]}", 'body_size']
    for col in key_features:
        if col in out.columns:
            for i in range(1, lag_periods + 1):
                out[f"{col}_lag_{i}"] = out[col].shift(i)

    return out.dropna().reset_index(drop=True)

def add_lags_custom(df: pd.DataFrame, columns: list, lags: int = 3, stride: int = 1) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            for i in range(1, lags + 1):
                actual_lag = i * stride
                out[f"{col}_lag_{actual_lag}"] = out[col].shift(actual_lag)
    return out

def prepare_features_v2(df, config, lags=5, stride=1):
    df_features = prepare_features(df, config=config, lag_periods=lags)
    atr_col = f"ATR_{config.get('atr', 48)}"
    if atr_col in df_features.columns:
        df_final = add_lags_custom(df_features, columns=[atr_col], lags=lags, stride=stride)
    else:
        df_final = df_features
    return df_final.dropna()

def create_labels(df, future_bars=24):
    """สร้าง Label จากการเปรียบเทียบค่าอนาคต (กระบวนการของคุณ)"""
    out = df.copy()
    out[f'EMA{future_bars}'] = ta.trend.ema_indicator(out['close'], window=future_bars)
    out[f'EMA{future_bars}_Future'] = out[f'EMA{future_bars}'].shift(-future_bars)
    out['Label'] = 0
    out.loc[out[f'EMA{future_bars}_Future'] > out[f'EMA{future_bars}'], 'Label'] = 1
    return out.dropna(subset=[f'EMA{future_bars}_Future']).copy()


# ==========================================
# SECTION 3: MODEL TRAINING (แยก Dataset, Train XGBoost, Save)
# ==========================================
def prepare_for_training(df, target_col='Label', train_size=0.7, val_size=0.15):
    X = df.drop(columns=['time', target_col], errors='ignore')
    target_map = {0: 0, 1: 1}
    y = df[target_col].map(target_map).fillna(1)
    
    n = len(df)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    
    X_train_raw, X_val_raw, X_test_raw = X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:]
    y_train, y_val, y_test = y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:]
    
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X.columns)
    X_val = pd.DataFrame(scaler.transform(X_val_raw), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=X.columns)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

def train_xgboost(X_train, y_train, X_val, y_val, model_num=1):
    """รัน GridSearchCV ค้นหาพารามิเตอร์ที่ดีที่สุดและบันทึกโมเดล"""
    param_grid = {
        'xgb__max_depth': [3, 5, 7],
        'xgb__n_estimators': [50, 100],
        'xgb__learning_rate': [0.01, 0.1]
    }

    pipeline = Pipeline([
        ('scaler', StandardScaler()), 
        ('xgb', XGBClassifier(objective='binary:logistic', random_state=42))
    ])

    # 🌟 สร้าง TimeSeriesSplit สำหรับเดินหน้าอย่างเดียว (ป้องกัน Lookahead Bias)
    tscv = TimeSeriesSplit(n_splits=3)

    # 🌟 ใส่ tscv เข้าไปที่พารามิเตอร์ cv
    grid_XGB = GridSearchCV(pipeline, param_grid, cv=tscv, scoring='accuracy', verbose=0, n_jobs=-1)
    grid_XGB.fit(X_train, y_train)

    model = grid_XGB.best_estimator_
    y_pred = model.predict(X_val)

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred) * 100),
        "precision": float(precision_score(y_val, y_pred, average='macro', zero_division=0)),
        "recall": float(recall_score(y_val, y_pred, average='macro', zero_division=0)),
        "f1": float(f1_score(y_val, y_pred, average='macro', zero_division=0))
    }
    return model, metrics

def save_model_artifacts(model, scaler, model_num):
    os.makedirs("export/models", exist_ok=True)
    os.makedirs("export/scalers", exist_ok=True)
    
    model_path = f"export/models/XGBOOST_Model_{model_num}.pkl"
    scaler_path = f"export/scalers/XGBOOST_scaler_{model_num}.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    return model_path, scaler_path


# ==========================================
# SECTION 4: TRADE CODE GENERATOR
# ==========================================
def generate_trade_script(config: dict, model_num: int):
    """สร้างไฟล์ Tradecode.py แบบ Clean Text (ไม่มี Emoji ป้องกัน .bat แครช)"""
    symbol = config['data']['symbol']
    tf = config['data']['timeframe']
    sl_mult = config['strategy']['sl_mult']
    tp_mult = config['strategy']['tp_mult']
    
    indicators_cfg = repr(config['indicators'])
    lags_cfg = repr(config['lags'])

    script_content = f"""import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import talib

# ==============================
# 0) GLOBAL CONFIG
# ==============================
SYMBOL = "{symbol}"
TIMEFRAME = mt5.TIMEFRAME_{tf}
MAGIC_NUMBER = 151515
MODEL_NUM = {model_num}
SL_MULTIPLIER = {sl_mult}
TP_MULTIPLIER = {tp_mult}

# Money Management
EQUITY_PER_LOT = 300 

MODEL_PATH = "models/XGBOOST_Model_{model_num}.pkl"
SCALER_PATH = "scalers/XGBOOST_scaler_{model_num}.pkl"
HISTORY_BARS = 1000 

INDICATOR_CONFIG = {indicators_cfg}
LAGS_CONFIG = {lags_cfg}

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
        ema_name = f"EMA_{{span}}"
        out[ema_name] = out['close'].ewm(span=span, adjust=False).mean()
        out[f"dist_{{ema_name}}"] = (out['close'] - out[ema_name]) / out[ema_name]

    for w in config.get('sma', [20, 50, 400]):
        sma_name = f"SMA_{{w}}"
        out[sma_name] = out['close'].rolling(window=w).mean()
        out[f"dist_{{sma_name}}"] = (out['close'] - out[sma_name]) / out[sma_name]

    p = int(config.get('atr', 48))
    tr = pd.concat([out['high']-out['low'], (out['high']-out['close'].shift(1)).abs(), (out['low']-out['close'].shift(1)).abs()], axis=1).max(axis=1)
    out[f"ATR_{{p}}"] = tr.ewm(span=p, adjust=False).mean()
    
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

    key_features = ['close', 'RSI', 'MACD_hist', f"dist_SMA_{{config.get('sma',[400])[-1]}}", 'body_size']
    for col in key_features:
        if col in out.columns:
            for i in range(1, lag_periods + 1):
                out[f"{{col}}_lag_{{i}}"] = out[col].shift(i)

    return out

def add_lags_custom(df: pd.DataFrame, columns: list, lags: int = 3, stride: int = 1) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            for i in range(1, lags + 1):
                actual_lag = i * stride
                out[f"{{col}}_lag_{{actual_lag}}"] = out[col].shift(actual_lag)
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
    request = {{
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
    }}
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[ERROR] Order Send Failed: {{result.comment}}")
    else:
        print(f"[SUCCESS] Trade Opened! Ticket: {{result.order}} | Lot: {{dynamic_lot}}")

def close_position(position):
    tick = mt5.symbol_info_tick(position.symbol)
    close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    close_price = tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask

    request = {{
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
    }}
    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"[CLOSED] Position {{position.ticket}} Closed Successfully.")
        return True
    else:
        print(f"[ERROR] Failed to close position: {{result.comment}}")
        return False

# ==============================
# 3) MAIN BOT LOGIC
# ==============================
def main():
    if not mt5.initialize():
        print("[ERROR] MT5 Initialize failed")
        return
        
    print(f"[START] Running Trade Bot for {{SYMBOL}}...")
    
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except Exception as e:
        print(f"[ERROR] Error loading model: {{e}}")
        mt5.shutdown()
        return

    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, HISTORY_BARS)
    if rates is None or len(rates) == 0:
        mt5.shutdown()
        return
        
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    df_feat = prepare_features(df, INDICATOR_CONFIG, LAGS_CONFIG['number'])
    atr_col = f"ATR_{{INDICATOR_CONFIG.get('atr', 48)}}"
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

    print(f"[PREDICT] ML Prediction: {{'UPTREND (BUY)' if prediction == 1 else 'DOWNTREND (SELL)'}} | Confidence: {{prob*100:.2f}}%")

    positions = mt5.positions_get(symbol=SYMBOL)
    has_buy = False
    has_sell = False

    if positions:
        for pos in positions:
            if pos.magic != MAGIC_NUMBER:
                continue

            if pos.type == mt5.ORDER_TYPE_BUY:
                if prob < 0.55: 
                    print(f"[ALERT] Confidence dropped to {{prob*100:.2f}}%: Closing existing BUY position...")
                    close_position(pos)
                else:
                    has_buy = True 
            elif pos.type == mt5.ORDER_TYPE_SELL:
                if (1 - prob) < 0.55: 
                    print(f"[ALERT] Confidence dropped to {{(1-prob)*100:.2f}}%: Closing existing SELL position...")
                    close_position(pos)
                else:
                    has_sell = True 

    if not has_buy and not has_sell:
        if prediction == 1 and prob > 0.60:
            sl = current_tick.ask - (current_atr * SL_MULTIPLIER)
            tp = current_tick.ask + (current_atr * TP_MULTIPLIER)
            print(f"[BUY] Signal: STRONG BUY | High Confidence: {{prob*100:.2f}}%")
            open_trade(mt5.ORDER_TYPE_BUY, current_tick.ask, sl, tp)
            
        elif prediction == 0 and prob < 0.40: 
            sl = current_tick.bid + (current_atr * SL_MULTIPLIER)
            tp = current_tick.bid - (current_atr * TP_MULTIPLIER)
            print(f"[SELL] Signal: STRONG SELL | High Confidence: {{(1-prob)*100:.2f}}%")
            open_trade(mt5.ORDER_TYPE_SELL, current_tick.bid, sl, tp)
            
        else:
            print("[HOLD] Waiting... Confidence is in the neutral zone (between 40% - 60%).")
    else:
        print("[WAIT] Already holding the correct position. Waiting for next signal.")

    mt5.shutdown()

if __name__ == "__main__":
    main()
"""
    os.makedirs("export", exist_ok=True)
    file_path = "export/Tradecode.py"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(script_content)
    return file_path