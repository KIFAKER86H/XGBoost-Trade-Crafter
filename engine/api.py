import uuid
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import datetime # เพิ่ม datetime สำหรับแสดงเวลาใน Log
from db import init_db, save_latest_config, get_latest_config, save_training_history, get_training_history

# นำเข้าเครื่องยนต์ที่เราเขียนไว้
from ai_core import (
    fetch_mt5_data, 
    prepare_features_v2, 
    create_labels, 
    prepare_for_training, 
    train_xgboost, 
    save_model_artifacts, 
    generate_trade_script
)


import os

app = FastAPI(title="XGBoost Trade Crafter API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
init_db()
class TrainConfig(BaseModel):
    data: dict
    labeling: dict
    indicators: dict
    lags: dict
    strategy: dict
    split: dict

# ฟังก์ชันช่วยพิมพ์ Log ให้สวยงาม
def api_log(message: str, level: str = "INFO"):
    time_now = datetime.datetime.now().strftime("%H:%M:%S")
    if level == "INFO":
        print(f"[{time_now}] 🔵 [INFO] {message}")
    elif level == "SUCCESS":
        print(f"[{time_now}] 🟢 [SUCCESS] {message}")
    elif level == "WARNING":
        print(f"[{time_now}] 🟡 [WARNING] {message}")
    elif level == "ERROR":
        print(f"[{time_now}] 🔴 [ERROR] {message}")
    print("-" * 50) # เส้นคั่นให้อ่านง่าย

@app.get("/api/config/latest")
def get_config():
    """API สำหรับให้ UI มาขอดึงค่าเริ่มต้นตอนเปิดโปรแกรม"""
    config = get_latest_config()
    if config:
        return {"status": "success", "config": config}
    return {"status": "empty"}

# 🌟 สร้าง Dictionary สำหรับเก็บเปอร์เซ็นต์การทำงานของแต่ละรอบ
training_status = {}

def background_training_job(task_id: str, cfg: dict):
    """ฟังก์ชันที่จะถูกรันเป็น Background เพื่อไม่ให้ API ค้าง"""
    try:
        symbol = cfg['data']['symbol']
        timeframe = cfg['data']['timeframe']
        
        # ฟังก์ชันช่วยอัปเดตเปอร์เซ็นต์
        def update_progress(prog: int, msg: str):
            training_status[task_id]['progress'] = prog
            training_status[task_id]['message'] = msg
            api_log(f"Task [{task_id[:4]}] | {msg}", "INFO")

        update_progress(10, "Step 1: Fetching data from MT5...")
        df_raw, warning = fetch_mt5_data(symbol, timeframe, cfg['data']['history'])
        api_log(f"Fetch [{symbol}, {timeframe}] | FINISHED!", "SUCCESS")
        
        update_progress(30, "Step 2: Engineering Features & Lags...")
        lags = cfg['lags']['number']
        stride = cfg['lags']['stride']
        df_features = prepare_features_v2(df_raw, config=cfg['indicators'], lags=lags, stride=stride)
        
        update_progress(50, "Step 3: Creating Future Labels...")
        future_bars = cfg['labeling']['future_bars']
        df_labeled = create_labels(df_features, future_bars=future_bars)
        
        update_progress(65, "Step 4: Preparing Train/Test splits...")
        train_size = cfg['split']['train']
        val_size = cfg['split']['val']
        
        # 🌟 ทำให้การลบคอลัมน์เป็นแบบ Dynamic ตามค่าที่ผู้ใช้ตั้ง
        future_col = f"EMA{future_bars}"
        future_col_f = f"EMA{future_bars}_Future"
        cols_to_drop = [future_col, future_col_f, 'EMA24', 'EMA24_Future', 'EMA48', 'EMA48_Future']
        
        df_model = df_labeled.drop(columns=[col for col in cols_to_drop if col in df_labeled.columns], errors='ignore')
        
        # 🌟 [ส่วนที่ต้องเพิ่มกลับมา] เรียกใช้ฟังก์ชันเพื่อสร้าง X_train, y_train และ scaler
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_for_training(
            df_model, target_col='Label', train_size=train_size, val_size=val_size
        )
        
        update_progress(80, "Step 5: Training XGBoost Model (GridSearch may take a while)...")
        model_num = 999 
        model, metrics = train_xgboost(X_train, y_train, X_val, y_val, model_num=model_num)
        
        update_progress(80, "Step 5: Training XGBoost Model (GridSearch may take a while)...")
        model_num = 999 
        model, metrics = train_xgboost(X_train, y_train, X_val, y_val, model_num=model_num)
        
        update_progress(95, "Step 6: Saving Models & DB...")
        save_model_artifacts(model, scaler, model_num)
        generate_trade_script(cfg, model_num)
        
        metrics['data_rows'] = len(df_raw)
        save_latest_config(cfg)
        save_training_history(metrics, cfg, model_num)

        # 🌟 แจ้งสถานะ 100% และส่งผลลัพธ์กลับ
        training_status[task_id]['progress'] = 100
        training_status[task_id]['status'] = 'completed'
        training_status[task_id]['message'] = 'Training completed successfully!'
        training_status[task_id]['result'] = {
            "warning": warning,
            "metrics": {
                "accuracy": round(metrics['accuracy'], 2),
                "f1": round(metrics['f1'], 2),
                "data_rows": len(df_raw)
            }
        }
        api_log(f"Task [{task_id[:4]}] | 🎉 FINISHED!", "SUCCESS")

    except Exception as e:
        training_status[task_id]['status'] = 'error'
        training_status[task_id]['message'] = str(e)
        api_log(f"Task [{task_id[:4]}] | ERROR: {str(e)}", "ERROR")


@app.post("/api/train")
def run_training_pipeline(config: TrainConfig, bg_tasks: BackgroundTasks):
    """API นี้แค่รับคำสั่ง แล้วโยนงานไปทำ Background ทันที"""
    task_id = str(uuid.uuid4()) # สุ่มรหัสงาน
    training_status[task_id] = {
        "status": "running", "progress": 0, "message": "Initializing...", "result": None
    }
    # โยนงานให้ทำเบื้องหลัง
    bg_tasks.add_task(background_training_job, task_id, config.dict())
    
    # ตอบกลับ UI ทันทีพร้อมรหัสงาน
    return {"status": "success", "task_id": task_id}

@app.get("/api/train/status/{task_id}")
def get_training_status(task_id: str):
    """API สำหรับให้ UI คอยยิงมาเช็คสถานะ / เปอร์เซ็นต์"""
    status = training_status.get(task_id)
    if not status:
        return {"status": "error", "message": "Task not found"}
    return status

@app.get("/api/open_folder")
def open_folder():
    try:
        export_path = os.path.abspath("export")
        os.makedirs(export_path, exist_ok=True) 
        os.startfile(export_path)
        return {"status": "success"}
    except Exception as e:
        api_log(f"Failed to open folder: {str(e)}", "ERROR")
        return {"status": "error", "message": str(e)}
    
@app.get("/api/history")
def get_history():
    """API สำหรับให้ UI มาขอดึงประวัติการเทรนทั้งหมด"""
    try:
        history = get_training_history()
        return {"status": "success", "data": history}
    except Exception as e:
        api_log(f"Failed to fetch history: {str(e)}", "ERROR")
        return {"status": "error", "message": str(e)}