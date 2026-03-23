import sqlite3
import json
from datetime import datetime
import os

DB_PATH = "app_database.db"

def init_db():
    """สร้างตาราง 2 ตัว: สำหรับเก็บ Config ล่าสุด และ เก็บประวัติการเทรน"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. ตารางเก็บ Config ปัจจุบัน (ใช้ Key-Value เก็บเป็น JSON ง่ายที่สุด)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    
    # 2. ตารางเก็บประวัติการ Train และ Metrics
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            symbol TEXT,
            timeframe TEXT,
            accuracy REAL,
            f1_score REAL,
            precision REAL,
            recall REAL,
            data_rows INTEGER,
            config_json TEXT,
            model_num INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def save_latest_config(config_dict):
    """เซฟ Config ล่าสุดลงตาราง app_settings"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO app_settings (key, value)
        VALUES (?, ?)
    ''', ("latest_config", json.dumps(config_dict)))
    conn.commit()
    conn.close()

def get_latest_config():
    """ดึง Config ล่าสุดกลับมา"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT value FROM app_settings WHERE key = ?', ("latest_config",))
    row = cursor.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None

def save_training_history(metrics, config_dict, model_num):
    """บันทึกประวัติการเทรนพร้อมความแม่นยำ"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO training_history (
            timestamp, symbol, timeframe, accuracy, f1_score, precision, recall, data_rows, config_json, model_num
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        now,
        config_dict['data']['symbol'],
        config_dict['data']['timeframe'],
        metrics['accuracy'],
        metrics.get('f1', 0),
        metrics.get('precision', 0),
        metrics.get('recall', 0),
        metrics['data_rows'],
        json.dumps(config_dict),
        model_num
    ))
    conn.commit()
    conn.close()

def get_training_history():
    """ดึงประวัติการเทรนทั้งหมดจากฐานข้อมูล เรียงจากใหม่ไปเก่า"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row # ทำให้ผลลัพธ์ที่ได้เป็น Dictionary อัตโนมัติ
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, timestamp, symbol, timeframe, accuracy, f1_score, data_rows, model_num
        FROM training_history
        ORDER BY id DESC
    ''')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]