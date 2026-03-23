function logToConsole(message: string, type: 'info' | 'error' | 'warning' = 'info') {
  const consoleDiv = document.getElementById('console-output');
  if (consoleDiv) {
    let colorClass = "";
    if (type === 'error') colorClass = "error-text";
    if (type === 'warning') colorClass = "warning-text";
    
    consoleDiv.innerHTML += `<span class="${colorClass}">> ${message}</span><br>`;
    consoleDiv.scrollTop = consoleDiv.scrollHeight;
  }
}

async function loadLatestConfig() {
  try {
    const response = await fetch("http://127.0.0.1:8000/api/config/latest");
    const result = await response.json();
    
    if (result.status === "success" && result.config) {
      const cfg = result.config;
      
      const setValue = (id: string, val: any) => {
        const el = document.getElementById(id) as HTMLInputElement;
        if (el && val !== undefined) el.value = String(val);
      };

      // 1. Data Source
      setValue('symbol', cfg.data.symbol);
      setValue('timeframe', cfg.data.timeframe);
      
      // จัดการ History Checkbox
      if (cfg.data.history !== "MAX") {
        const checkbox = document.getElementById('use_custom_history') as HTMLInputElement;
        checkbox.checked = true;
        checkbox.dispatchEvent(new Event('change')); // Trigger event เพื่อโชว์กล่องวันที่
        setValue('start_date', cfg.data.history.start);
        setValue('end_date', cfg.data.history.end);
      }

      // 2. Data Split
      setValue('train_size', cfg.split.train * 100);
      setValue('val_size', cfg.split.val * 100);
      document.getElementById('train_size')?.dispatchEvent(new Event('input')); // อัปเดต Test Size

      // 3. Indicators & Labeling
      setValue('future_ema_bars', cfg.labeling.future_bars);
      setValue('atr_period', cfg.indicators.atr);
      setValue('rsi_period', cfg.indicators.rsi);
      setValue('bb_window', cfg.indicators.bb);
      setValue('roc_period', cfg.indicators.roc);
      setValue('cci_period', cfg.indicators.cci);
      setValue('macd_params', cfg.indicators.macd.join(','));
      setValue('ema_spans', cfg.indicators.ema.join(','));
      setValue('sma_windows', cfg.indicators.sma.join(','));

      // 4. Lags & Strategy
      setValue('num_lags', cfg.lags.number);
      setValue('lag_stride', cfg.lags.stride);
      setValue('sl_multiplier', cfg.strategy.sl_mult);
      setValue('tp_multiplier', cfg.strategy.tp_mult);

      logToConsole("🔄 Loaded previous configuration from database.");
    }
  } catch (error) {
    // ถ้าเพิ่งเปิดครั้งแรก หรือยังไม่ได้รัน Backend
    console.log("No previous config found or backend not ready yet.");
  }
}

async function openHistory() {
  const modal = document.getElementById('history-modal') as HTMLDivElement;
  const tbody = document.getElementById('history-tbody') as HTMLTableSectionElement;
  
  modal.style.display = "block"; // เปิดหน้าต่าง
  tbody.innerHTML = "<tr><td colspan='7'>⏳ Loading history...</td></tr>";

  try {
    const response = await fetch("http://127.0.0.1:8000/api/history");
    const result = await response.json();

    if (result.status === "success") {
      const data = result.data;
      if (data.length === 0) {
        tbody.innerHTML = "<tr><td colspan='7'>No history found. Start your first training!</td></tr>";
        return;
      }
      
      // นำข้อมูลที่ได้มาวาดเป็นตาราง
      tbody.innerHTML = data.map((row: any) => {
        const accClass = row.accuracy >= 60 ? 'acc-high' : 'acc-low';
        return `
          <tr>
            <td style="color: #aaa;">#${row.id}</td>
            <td>${row.timestamp}</td>
            <td><strong>${row.symbol}</strong></td>
            <td>${row.timeframe}</td>
            <td>${row.data_rows.toLocaleString()}</td>
            <td class="${accClass}">${row.accuracy.toFixed(2)}%</td>
            <td>${row.f1_score.toFixed(2)}</td>
          </tr>
        `;
      }).join('');
    } else {
      tbody.innerHTML = `<tr><td colspan='7' style="color: red;">Error: ${result.message}</td></tr>`;
    }
  } catch (err) {
    tbody.innerHTML = "<tr><td colspan='7' style='color: red;'>❌ Failed to connect to Python Backend.</td></tr>";
  }
}

function gatherConfig() {
  const getValue = (id: string) => (document.getElementById(id) as HTMLInputElement).value;
  const isCustomHistory = (document.getElementById('use_custom_history') as HTMLInputElement).checked;
  
  let historyConfig: any = "MAX";
  if (isCustomHistory) {
    const start = getValue('start_date');
    const end = getValue('end_date');
    if (start && end) historyConfig = { start: start, end: end };
  }

  return {
    data: {
      symbol: getValue('symbol').trim(),
      timeframe: getValue('timeframe'),
      history: historyConfig 
    },
    labeling: {
      future_bars: Number(getValue('future_ema_bars'))
    },
    // 🌟 สิ่งที่เพิ่มเข้ามา: สัดส่วนข้อมูล (หาร 100 เพื่อทำเป็นทศนิยมส่งให้ Python)
    split: {
      train: Number(getValue('train_size')) / 100,
      val: Number(getValue('val_size')) / 100
    },
    indicators: {
      atr: Number(getValue('atr_period')),
      rsi: Number(getValue('rsi_period')),
      bb: Number(getValue('bb_window')),
      roc: Number(getValue('roc_period')),
      cci: Number(getValue('cci_period')),
      macd: getValue('macd_params').split(',').map(Number),
      ema: getValue('ema_spans').split(',').map(Number),
      sma: getValue('sma_windows').split(',').map(Number)
    },
    lags: {
      number: Number(getValue('num_lags')),
      stride: Number(getValue('lag_stride'))
    },
    strategy: {
      sl_mult: Number(getValue('sl_multiplier')),
      tp_mult: Number(getValue('tp_multiplier'))
    }
  };
}

let pollingInterval: number;

async function startTraining() {
  const config = gatherConfig();
  const trainBtn = document.getElementById('btn-train') as HTMLButtonElement;
  const exportBtn = document.getElementById('btn-export') as HTMLButtonElement;
  
  // Elements ของ Progress Bar
  const progressContainer = document.getElementById('progress-container') as HTMLDivElement;
  const progressFill = document.getElementById('progress-fill') as HTMLDivElement;
  const progressText = document.getElementById('progress-text') as HTMLSpanElement;
  const progressPercent = document.getElementById('progress-percent') as HTMLSpanElement;

  if (!config.data.symbol) {
    logToConsole("Error: Please enter a valid Symbol.", 'error');
    return;
  }

  logToConsole(`Sending job to Engine for ${config.data.symbol}...`);
  trainBtn.disabled = true;
  exportBtn.disabled = true;
  trainBtn.innerText = "⏳ Initializing...";

  // 🌟 เปิดโชว์ Progress Bar เริ่มที่ 0%
  progressContainer.style.display = "block";
  progressFill.style.width = "0%";
  progressPercent.innerText = "0%";
  progressText.innerText = "Connecting to Engine...";

  try {
    // 1. ส่งคำสั่งเริ่ม Train (API จะตอบกลับมาทันทีพร้อม Task ID)
    const response = await fetch("http://127.0.0.1:8000/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config)
    });
    
    const result = await response.json();
    if (result.status === "error") {
      throw new Error(result.message);
    }

    const taskId = result.task_id;
    trainBtn.innerText = "⏳ Training in progress...";

    // 2. ตั้งเวลาวนลูปเช็คสถานะทุกๆ 800ms
    pollingInterval = window.setInterval(async () => {
      try {
        const statusRes = await fetch(`http://127.0.0.1:8000/api/train/status/${taskId}`);
        const statusData = await statusRes.json();

        // 🌟 อัปเดตหลอดโหลด
        progressFill.style.width = `${statusData.progress}%`;
        progressPercent.innerText = `${statusData.progress}%`;
        progressText.innerText = statusData.message;

        // เช็คว่างานเสร็จหรือยัง
        if (statusData.status === "completed") {
          clearInterval(pollingInterval); // หยุดวนลูป
          
          const metrics = statusData.result.metrics;
          if (statusData.result.warning) logToConsole(`⚠️ Warning: ${statusData.result.warning}`, 'warning');
          
          logToConsole(`✅ Success! Trained on ${metrics.data_rows.toLocaleString()} bars.`);
          logToConsole(`📊 Accuracy: ${metrics.accuracy}%, F1: ${metrics.f1}`);
          
          trainBtn.disabled = false;
          trainBtn.innerText = "▶️ Retrain Model";
          exportBtn.disabled = false;

          // ซ่อนหลอดโหลดหลังจากผ่านไป 3 วินาที
          setTimeout(() => { progressContainer.style.display = "none"; }, 3000);
          
        } else if (statusData.status === "error") {
          clearInterval(pollingInterval);
          logToConsole(`❌ Error: ${statusData.message}`, 'error');
          resetUI();
        }
      } catch (err) {
        // Error ระหว่างดึงสถานะ
      }
    }, 800); // เช็คทุก 0.8 วินาที

  } catch (err: any) {
    logToConsole(`❌ Connection failed: ${err.message}`, 'error');
    resetUI();
  }

  function resetUI() {
    trainBtn.disabled = false;
    trainBtn.innerText = "▶️ Retrain Model";
    progressContainer.style.display = "none";
  }
}

async function exportModel() {
  logToConsole("Opening export folder...");
  try {
    // 🌟 เรียก API ให้ Python สั่ง Windows เด้งโฟลเดอร์ขึ้นมา
    const response = await fetch("http://127.0.0.1:8000/api/open_folder");
    const result = await response.json();
    
    if (result.status === "success") {
      logToConsole("✅ Export folder opened!");
    } else {
      logToConsole(`❌ Failed to open folder: ${result.message}`, "error");
    }
  } catch (err) {
    logToConsole("❌ Connection failed.", "error");
  }
}

// ผูก Event Listener ทันทีที่โหลดหน้าจอเสร็จ
window.addEventListener("DOMContentLoaded", () => {
  const customHistoryCheckbox = document.getElementById('use_custom_history') as HTMLInputElement;
  const dateRangeContainer = document.getElementById('custom_date_range') as HTMLDivElement;

  const trainInput = document.getElementById('train_size') as HTMLInputElement;
  const valInput = document.getElementById('val_size') as HTMLInputElement;
  const testInput = document.getElementById('test_size') as HTMLInputElement;

  function updateTestSize() {
    const train = Number(trainInput.value);
    const val = Number(valInput.value);
    const test = 100 - train - val;
    testInput.value = test.toString();
  }

  trainInput.addEventListener('input', updateTestSize);
  valInput.addEventListener('input', updateTestSize);

  // Toggle การเปิด/ปิด ช่องกรอกวันที่
  customHistoryCheckbox.addEventListener('change', (e) => {
    if ((e.target as HTMLInputElement).checked) {
      dateRangeContainer.style.display = 'grid'; // โชว์กล่องวันที่
      
      // ตั้งค่า Default ให้เป็นวันปัจจุบันสำหรับ End Date
      const today = new Date().toISOString().split('T')[0];
      (document.getElementById('end_date') as HTMLInputElement).value = today;
    } else {
      dateRangeContainer.style.display = 'none'; // ซ่อนกล่องวันที่
    }
  });

  document.getElementById('btn-train')?.addEventListener('click', startTraining);
  document.getElementById('btn-export')?.addEventListener('click', exportModel);
  document.getElementById('btn-history')?.addEventListener('click', openHistory);
  document.getElementById('close-history')?.addEventListener('click', () => {
    (document.getElementById('history-modal') as HTMLDivElement).style.display = "none";
  });

  // (ตัวเลือกเสริม) กดพื้นที่ว่างนอกหน้าต่างเพื่อปิด
  window.addEventListener('click', (event) => {
    const modal = document.getElementById('history-modal') as HTMLDivElement; // 🌟 เติม type เข้าไปตรงนี้
    if (modal && event.target === modal) { // 🌟 เพิ่มการเช็คว่า modal ไม่ใช่ null
      modal.style.display = "none";
    }
  });
  loadLatestConfig();
  
});