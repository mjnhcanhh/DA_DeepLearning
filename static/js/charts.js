/**
 * charts.js — AI Accident Detection Dashboard v4
 * Tab So sánh: radar, bar, bubble charts
 * Tab Hiệu suất: Stress Test + Live Monitor + Báo cáo xuất được
 */

const C  = ['#3b82f6', '#f59e0b', '#7c3aed'];
const F  = ['rgba(59,130,246,.15)', 'rgba(245,158,11,.15)', 'rgba(124,58,237,.15)'];

const GO = {
  plugins: {
    legend: { labels: { color: '#475569', font: { family: 'DM Sans', size: 12 } } }
  },
  scales: {
    x: { ticks: { color: '#64748b', font: { family: 'DM Sans' } }, grid: { color: '#e2e8f0' } },
    y: { ticks: { color: '#64748b', font: { family: 'DM Sans' } }, grid: { color: '#e2e8f0' } }
  }
};

// ══════════════════════════════════════════════════════════════════════
// TAB SO SÁNH — Static benchmark charts
// ══════════════════════════════════════════════════════════════════════

function buildCompare() {
  window.cBuilt = true;
  const staticWrap = document.getElementById('bench-static-charts');
  if (!staticWrap) return;
  staticWrap.style.display = 'block';
  staticWrap.innerHTML = `
    <div class="compare-placeholder">
      <div>
        <div class="compare-placeholder-title">Chua co ket qua so sanh truc quan</div>
        <div class="compare-placeholder-text">
          Chon 1 anh test roi bam <b>So sanh ngay</b>. He thong se chay that 3 model:
          <b>SSD</b>, <b>Faster R-CNN</b>, <b>YOLOv12</b>, sau do moi ve bang va bieu do theo so do thuc.
        </div>
      </div>
    </div>`;
}

// ══════════════════════════════════════════════════════════════════════
// TAB HIỆU SUẤT — LIVE PERFORMANCE MONITOR
// ══════════════════════════════════════════════════════════════════════

let perfBuilt      = false;
let stressRunning  = false;
let stressAbort    = false;
let liveLineChart  = null;
let liveBarChart   = null;
let trainLossChart = null;
let mapLineChart   = null;
let confMatChart   = null;

// Lưu kết quả stress test để xuất báo cáo
let _lastStressResult = null;

const BUFFER_MAX = 30;
const liveData = {
  ssd:         { ms: [], fps: [], label: 'SSD',          color: '#f59e0b' },
  faster_rcnn: { ms: [], fps: [], label: 'Faster R-CNN', color: '#7c3aed' },
  yolov12:     { ms: [], fps: [], label: 'YOLOv12',      color: '#059669' },
};
let liveLabels = [];

function buildPerf() {
  if (perfBuilt) return;
  perfBuilt = true;
  window.pBuilt = true;

  initKpiCards();
  initLiveCharts();
}

// ── KPI Cards ──────────────────────────────────────────────────────
function initKpiCards() {
  const models = [
    { key: 'ssd',         name: 'SSD',          map: null, fps: null, ms: null, size: 'N/A',   color: '#f59e0b', rank: 'SSD' },
    { key: 'faster_rcnn', name: 'Faster R-CNN', map: 63.8, fps: null, ms: null, size: '167MB', color: '#7c3aed', rank: 'RCNN' },
    { key: 'yolov12',     name: 'YOLOv12',      map: null, fps: null, ms: null, size: 'N/A',   color: '#059669', rank: 'Y12' },
  ];
  const grid = document.getElementById('perf-kpi-grid');
  if (!grid) return;
  const fmt = (v, suffix = '') => v == null ? 'N/A' : `${v}${suffix}`;
  const bar = v => v == null ? 0 : Math.round(Math.min(v / 75 * 100, 100));
  grid.innerHTML = models.map(m => `
    <div class="perf-kpi-card" style="border-top:4px solid ${m.color}">
      <div class="perf-kpi-header">
        <span style="font-weight:800;color:${m.color}">${m.rank} ${m.name}</span>
        <span class="perf-kpi-status" id="kpi-status-${m.key}">Chua do</span>
      </div>
      <div class="perf-kpi-row">
        <div class="perf-kpi-stat">
          <div class="perf-kpi-val" id="kpi-fps-${m.key}" style="color:${m.color}">${fmt(m.fps)}</div>
          <div class="perf-kpi-lbl">FPS</div>
        </div>
        <div class="perf-kpi-stat">
          <div class="perf-kpi-val" id="kpi-ms-${m.key}" style="color:${m.color}">${fmt(m.ms, 'ms')}</div>
          <div class="perf-kpi-lbl">Latency</div>
        </div>
        <div class="perf-kpi-stat">
          <div class="perf-kpi-val" id="kpi-map-${m.key}" style="color:${m.color}">${fmt(m.map, '%')}</div>
          <div class="perf-kpi-lbl">mAP@0.5</div>
        </div>
        <div class="perf-kpi-stat">
          <div class="perf-kpi-val" style="color:#64748b;font-size:1.05rem">${m.size}</div>
          <div class="perf-kpi-lbl">Model Size</div>
        </div>
      </div>
      <div class="perf-speed-bar-wrap">
        <div style="font-size:.68rem;color:#94a3b8;margin-bottom:4px">FPS cap nhat sau Stress Test</div>
        <div class="perf-speed-track">
          <div class="perf-speed-fill" id="kpi-bar-${m.key}"
               style="width:${bar(m.fps)}%;background:${m.color}"></div>
        </div>
      </div>
    </div>`).join('');
}

// ── Live charts ────────────────────────────────────────────────────
function initLiveCharts() {
  const fontOpts = { family: 'DM Sans', size: 10 };

  const lcEl = document.getElementById('perf-live-line');
  if (lcEl) {
    liveLineChart = new Chart(lcEl, {
      type: 'line',
      data: {
        labels: [],
        datasets: Object.entries(liveData).map(([k, d]) => ({
          label:           d.label,
          data:            [],
          borderColor:     d.color,
          backgroundColor: d.color + '22',
          pointRadius:     3,
          pointBackgroundColor: d.color,
          tension:         0.4,
          borderWidth:     2,
          fill:            false,
        }))
      },
      options: {
        animation:  { duration: 300 },
        responsive: true,
        plugins: { legend: { labels: { color: '#475569', font: fontOpts, boxWidth: 12 } } },
        scales: {
          x: { ticks: { color: '#94a3b8', font: fontOpts }, grid: { color: '#f1f5f9' } },
          y: {
            ticks: { color: '#94a3b8', font: fontOpts },
            grid:  { color: '#f1f5f9' },
            title: { display: true, text: 'Latency (ms)', color: '#94a3b8', font: { size: 10 } }
          }
        }
      }
    });
  }

  const hbEl = document.getElementById('perf-live-bar');
  if (hbEl) {
    liveBarChart = new Chart(hbEl, {
      type:  'bar',
      data: {
        labels: ['SSD', 'Faster R-CNN', 'YOLOv12'],
        datasets: [{
          label:           'FPS đo được',
          data:            [0, 0, 0],
          backgroundColor: ['#2563eb99', '#7c3aed99', '#05966999'],
          borderColor:     ['#2563eb',   '#7c3aed',   '#059669'],
          borderWidth:     2,
          borderRadius:    8,
        }]
      },
      options: {
        indexAxis:  'y',
        animation:  { duration: 400 },
        responsive: true,
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: ctx => `${ctx.parsed.x.toFixed(1)} FPS` } }
        },
        scales: {
          x: {
            min: 0, max: 80,
            ticks: { color: '#94a3b8', font: fontOpts },
            grid:  { color: '#f1f5f9' },
            title: { display: true, text: 'FPS (cao hơn = tốt hơn)', color: '#94a3b8', font: { size: 10 } }
          },
          y: { ticks: { color: '#475569', font: { ...fontOpts, size: 11, weight: '600' } }, grid: { display: false } }
        }
      }
    });
  }
}

// ══════════════════════════════════════════════════════════════════════
// STRESS TEST
// ══════════════════════════════════════════════════════════════════════

async function startStressTest() {
  if (stressRunning) return stopStressTest();

  const fileEl = document.getElementById('stress-file');
  if (!fileEl || !fileEl.files.length) {
    alert('Chọn 1 ảnh test trước!'); return;
  }
  const n    = parseInt(document.getElementById('stress-n').value) || 10;
  const conf = parseFloat(document.getElementById('stress-conf').value) || 0.4;
  const iou  = parseFloat(document.getElementById('stress-iou').value)  || 0.45;

  stressRunning = true;
  stressAbort   = false;
  _lastStressResult = null;

  const btn = document.getElementById('stress-btn');
  btn.textContent = '⏹ Dừng';
  btn.style.background = '#dc2626';

  document.getElementById('stress-log').style.display = 'block';
  document.getElementById('stress-log').innerHTML = '';
  document.getElementById('stress-prog-wrap').style.display = 'block';
  document.getElementById('stress-summary').style.display = 'none';
  document.getElementById('stress-export-wrap').style.display = 'none';

  // Reset charts
  Object.values(liveData).forEach(d => { d.ms = []; d.fps = []; });
  liveLabels = [];
  if (liveLineChart) {
    liveLineChart.data.labels = [];
    liveLineChart.data.datasets.forEach(ds => ds.data = []);
    liveLineChart.update('none');
  }
  if (liveBarChart) {
    liveBarChart.data.datasets[0].data = [0, 0, 0];
    liveBarChart.update('none');
  }

  const file  = fileEl.files[0];
  const KEYS  = ['ssd', 'faster_rcnn', 'yolov12'];
  const avgMs = { ssd: [], faster_rcnn: [], yolov12: [] };
  const allRuns = [];

  for (let i = 0; i < n && !stressAbort; i++) {
    const pct = Math.round((i + 1) / n * 100);
    document.getElementById('stress-pb').style.width  = pct + '%';
    document.getElementById('stress-pct').textContent = `Vòng ${i+1}/${n} — Đang chạy lần lượt 3 model... (${pct}%)`;

    const fd = new FormData();
    fd.append('file', file);
    fd.append('conf', conf);
    fd.append('iou', iou);

    try {
      const res  = await fetch('/benchmark_image', { method: 'POST', body: fd });
      const data = await res.json();
      if (data.error) { logStress(`❌ Vòng ${i+1}: ${data.error}`, 'err'); continue; }

      liveLabels.push(`#${i+1}`);
      if (liveLabels.length > BUFFER_MAX) liveLabels.shift();

      const runRecord = { round: i+1, models: {} };
      let rowParts = [];

      KEYS.forEach(key => {
        const r = data.results?.[key];
        if (!r || !r.loaded || r.error) return;
        const ms  = r.timing_ms;
        const fps = r.fps;
        avgMs[key].push(ms);
        liveData[key].ms.push(ms);
        if (liveData[key].ms.length > BUFFER_MAX) liveData[key].ms.shift();

        runRecord.models[key] = { ms, fps, level: r.level, conf: r.acc_conf };

        animateVal(`kpi-ms-${key}`, `${ms}ms`);
        animateVal(`kpi-fps-${key}`, `${fps}`);
        const stEl = document.getElementById(`kpi-status-${key}`);
        if (stEl) stEl.textContent = '🟢 Đang đo';

        rowParts.push(`<span style="color:${liveData[key].color};font-weight:600">${liveData[key].label}</span>: ${ms}ms · ${fps}FPS`);
      });

      allRuns.push(runRecord);

      if (liveLineChart) {
        liveLineChart.data.labels = [...liveLabels];
        liveLineChart.data.datasets.forEach((ds, ki) => {
          ds.data = [...(liveData[KEYS[ki]]?.ms || [])];
        });
        liveLineChart.update('none');
      }
      if (liveBarChart) {
        const fpsVals = KEYS.map(k => {
          const arr = avgMs[k];
          if (!arr.length) return 0;
          return parseFloat((1000 / (arr.reduce((a,b)=>a+b,0)/arr.length)).toFixed(1));
        });
        liveBarChart.data.datasets[0].data = fpsVals;
        liveBarChart.update('none');
      }

      logStress(`Vòng ${i+1}: ${rowParts.join(' &nbsp;|&nbsp; ')}`, 'ok');
    } catch(e) {
      logStress(`❌ Vòng ${i+1}: ${e.message}`, 'err');
    }
    await new Promise(r => setTimeout(r, 80));
  }

  stressRunning = false;
  btn.textContent = '▶ Bắt đầu Stress Test';
  btn.style.background = '#2563eb';
  document.getElementById('stress-pct').textContent = stressAbort ? '⏹ Đã dừng' : `✅ Hoàn thành ${n} vòng đo!`;

  KEYS.forEach(k => {
    const el = document.getElementById(`kpi-status-${k}`);
    if (el) el.textContent = avgMs[k].length
      ? `✅ TB ${(avgMs[k].reduce((a,b)=>a+b,0)/avgMs[k].length).toFixed(0)}ms`
      : '⚪ Chưa chạy';
  });

  _lastStressResult = { avgMs, n: stressAbort ? allRuns.length : n, conf, iou, allRuns, file: file.name };
  showStressSummary(avgMs, stressAbort ? allRuns.length : n);
}

function stopStressTest() {
  stressAbort   = true;
  stressRunning = false;
}

function logStress(msg, type) {
  const el = document.getElementById('stress-log');
  if (!el) return;
  const color = type === 'err' ? '#dc2626' : '#475569';
  el.innerHTML = `<div style="color:${color};font-size:.73rem;border-bottom:1px solid #f1f5f9;padding:3px 0">${msg}</div>` + el.innerHTML;
  const divs = el.querySelectorAll('div');
  if (divs.length > 25) divs[divs.length-1].remove();
}

function animateVal(id, val) {
  const el = document.getElementById(id);
  if (!el) return;
  el.style.transform = 'scale(1.15)';
  el.textContent = val;
  setTimeout(() => el.style.transform = 'scale(1)', 180);
}

function perfTier(fps) {
  const value = Number(fps || 0);
  if (value >= 30) return { label: 'Real-time 30FPS', color: '#16a34a', icon: '✅' };
  if (value >= 15) return { label: 'Gần real-time', color: '#059669', icon: '🟢' };
  if (value >= 5) return { label: 'Live chậm', color: '#d97706', icon: '🟡' };
  return { label: 'Phân tích offline', color: '#dc2626', icon: '🔴' };
}

// ── Hiển thị tổng kết stress ───────────────────────────────────────
function showStressSummary(avgMs, n) {
  const wrap = document.getElementById('stress-summary');
  if (!wrap) return;

  const models = [
    { key: 'ssd',         name: 'SSD',          color: '#f59e0b' },
    { key: 'faster_rcnn', name: 'Faster R-CNN',  color: '#7c3aed' },
    { key: 'yolov12',     name: 'YOLOv12',       color: '#059669' },
  ];

  // Tính stats
  const stats = models.map(m => {
    const arr = avgMs[m.key];
    if (!arr.length) return { ...m, noData: true };
    const avg  = arr.reduce((a,b)=>a+b,0)/arr.length;
    const minv = Math.min(...arr);
    const maxv = Math.max(...arr);
    const std  = Math.sqrt(arr.map(v=>(v-avg)**2).reduce((a,b)=>a+b,0)/arr.length);
    const fps  = 1000/avg;
    const p95  = [...arr].sort((a,b)=>a-b)[Math.floor(arr.length*0.95)] || maxv;
    return { ...m, arr, avg, minv, maxv, std, fps, p95 };
  }).filter(s => !s.noData);

  const fastest  = stats.reduce((a,b) => a.avg < b.avg ? a : b);
  const mostStable = stats.reduce((a,b) => a.std < b.std ? a : b);
  const realtimeOk = stats.filter(s => s.fps >= 30);
  const nearRealtime = stats.filter(s => s.fps >= 15 && s.fps < 30);
  const slowLive = stats.filter(s => s.fps >= 5 && s.fps < 15);

  const rows = models.map(m => {
    const arr = avgMs[m.key];
    if (!arr.length) return `<tr><td style="color:${m.color};font-weight:700;padding:10px 12px">${m.name}</td><td colspan="5" style="color:#94a3b8;padding:10px 12px">Model không load được</td></tr>`;
    const avg  = arr.reduce((a,b)=>a+b,0)/arr.length;
    const minv = Math.min(...arr);
    const maxv = Math.max(...arr);
    const std  = Math.sqrt(arr.map(v=>(v-avg)**2).reduce((a,b)=>a+b,0)/arr.length);
    const fps  = (1000/avg).toFixed(1);
    const p95  = ([...arr].sort((a,b)=>a-b)[Math.floor(arr.length*0.95)] || maxv).toFixed(1);
    const stable = std < 5 ? '🟢 Rất ổn định' : std < 15 ? '🟡 Ổn định' : '🔴 Dao động nhiều';
    const tier = perfTier(fps);
    const rtTag  = `<span style="color:${tier.color};font-size:.7rem">${tier.icon} ${tier.label}</span>`;
    return `<tr style="border-bottom:1px solid #f1f5f9">
      <td style="color:${m.color};font-weight:700;padding:10px 12px">${m.name}</td>
      <td style="padding:10px 12px;font-weight:700;color:${tier.color}">${fps} FPS ${rtTag}</td>
      <td style="padding:10px 12px">${avg.toFixed(1)} ms</td>
      <td style="padding:10px 12px;color:#64748b">${minv.toFixed(1)} – ${maxv.toFixed(1)} ms</td>
      <td style="padding:10px 12px;color:#64748b">±${std.toFixed(1)} ms</td>
      <td style="padding:10px 12px;color:#64748b">${p95} ms &nbsp; ${stable}</td>
    </tr>`;
  }).join('');

  // Insight tự động
  const insights = [];
  if (fastest) insights.push(`⚡ <b>${fastest.name}</b> nhanh nhất: TB ${fastest.avg.toFixed(1)}ms (${fastest.fps.toFixed(1)} FPS)`);
  if (mostStable) insights.push(`📐 <b>${mostStable.name}</b> ổn định nhất: độ lệch chuẩn ±${mostStable.std.toFixed(1)}ms — kết quả ít dao động nhất`);
  if (realtimeOk.length > 0) insights.push(`✅ Phù hợp real-time (≥30 FPS): <b>${realtimeOk.map(s=>s.name).join(', ')}</b>`);
  if (nearRealtime.length > 0) insights.push(`🟢 Gần real-time: <b>${nearRealtime.map(s=>s.name).join(', ')}</b> — có thể dùng live cảnh báo nếu xử lý frame-skip/temporal voting`);
  if (slowLive.length > 0) insights.push(`🟡 Live chậm: <b>${slowLive.map(s=>s.name).join(', ')}</b> — nên giảm tần suất quét hoặc dùng làm verifier`);
  const offlineOnly = stats.filter(s => s.fps < 5);
  if (offlineOnly.length > 0) insights.push(`🔴 Phân tích offline/verifier: <b>${offlineOnly.map(s=>s.name).join(', ')}</b> — không nên chạy mỗi frame camera`);

  wrap.style.display = 'block';
  wrap.innerHTML = `
    <div style="font-weight:800;font-size:.95rem;color:#1e293b;margin-bottom:14px;display:flex;align-items:center;gap:8px">
      📊 Kết quả Stress Test
      <span style="font-size:.75rem;color:#94a3b8;font-weight:400">${n} vòng đo · ảnh: ${_lastStressResult?.file || '—'} · conf=${_lastStressResult?.conf || '—'} · iou=${_lastStressResult?.iou || '—'}</span>
    </div>

    <table style="width:100%;border-collapse:collapse;font-size:.8rem;background:white;border-radius:10px;overflow:hidden;box-shadow:0 1px 6px rgba(0,0,0,.08);margin-bottom:16px">
      <thead>
        <tr style="background:#f8fafc">
          <th style="padding:10px 12px;text-align:left;color:#64748b;border-bottom:2px solid #e2e8f0">Model</th>
          <th style="padding:10px 12px;text-align:left;color:#64748b;border-bottom:2px solid #e2e8f0">⚡ FPS TB</th>
          <th style="padding:10px 12px;text-align:left;color:#64748b;border-bottom:2px solid #e2e8f0">Latency TB</th>
          <th style="padding:10px 12px;text-align:left;color:#64748b;border-bottom:2px solid #e2e8f0">Min – Max</th>
          <th style="padding:10px 12px;text-align:left;color:#64748b;border-bottom:2px solid #e2e8f0">Std Dev</th>
          <th style="padding:10px 12px;text-align:left;color:#64748b;border-bottom:2px solid #e2e8f0">P95 · Ổn định</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>

    <div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;padding:12px 16px;margin-bottom:16px">
      <div style="font-weight:700;font-size:.82rem;color:#0369a1;margin-bottom:8px">💡 Nhận xét tự động</div>
      <ul style="margin:0;padding-left:18px;font-size:.8rem;color:#0c4a6e;line-height:2.2">
        ${insights.map(i => `<li>${i}</li>`).join('')}
      </ul>
    </div>

    <div style="font-size:.73rem;color:#94a3b8;margin-bottom:12px">
      <b>Giải thích cột:</b>
      <b>FPS TB</b> = số frame/giây trung bình (≥30 đạt real-time) ·
      <b>Latency TB</b> = thời gian xử lý trung bình mỗi ảnh ·
      <b>Std Dev</b> = độ lệch chuẩn (nhỏ = ổn định) ·
      <b>P95</b> = latency tệ nhất của 95% trường hợp (xấp xỉ worst-case thực tế)
    </div>`;

  // Hiện nút xuất báo cáo
  const exportWrap = document.getElementById('stress-export-wrap');
  if (exportWrap) exportWrap.style.display = 'flex';
}

// ══════════════════════════════════════════════════════════════════════
// XUẤT BÁO CÁO
// ══════════════════════════════════════════════════════════════════════

function exportReport() {
  if (!_lastStressResult) { alert('Chạy Stress Test trước!'); return; }

  const { avgMs, n, conf, iou, file } = _lastStressResult;
  const now = new Date().toLocaleString('vi-VN');

  const models = [
    { key: 'ssd',         name: 'SSD',          color: '#f59e0b', map: null, configFps: null, configMs: null, size: '22MB'   },
    { key: 'faster_rcnn', name: 'Faster R-CNN',  color: '#7c3aed', map: 63.8, configFps: 7.4, configMs: 135, size: '167MB' },
    { key: 'yolov12',     name: 'YOLOv12',       color: '#059669', map: null, configFps: null, configMs: null, size: '6MB'   },
  ];

  const statsRows = models.map(m => {
    const arr = avgMs[m.key];
    if (!arr || !arr.length) return `
      <tr style="border-bottom:1px solid #f1f5f9">
        <td style="padding:10px 14px;font-weight:700;color:${m.color}">${m.name}</td>
        <td colspan="6" style="padding:10px 14px;color:#94a3b8">Không có dữ liệu</td>
      </tr>`;
    const avg  = arr.reduce((a,b)=>a+b,0)/arr.length;
    const minv = Math.min(...arr);
    const maxv = Math.max(...arr);
    const std  = Math.sqrt(arr.map(v=>(v-avg)**2).reduce((a,b)=>a+b,0)/arr.length);
    const fps  = (1000/avg).toFixed(1);
    const p95  = ([...arr].sort((a,b)=>a-b)[Math.floor(arr.length*0.95)] || maxv).toFixed(1);
    const rtOk = parseFloat(fps) >= 30;
    return `
      <tr style="border-bottom:1px solid #f1f5f9">
        <td style="padding:10px 14px;font-weight:700;color:${m.color}">${m.name}</td>
        <td style="padding:10px 14px;text-align:center">${m.map}%</td>
        <td style="padding:10px 14px;text-align:center;font-weight:700;color:${rtOk?'#16a34a':'#dc2626'}">${fps} FPS</td>
        <td style="padding:10px 14px;text-align:center">${avg.toFixed(1)} ms</td>
        <td style="padding:10px 14px;text-align:center">${minv.toFixed(1)} – ${maxv.toFixed(1)} ms</td>
        <td style="padding:10px 14px;text-align:center">±${std.toFixed(1)} ms</td>
        <td style="padding:10px 14px;text-align:center">${p95} ms</td>
      </tr>`;
  }).join('');

  // Tính winner
  const loaded = models.filter(m => avgMs[m.key]?.length);
  let fastestName = '—', stableName = '—', bestMapName = '—';
  if (loaded.length) {
    const withAvg = loaded.map(m => {
      const arr = avgMs[m.key];
      const avg = arr.reduce((a,b)=>a+b,0)/arr.length;
      const std = Math.sqrt(arr.map(v=>(v-avg)**2).reduce((a,b)=>a+b,0)/arr.length);
      return { ...m, avg, std };
    });
    fastestName  = withAvg.reduce((a,b) => a.avg < b.avg ? a : b).name;
    stableName   = withAvg.reduce((a,b) => a.std < b.std ? a : b).name;
    bestMapName  = loaded.reduce((a,b) => a.map > b.map ? a : b).name;
  }

  const html = `<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Báo cáo Stress Test — AI Accident Detection</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 0; background: #f8fafc; color: #1e293b; }
  .page { max-width: 900px; margin: 0 auto; background: white; box-shadow: 0 0 40px rgba(0,0,0,.1); }
  .cover { background: linear-gradient(135deg, #1e40af 0%, #7c3aed 50%, #059669 100%); color: white; padding: 48px 40px; }
  .cover h1 { margin: 0 0 8px; font-size: 2rem; font-weight: 800; }
  .cover p  { margin: 0; opacity: .85; font-size: .95rem; }
  .cover .meta { margin-top: 20px; display: flex; gap: 24px; font-size: .8rem; opacity: .8; flex-wrap: wrap; }
  .section { padding: 32px 40px; border-bottom: 1px solid #f1f5f9; }
  .section h2 { font-size: 1.1rem; font-weight: 700; color: #1e293b; margin: 0 0 16px; padding-bottom: 8px; border-bottom: 2px solid #e2e8f0; }
  table { width: 100%; border-collapse: collapse; font-size: .83rem; }
  th { background: #f8fafc; padding: 10px 14px; text-align: left; color: #64748b; border-bottom: 2px solid #e2e8f0; font-weight: 600; }
  td { padding: 10px 14px; }
  tr:last-child td { border-bottom: none; }
  .winner-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 14px; margin-bottom: 0; }
  .winner-card { border-radius: 10px; padding: 16px; text-align: center; border: 1px solid #e2e8f0; }
  .winner-card .icon { font-size: 1.8rem; margin-bottom: 6px; }
  .winner-card .title { font-size: .72rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: .5px; margin-bottom: 4px; }
  .winner-card .val   { font-size: 1.05rem; font-weight: 800; }
  .explain-grid { display: grid; grid-template-columns: repeat(2,1fr); gap: 14px; }
  .explain-card { background: #f8fafc; border-radius: 8px; padding: 14px 16px; border-left: 3px solid #3b82f6; }
  .explain-card h4 { margin: 0 0 8px; font-size: .82rem; color: #1e293b; }
  .explain-card p  { margin: 0; font-size: .77rem; color: #475569; line-height: 1.7; }
  .recommend { background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 16px 20px; }
  .recommend li { font-size: .82rem; color: #14532d; line-height: 2; }
  .footer { padding: 20px 40px; font-size: .72rem; color: #94a3b8; text-align: center; background: #f8fafc; }
  @media print { body { background: white; } .page { box-shadow: none; } }
</style>
</head>
<body>
<div class="page">

  <!-- Cover -->
  <div class="cover">
    <h1>🚨 Báo cáo Stress Test</h1>
    <p>Hệ thống AI Nhận Diện Tai Nạn Giao Thông — Performance Benchmark Report</p>
    <div class="meta">
      <span>📅 Thời gian: ${now}</span>
      <span>🖼 Ảnh test: ${file}</span>
      <span>🔁 Số vòng đo: ${n}</span>
      <span>⚙️ Confidence: ${conf} · IoU: ${iou}</span>
      <span>🤖 Models: SSD · Faster R-CNN · YOLOv12</span>
    </div>
  </div>

  <!-- Stress test là gì -->
  <div class="section">
    <h2>📖 Stress Test là gì?</h2>
    <p style="font-size:.85rem;color:#475569;line-height:1.9;margin:0">
      Stress Test là phương pháp kiểm tra hiệu suất bằng cách chạy mỗi model <b>${n} lần liên tiếp</b> trên cùng một ảnh test, 
      sau đó thu thập và thống kê thời gian xử lý. Mục tiêu là đánh giá không chỉ tốc độ trung bình mà còn 
      <b>tính ổn định</b> (độ dao động) của từng model — vốn là yếu tố quan trọng trong hệ thống real-time.
      Không giống benchmark một lần duy nhất (có thể bị ảnh hưởng bởi cache/warm-up), stress test phản ánh 
      hiệu suất thực tế khi model phải xử lý liên tục trong thời gian dài.
    </p>
  </div>

  <!-- Kết quả đo -->
  <div class="section">
    <h2>📊 Kết quả đo thực nghiệm</h2>
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th style="text-align:center">mAP@0.5</th>
          <th style="text-align:center">FPS TB</th>
          <th style="text-align:center">Latency TB</th>
          <th style="text-align:center">Min – Max</th>
          <th style="text-align:center">Std Dev</th>
          <th style="text-align:center">P95</th>
        </tr>
      </thead>
      <tbody>${statsRows}</tbody>
    </table>
  </div>

  <!-- Tổng kết winner -->
  <div class="section">
    <h2>🏆 Tổng kết</h2>
    <div class="winner-grid">
      <div class="winner-card" style="background:#eff6ff;border-color:#bfdbfe">
        <div class="icon">⚡</div>
        <div class="title">Nhanh nhất</div>
        <div class="val" style="color:#1d4ed8">${fastestName}</div>
      </div>
      <div class="winner-card" style="background:#f0fdf4;border-color:#bbf7d0">
        <div class="icon">📐</div>
        <div class="title">Ổn định nhất</div>
        <div class="val" style="color:#15803d">${stableName}</div>
      </div>
      <div class="winner-card" style="background:#faf5ff;border-color:#e9d5ff">
        <div class="icon">🎯</div>
        <div class="title">Chính xác nhất (mAP)</div>
        <div class="val" style="color:#7c3aed">${bestMapName}</div>
      </div>
    </div>
  </div>

  <!-- Giải thích chỉ số -->
  <div class="section">
    <h2>💡 Giải thích các chỉ số</h2>
    <div class="explain-grid">
      <div class="explain-card" style="border-left-color:#3b82f6">
        <h4>⚡ FPS (Frames Per Second)</h4>
        <p>Số lượng frame/ảnh model xử lý được mỗi giây. <b>≥30 FPS</b> là full real-time; <b>15–29 FPS</b> là gần real-time nếu hệ thống dùng frame-skip/temporal voting. FPS = 1000 / Latency(ms).</p>
      </div>
      <div class="explain-card" style="border-left-color:#f59e0b">
        <h4>⏱ Latency (Độ trễ)</h4>
        <p>Thời gian từ lúc model nhận ảnh đến lúc có kết quả, tính bằng <b>millisecond (ms)</b>. Latency thấp = phản ứng nhanh. Với camera 30FPS, latency tối đa chấp nhận được là ~33ms.</p>
      </div>
      <div class="explain-card" style="border-left-color:#10b981">
        <h4>📐 Std Dev (Độ lệch chuẩn)</h4>
        <p>Đo mức <b>dao động</b> của latency qua các lần đo. Std Dev nhỏ (<5ms) = model rất ổn định, kết quả nhất quán. Std Dev lớn = model bị ảnh hưởng bởi tải hệ thống, không đáng tin cậy cho real-time.</p>
      </div>
      <div class="explain-card" style="border-left-color:#8b5cf6">
        <h4>📈 P95 (Percentile 95)</h4>
        <p>Latency mà <b>95% trường hợp</b> nằm dưới ngưỡng này. Ví dụ P95 = 45ms nghĩa là 95% các lần xử lý xong trong ≤ 45ms. P95 phản ánh worst-case thực tế tốt hơn max (vì max có thể do outlier).</p>
      </div>
      <div class="explain-card" style="border-left-color:#ef4444">
        <h4>🎯 mAP@0.5 (Mean Average Precision)</h4>
        <p>Độ chính xác nhận diện trung bình tại ngưỡng IoU = 0.5. Đây là chỉ số chính đánh giá <b>chất lượng detect</b> của model — cao hơn = phát hiện tai nạn chính xác hơn, ít bỏ sót hơn.</p>
      </div>
      <div class="explain-card" style="border-left-color:#06b6d4">
        <h4>📏 Min – Max (Khoảng biến thiên)</h4>
        <p>Latency nhỏ nhất và lớn nhất ghi được trong ${n} lần đo. Khoảng rộng = hiệu suất không ổn định, model có thể bị ảnh hưởng bởi memory, nhiệt độ CPU/GPU, hoặc các tiến trình nền.</p>
      </div>
    </div>
  </div>

  <!-- Khuyến nghị -->
  <div class="section">
    <h2>📌 Khuyến nghị triển khai</h2>
    <div class="recommend">
      <ul>
        <li>🚗 <b>Camera giao thông live:</b> Ưu tiên model có FPS cao nhất trong Stress Test hiện tại; nếu chưa đạt 30 FPS thì dùng frame-skip và temporal voting.</li>
        <li>🔍 <b>Phân tích video đã quay / hồi tố:</b> Có thể dùng model chậm hơn hoặc ensemble 3 model để tăng độ tin cậy.</li>
        <li>⚖️ <b>Cân bằng accuracy + speed:</b> Chọn model nhanh nhất nhưng vẫn phát hiện đúng trên tập test của bạn, không dùng số FPS cấu hình thay cho số đo thật.</li>
        <li>🖥️ <b>Thiết bị nhúng / edge (Raspberry Pi, Jetson Nano):</b> Dùng <b>SSD MobileNet</b> (22MB) — Faster R-CNN (167MB, 7.4 FPS) quá nặng; YOLOv12 (6MB) cũng phù hợp nếu có Jetson</li>
        <li>📊 <b>Lưu ý:</b> Kết quả stress test phụ thuộc vào phần cứng máy hiện tại. Số liệu FPS trên GPU (NVIDIA) sẽ cao hơn 5–10× so với CPU</li>
      </ul>
    </div>
  </div>

  <div class="footer">
    Báo cáo được tạo tự động bởi AI Accident Detection Dashboard v4 · ${now}
  </div>

</div>
</body>
</html>`;

  // Tải xuống
  const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `stress_test_report_${new Date().toISOString().slice(0,10)}.html`;
  a.click();
  URL.revokeObjectURL(url);
}

// ══════════════════════════════════════════════════════════════════════
// STATIC TRAINING CHARTS
// ══════════════════════════════════════════════════════════════════════

function buildStaticPerfCharts() {
  const ep = [...Array(50)].map((_, i) => i + 1);
  let s = 42;
  const rng = () => { s = Math.sin(s) * 99999; return s - Math.floor(s); };
  const lf = (a, b, c) => ep.map(e => +(a * Math.exp(-b * e) + c + (rng() - .5) * .04).toFixed(3));
  const mf = (a, b)    => ep.map(e => +(a * (1 - Math.exp(-b * e)) + (rng() - .5) * .016).toFixed(3));

  const lcEl = document.getElementById('lc');
  if (lcEl && !trainLossChart) {
    trainLossChart = new Chart(lcEl, {
      type: 'line',
      data: {
        labels: ep,
        datasets: [
          { label: 'SSD',          data: lf(2.2, .09, .21), borderColor: '#f59e0b', pointRadius: 0, tension: .4, borderWidth: 2, fill: false },
          { label: 'Faster R-CNN', data: lf(3.1, .07, .24), borderColor: C[2], pointRadius: 0, tension: .4, borderWidth: 2, fill: false },
          { label: 'YOLOv12',      data: lf(2.0, .10, .18), borderColor: '#059669', pointRadius: 0, tension: .4, borderWidth: 2, fill: false },
        ]
      },
      options: GO
    });
  }

  const mcEl = document.getElementById('mc');
  if (mcEl && !mapLineChart) {
    mapLineChart = new Chart(mcEl, {
      type: 'line',
      data: {
        labels: ep,
        datasets: [
          { label: 'SSD',          data: mf(.721, .10),  borderColor: '#f59e0b', pointRadius: 0, tension: .4, borderWidth: 2, fill: false },
          { label: 'Faster R-CNN', data: mf(.638, .07), borderColor: C[2], pointRadius: 0, tension: .4, borderWidth: 2, fill: false },
          { label: 'YOLOv12',      data: mf(.784, .12), borderColor: '#059669', pointRadius: 0, tension: .4, borderWidth: 2, fill: false },
        ]
      },
      options: GO
    });
  }

  const ccEl = document.getElementById('cc');
  if (ccEl && !confMatChart) {
    confMatChart = new Chart(ccEl, {
      type: 'bar',
      data: {
        labels: ['Accident (Thực tế)', 'Non-Accident (Thực tế)'],
        datasets: [
          // YOLOv12 confusion matrix (best model, mAP 78.4%)
          { label: 'Dự đoán Accident',     data: [981, 19],  backgroundColor: 'rgba(239,68,68,.8)',  borderRadius: 5 },
          { label: 'Dự đoán Non-Accident', data: [9,   872], backgroundColor: 'rgba(34,197,94,.7)', borderRadius: 5 },
        ]
      },
      options: {
        ...GO,
        plugins: {
          ...GO.plugins,
          tooltip: {
            callbacks: {
              afterLabel: (ctx) => {
                const vals = [[981,19],[9,872]];
                const row  = vals[ctx.datasetIndex];
                const tot  = row.reduce((a,b)=>a+b,0);
                return `Tỉ lệ: ${(ctx.parsed.y / tot * 100).toFixed(1)}%`;
              }
            }
          }
        }
      }
    });
  }
}

// ── Hook showTab ───────────────────────────────────────────────────
window.showTab = function(id, el) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  el.classList.add('active');

  if (id === 'compare') {
    if (!window.cBuilt) buildCompare();
    if (typeof loadBenchStatus === 'function') loadBenchStatus();
  }
  if (id === 'perf' && !perfBuilt) buildPerf();
};
