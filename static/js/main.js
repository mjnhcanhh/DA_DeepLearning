/**
 * main.js — AI Accident Detection Dashboard v4
 * Xử lý: multi-image grid, accuracy checker, session stats, webcam/video
 * UPDATED:
 *   - Checkbox trên TẤT CẢ ảnh (2 chiều: model sai theo cả 2 hướng)
 *   - So sánh accuracy 2 mô hình ngay bên dưới panel accuracy
 *   - Level 1 (sắp tai nạn) hiển thị màu VÀNG (border + badge)
 */

// ══════════════════════════════════════════════════════════════════
// CLOCK
// ══════════════════════════════════════════════════════════════════
setInterval(() => {
  const el = document.getElementById('clk');
  if (el) el.textContent = new Date().toLocaleString('vi-VN');
}, 1000);

// ══════════════════════════════════════════════════════════════════
// TABS
// ══════════════════════════════════════════════════════════════════
let cBuilt = false, pBuilt = false;

function showTab(id, el) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + id).classList.add('active');
  el.classList.add('active');
  if (id === 'compare' && !cBuilt) buildCompare();
  if (id === 'perf'    && !pBuilt) buildPerf();
  // Leaflet cần invalidateSize sau khi panel visible
  if (id === 'map' && window.cameraMap) {
    setTimeout(() => window.cameraMap.invalidateSize(), 100);
  }
}

// ══════════════════════════════════════════════════════════════════
// SOURCE TOGGLE
// ══════════════════════════════════════════════════════════════════
function toggleSrc() {
  const s = document.getElementById('src').value;
  document.getElementById('fg').style.display = s === 'webcam' ? 'none' : 'flex';
  document.getElementById('fi').multiple      = (s === 'image');
  document.getElementById('fi').accept        = s === 'image' ? 'image/*' : (s === 'video' ? 'video/*' : 'image/*');
  document.getElementById('file-count-label').textContent = s === 'image' ? '(chọn nhiều)' : '';
  document.getElementById('grid-cols-wrap').style.display = s === 'image' ? 'flex' : 'none';
  document.getElementById('accuracy-panel').classList.remove('visible');
}

// ══════════════════════════════════════════════════════════════════
// ALGORITHM SWITCH
// ══════════════════════════════════════════════════════════════════
const ALGO_INFO = {
  ssd:         { map: 'N/A', fps: 'N/A', lat: 'N/A', badge: 'badge-ssd',   label: 'SSD' },
  faster_rcnn: { map: '63.8%', fps: '7.4', lat: '135ms', badge: 'badge-rcnn',  label: 'Faster R-CNN' },
  yolov12:     { map: 'N/A', fps: 'N/A', lat: 'N/A', badge: 'badge-yolo12', label: 'YOLOv12' },
};

let currentAlgo = 'ssd';

function setAlgo(key, name, badgeClass, map, fps, lat) {
  currentAlgo = key;
  document.querySelectorAll('.algo-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('btn-' + key).classList.add('active');
  const b = document.getElementById('active-badge');
  b.textContent = name; b.className = 'badge ' + badgeClass;
  document.getElementById('m-map').textContent = map;
  document.getElementById('m-fps').textContent = fps;
  document.getElementById('m-lat').textContent = lat;
  const st = (window.ALGO_STATUS && window.ALGO_STATUS[key]) || '❓ Chưa kiểm tra';
  document.getElementById('algo-status').textContent = name + ' — ' + st;
  fetch('/set_algo/' + key);
}

// ══════════════════════════════════════════════════════════════════
// SESSION STATS
// ══════════════════════════════════════════════════════════════════
let sF = 0, sA = 0, sW = 0, log = [], running = false;

function upd(lvl) {
  sF++;
  if (lvl === 2) sA++;
  if (lvl === 1) sW++;
  document.getElementById('ma').textContent = sA;
  document.getElementById('mw').textContent = sW;
  document.getElementById('sf').textContent = sF;
  document.getElementById('sa').textContent = sA;
  document.getElementById('sw').textContent = sW;
  const danger = sA + sW;
  document.getElementById('sr').textContent = sF ? Math.round(danger / sF * 100) + '%' : '0%';
}

function addLog(src, lvl, conf) {
  if (lvl === 0) return;
  const t = new Date().toLocaleTimeString('vi-VN');
  const lbl = lvl === 2 ? '🚨 TAI NẠN' : '⚠️ CẢNH BÁO';
  log.unshift({ t, algo: ALGO_INFO[currentAlgo]?.label || currentAlgo, src, lbl, conf, lvl });
  renderLog();
  document.getElementById('alert-card').style.display = 'block';
  document.getElementById('latest-alert').innerHTML = lvl === 2
    ? `<div class="abox-acc">🚨 <b>TAI NẠN PHÁT HIỆN!</b><br>${t} | ${conf}<br>${src}</div>`
    : `<div class="abox-warn">⚠️ <b>CẢNH BÁO: Có khả năng xảy ra tai nạn!</b><br>${t} | ${conf}<br>${src}</div>`;
}

function renderLog() {
  const lb = document.getElementById('lb');
  if (!log.length) {
    lb.innerHTML = '<tr><td colspan="5" style="color:#94a3b8;text-align:center;padding:18px">Chưa có sự kiện nguy hiểm</td></tr>';
    document.getElementById('la').innerHTML = ''; return;
  }
  lb.innerHTML = log.map(r =>
    `<tr><td>${r.t}</td><td>${r.algo}</td><td>${r.src}</td><td>${r.lbl}</td><td>${r.conf}</td></tr>`
  ).join('');
  const latest = log[0];
  document.getElementById('la').innerHTML = latest.lvl === 2
    ? `<div class="abox-acc">🚨 <b>TAI NẠN MỚI NHẤT:</b> ${latest.t} | ${latest.src} | ${latest.conf}</div>`
    : `<div class="abox-warn">⚠️ <b>CẢNH BÁO MỚI NHẤT:</b> ${latest.t} | ${latest.src} | ${latest.conf}</div>`;
}

function clrLog() {
  log = []; sA = 0; sW = 0; sF = 0; renderLog();
  document.getElementById('ma').textContent = 0;
  document.getElementById('mw').textContent = 0;
  document.getElementById('alert-card').style.display = 'none';
}

// ══════════════════════════════════════════════════════════════════
// SHOW SINGLE RESULT (video/webcam)
// ══════════════════════════════════════════════════════════════════
function updateTemporalPanel(data) {
  const panel = document.getElementById('temporal-panel');
  if (!panel) return;

  const t = data.temporal;
  if (!t) {
    panel.style.display = 'none';
    return;
  }

  panel.style.display = 'block';
  const state = document.getElementById('temporal-state');
  const risk = document.getElementById('temporal-risk');
  const votes = document.getElementById('temporal-votes');
  const conf = document.getElementById('temporal-conf');
  const bar = document.getElementById('temporal-risk-bar');
  const note = document.getElementById('temporal-note');
  const incident = document.getElementById('temporal-incident');

  panel.className = 'card temporal-card ' + (
    t.confirmed ? 'temporal-confirmed' : (t.suspect ? 'temporal-suspect' : 'temporal-normal')
  );
  state.textContent = t.confirmed ? 'CONFIRMED ACCIDENT' : (t.suspect ? 'SUSPECT' : 'NORMAL');
  risk.textContent = `${t.risk_score}%`;
  votes.textContent = `${t.votes}/${t.window} frames`;
  conf.textContent = t.avg_conf ? Number(t.avg_conf).toFixed(2) : '0.00';
  bar.style.width = `${t.risk_score}%`;
  note.textContent = t.message || '';

  if (data.incident_id) {
    incident.style.display = 'block';
    incident.textContent = `Auto incident created: ${data.incident_id}`;
  } else {
    incident.style.display = 'none';
    incident.textContent = '';
  }
}

function showRes(data, src) {
  if (data.error) { alert('Lỗi: ' + data.error); return; }
  document.getElementById('ph').style.display = 'none';
  const img = document.getElementById('ri');
  img.src = data.image; img.style.display = 'block';
  const lvl = data.level === 1 ? 0 : data.level;
  const stEl = document.getElementById('st');
  if (lvl === 2) {
    stEl.textContent = '🚨 TAI NẠN PHÁT HIỆN!'; stEl.className = 'sacc';
    document.getElementById('cd').textContent = 'Đã xảy ra va chạm | ' + data.acc_conf;
  } else if (lvl === 1) {
    stEl.textContent = '⚠️ CẢNH BÁO: Có khả năng xảy ra tai nạn!'; stEl.className = 'swarn';
    document.getElementById('cd').textContent = 'Phát hiện nguy cơ | ' + data.acc_conf;
  } else {
    stEl.textContent = '✅ BÌNH THƯỜNG'; stEl.className = 'snormal';
    document.getElementById('cd').textContent = '';
  }
  if (data.temporal) {
    updateTemporalPanel(data);
    if (data.temporal.confirmed) {
      stEl.textContent = 'CONFIRMED ACCIDENT - Temporal AI';
      stEl.className = 'sacc';
      document.getElementById('cd').textContent =
        `Voting ${data.temporal.votes}/${data.temporal.window} frames | Risk ${data.temporal.risk_score}%`;
    } else if (data.temporal.suspect) {
      stEl.textContent = 'SUSPECT - waiting for more frames';
      stEl.className = 'swarn';
      document.getElementById('cd').textContent =
        `Voting ${data.temporal.votes}/${data.temporal.window} frames | Risk ${data.temporal.risk_score}%`;
    }
  }
  const rows = (data.detections || []).map(d => {
    const cls  = d.level === 2 ? 'acc' : (d.level === 1 ? 'warn' : '');
    const icon = d.level === 2 ? '🚨' : (d.level === 1 ? '⚠️' : '✅');
    const muc  = d.level === 2 ? 'Tai nạn' : (d.level === 1 ? 'Có khả năng tai nạn' : 'Bình thường');
    return `<tr class="${cls}"><td>${icon} ${d.label}</td><td>${d.conf}</td><td>${muc}</td></tr>`;
  }).join('') || '<tr><td colspan="3" style="color:#94a3b8">Không phát hiện</td></tr>';
  document.getElementById('db').innerHTML = rows;
  upd(lvl);
  if (lvl > 0) addLog(src, lvl, data.acc_conf);
}

// ══════════════════════════════════════════════════════════════════
// MULTI-IMAGE GRID
// ══════════════════════════════════════════════════════════════════
let gridAcc = 0, gridWarn = 0, gridOk = 0, gridDone = 0, gridTotal = 0;

/**
 * cardData[idx]:
 *   level      : -1 (chưa xong) | 0 (bình thường) | 1 (cảnh báo) | 2 (tai nạn)
 *   fileName   : tên file
 *   conf       : chuỗi confidence
 *   algo       : tên algo lúc xử lý
 *   userMarked : true = người dùng đánh dấu "model đoán SAI"
 *                (tức ảnh thực tế khác với kết quả model)
 */
const cardData = {};

/**
 * algoResults[algoKey] = { correct, wrong, total, label }
 * Lưu kết quả accuracy của từng algo sau khi người dùng bấm "Tính"
 */
const algoResults = {};

function applyGridCols() {
  const cols = document.getElementById('grid-cols').value;
  document.getElementById('img-grid').style.gridTemplateColumns = `repeat(${cols}, minmax(0,1fr))`;
}

function initGrid(files) {
  const grid = document.getElementById('img-grid');
  grid.innerHTML = '';
  Object.keys(cardData).forEach(k => delete cardData[k]);
  gridAcc = 0; gridWarn = 0; gridOk = 0; gridDone = 0; gridTotal = files.length;
  document.getElementById('grid-done').textContent  = 0;
  document.getElementById('grid-total').textContent = files.length;
  document.getElementById('grid-acc').textContent   = 0;
  document.getElementById('grid-warn').textContent  = 0;
  document.getElementById('grid-ok').textContent    = 0;
  document.getElementById('multi-pb').style.width   = '0%';
  document.getElementById('multi-wrap').style.display = 'block';
  document.getElementById('single-wrap').style.display = 'none';
  document.getElementById('accuracy-panel').classList.remove('visible');
  document.getElementById('acc-result').style.display = 'none';
  document.getElementById('compare-result').style.display = 'none';
  applyGridCols();

  files.forEach((f, idx) => {
    cardData[idx] = { level: -1, fileName: f.name, conf: '', algo: currentAlgo, userMarked: false };
    const card = document.createElement('div');
    card.className = 'img-card loading';
    card.id = `card-${idx}`;
    card.innerHTML = `
      <img src="" alt="${f.name}">
      <div class="spinner"></div>
      <div class="card-name">${f.name}</div>
      <div class="card-conf">Đang xử lý...</div>`;
    grid.appendChild(card);
  });

  updateFolderCounts();
}

function updateCard(idx, data, fileName) {
  const card = document.getElementById(`card-${idx}`);
  if (!card) return;
  const lvl = data.level === 1 ? 0 : data.level;
  cardData[idx] = { level: lvl, fileName, conf: data.acc_conf || '', algo: currentAlgo, userMarked: false, timingMs: data.timing_ms || 0, image: data.image };

  // ── level-1 class được CSS tô vàng ──────────────────────────────
  card.className = `img-card level-${lvl}`;

  const badgeText = lvl === 2 ? '🚨 TAI NẠN'
                  : lvl === 1 ? '⚠️ CÓ KHẢ NĂNG TAI NẠN'
                  : '✅ BÌNH THƯỜNG';

  // ── Checkbox trên TẤT CẢ ảnh (2 chiều) ─────────────────────────
  // Nếu model nói BÌNH THƯỜNG hoặc CẢNH BÁO  → "Sai! Là tai nạn"
  // Nếu model nói TAI NẠN                     → "Sai! Là bình thường"
  let checkboxLabel, checkboxTitle;
  if (lvl === 2) {
    checkboxLabel = 'Sai! Là bình thường';
    checkboxTitle = 'Tích nếu ảnh này thực ra BÌNH THƯỜNG (model đoán nhầm tai nạn)';
  } else {
    checkboxLabel = 'Sai! Là tai nạn';
    checkboxTitle = 'Tích nếu ảnh này thực ra là TAI NẠN (model đoán nhầm)';
  }

  const checkboxHtml = `
    <div class="card-checkbox-wrap" title="${checkboxTitle}">
      <input type="checkbox" id="chk-${idx}" onchange="markWrong(${idx}, this.checked)"
             onclick="event.stopPropagation()">
      <span class="card-checkbox-label">${checkboxLabel}</span>
    </div>`;

  card.innerHTML = `
    <img src="${data.image}" alt="${fileName}"
         onclick="openModal('${data.image}','${fileName}',${lvl},'${data.acc_conf}')">
    ${checkboxHtml}
    <div class="card-badge b${lvl}">${badgeText}</div>
    <div class="card-name">${fileName}</div>
    <div class="card-conf" id="card-conf-${idx}">${data.acc_conf || 'Conf: —'}</div>`;

  gridDone++;
  if (lvl === 2) gridAcc++;
  else if (lvl === 1) gridWarn++;
  else gridOk++;

  document.getElementById('grid-done').textContent = gridDone;
  document.getElementById('grid-acc').textContent  = gridAcc;
  document.getElementById('grid-warn').textContent = gridWarn;
  document.getElementById('grid-ok').textContent   = gridOk;
  document.getElementById('multi-pb').style.width  = Math.round(gridDone / gridTotal * 100) + '%';

  upd(lvl);
  if (lvl > 0) addLog(fileName, lvl, data.acc_conf);

  updateFolderCounts();

  // Hiện panel accuracy khi xong hết
  if (gridDone === gridTotal) {
    smoothFrameSequence();
    document.getElementById('accuracy-panel').classList.add('visible');
    document.getElementById('calc-acc-info').textContent =
      `${gridTotal} ảnh đã xử lý. Tích ô "Sai!" vào các ảnh mà model đoán nhầm, rồi nhấn tính.`;
  }
}

function parseFrameName(name) {
  const m = String(name || '').match(/^(.*?)(?:_frame_|frame_)(\d+)/i);
  if (!m) return null;
  return { group: m[1], frame: parseInt(m[2], 10) };
}

function refreshGridCounts() {
  const items = Object.values(cardData).filter(d => d.level >= 0);
  gridAcc = items.filter(d => d.level === 2).length;
  gridWarn = items.filter(d => d.level === 1).length;
  gridOk = items.filter(d => d.level === 0).length;
  document.getElementById('grid-acc').textContent = gridAcc;
  document.getElementById('grid-warn').textContent = gridWarn;
  document.getElementById('grid-ok').textContent = gridOk;
  updateFolderCounts();
}

function promoteCardToAccident(idx, reason) {
  const item = cardData[idx];
  const card = document.getElementById(`card-${idx}`);
  if (!item || !card || item.level === 2) return;

  item.level = 2;
  item.conf = reason;
  card.className = 'img-card level-2';

  const badge = card.querySelector('.card-badge');
  if (badge) {
    badge.className = 'card-badge b2';
    badge.textContent = '🚨 TAI NẠN';
  }

  const conf = card.querySelector('.card-conf');
  if (conf) conf.textContent = reason;

  const img = card.querySelector('img');
  if (img) img.setAttribute('onclick', `openModal('${item.image}','${item.fileName}',2,'${reason}')`);

  const checkboxText = card.querySelector('.card-checkbox-label');
  if (checkboxText) checkboxText.textContent = 'Sai! Là bình thường';
}

function smoothFrameSequence() {
  if (currentAlgo !== 'faster_rcnn') return;
  const parsed = Object.entries(cardData)
    .map(([idx, item]) => ({ idx, item, meta: parseFrameName(item.fileName) }))
    .filter(x => x.meta && x.item.level >= 0);
  if (!parsed.length) return;

  const positives = parsed.filter(x => x.item.level === 2);
  if (!positives.length) return;

  for (const cur of parsed) {
    if (cur.item.level !== 0) continue;
    const nearPositive = positives.some(pos =>
      pos.meta.group === cur.meta.group && Math.abs(pos.meta.frame - cur.meta.frame) <= 6
    );
    if (nearPositive) promoteCardToAccident(cur.idx, 'Theo chuỗi frame');
  }
  refreshGridCounts();
}

function markCardError(idx, fileName, err) {
  const card = document.getElementById(`card-${idx}`);
  if (!card) return;
  card.className = 'img-card level-0';
  card.innerHTML = `
    <div style="height:138px;display:flex;align-items:center;justify-content:center;
                color:#94a3b8;font-size:.72rem;padding:8px;text-align:center">${err}</div>
    <div class="card-name">${fileName}</div>
    <div class="card-conf" style="color:#ef4444">Lỗi</div>`;
  gridDone++;
  document.getElementById('grid-done').textContent = gridDone;
  document.getElementById('multi-pb').style.width  = Math.round(gridDone / gridTotal * 100) + '%';
}

// ─── Người dùng tích "Sai" ────────────────────────────────────────
function markWrong(idx, checked) {
  if (cardData[idx]) cardData[idx].userMarked = checked;
}

// ─── Folder counts ────────────────────────────────────────────────
function updateFolderCounts() {
  const acc  = Object.values(cardData).filter(d => d.level === 2).length;
  const warn = Object.values(cardData).filter(d => d.level === 1).length;
  const ok   = Object.values(cardData).filter(d => d.level === 0).length;
  document.getElementById('folder-acc').textContent  = acc;
  document.getElementById('folder-warn').textContent = warn;
  document.getElementById('folder-ok').textContent   = ok;
}

// ══════════════════════════════════════════════════════════════════
// ACCURACY CALCULATOR
// Tính tỉ lệ đúng trong (tối đa 50) ảnh đã nhận diện.
// Model sai = người dùng tích "Sai!" (dù theo hướng nào)
// ══════════════════════════════════════════════════════════════════
function calcAccuracy() {
  const items = Object.values(cardData).filter(d => d.level >= 0);
  if (!items.length) { alert('Chưa có ảnh nào được xử lý!'); return; }

  const N      = Math.min(items.length, 50);
  const sample = items.slice(0, N);
  const wrong   = sample.filter(d => d.userMarked).length;
  const correct = N - wrong;
  const pct     = Math.round(correct / N * 100);

  // Lưu kết quả cho mô hình hiện tại để dùng khi so sánh
  const algoLabel = ALGO_INFO[currentAlgo]?.label || currentAlgo;
  // Tính speed trung bình từ cardData
  const tms = sample.map(d=>d.timingMs).filter(t=>t>0);
  const avgMs = tms.length ? Math.round(tms.reduce((a,b)=>a+b,0)/tms.length) : null;
  algoResults[currentAlgo] = { correct, wrong, total: N, label: algoLabel, pct, avgMs };

  const big = document.getElementById('acc-big');
  big.textContent = pct + '%';
  big.style.color = pct >= 80 ? 'var(--green-500)' : pct >= 60 ? 'var(--amber-500)' : 'var(--red-500)';

  document.getElementById('acc-correct').textContent = correct;
  document.getElementById('acc-wrong').textContent   = wrong;
  document.getElementById('acc-total').textContent   = N;
  document.getElementById('acc-result').style.display = 'flex';

  // Cập nhật panel so sánh nếu đã có kết quả từ cả 2 algo
  refreshComparePanel();
}

// ══════════════════════════════════════════════════════════════════
// SO SÁNH 2 MÔ HÌNH
// Chỉ hiện khi cả 2 algo đã được tính accuracy (mỗi algo phải chạy
// trên cùng bộ ảnh riêng biệt rồi bấm "Tính tỉ lệ chính xác")
// ══════════════════════════════════════════════════════════════════
function refreshComparePanel() {
  const keys   = Object.keys(algoResults);
  const panel  = document.getElementById('compare-result');
  if (!panel) return;

  const allKeys = Object.keys(ALGO_INFO);
  const missing = allKeys.filter(k => !algoResults[k]);

  if (keys.length < 2) {
    panel.style.display = 'block';
    panel.innerHTML = `<div class="compare-hint">
      💡 Đã có: <b>${keys.map(k=>ALGO_INFO[k]?.label||k).join(', ')}</b>.
      Chuyển sang <b>${ALGO_INFO[missing[0]]?.label || missing[0]}</b>,
      chạy ảnh rồi nhấn <b>Tính tỉ lệ chính xác</b> để so sánh.
    </div>`;
    return;
  }

  // Sort best → worst
  const sorted = keys.map(k => algoResults[k]).sort((a,b) => b.pct - a.pct);
  const best   = sorted[0];

  const colHtml = (r) => {
    const isBest = r.pct === best.pct;
    const color  = r.pct >= 80 ? 'var(--green-500)' : r.pct >= 60 ? 'var(--amber-500)' : 'var(--red-500)';
    const fpsTxt = r.avgMs ? `⚡ ${r.avgMs}ms (${(1000/r.avgMs).toFixed(1)} FPS)` : '';
    return `<div class="compare-col ${isBest?'compare-winner':''}">
      <div class="compare-algo-name">${r.label}</div>
      <div class="compare-pct" style="color:${color}">${r.pct}%</div>
      <div class="compare-detail">
        ✅ ${r.correct} đúng &nbsp;❌ ${r.wrong} sai &nbsp;/ ${r.total} ảnh
        ${fpsTxt ? `<br><span style="color:#3b82f6;font-size:.73rem">${fpsTxt}</span>` : ''}
      </div>
      ${isBest ? '<div class="compare-crown">🏆 Tốt nhất</div>' : ''}
    </div>`;
  };

  // Build columns — 2 or 3 models
  const cols = keys.map(k => colHtml(algoResults[k])).join('<div class="compare-vs">VS</div>');

  // Summary line
  const gap = sorted[0].pct - sorted[sorted.length-1].pct;
  const diffTxt = gap > 0
    ? `<b>${sorted[0].label}</b> chính xác hơn <b>${gap}%</b> so với mô hình kém nhất.`
    : '🤝 Các mô hình có độ chính xác tương đương nhau.';

  const withSpeed = keys.map(k=>algoResults[k]).filter(r=>r.avgMs).sort((a,b)=>a.avgMs-b.avgMs);
  const fastTxt   = withSpeed.length >= 2
    ? `⚡ Nhanh nhất: <b>${withSpeed[0].label}</b> (${withSpeed[0].avgMs}ms)`
    : '';

  panel.style.display = 'block';
  panel.innerHTML = `
    <div class="compare-title">📊 So sánh ${keys.length} mô hình</div>
    <div class="compare-grid" style="grid-template-columns:repeat(${keys.length === 3 ? 3 : 3},1fr)">${cols}</div>
    <div class="compare-summary">
      ${diffTxt}
      ${fastTxt ? '<br>' + fastTxt : ''}
      <br><small style="color:#94a3b8">Đánh giá trên tối đa 50 ảnh — kết quả phụ thuộc bộ test.</small>
    </div>
    ${missing.length ? `<div class="compare-hint" style="margin-top:8px">
      💡 Chưa có: <b>${missing.map(k=>ALGO_INFO[k]?.label||k).join(', ')}</b> — chạy model đó rồi tính accuracy để thêm vào.
    </div>` : ''}
    <button class="btn-sm btn-ghost" onclick="resetCompare()">🔄 Reset so sánh</button>`;
}


function resetCompare() {
  Object.keys(algoResults).forEach(k => delete algoResults[k]);
  const panel = document.getElementById('compare-result');
  if (panel) panel.style.display = 'none';
}

// ══════════════════════════════════════════════════════════════════
// LIGHTBOX
// ══════════════════════════════════════════════════════════════════
function openModal(src, name, lvl, conf) {
  document.getElementById('modal-img').src = src;
  const lvlText = lvl === 2 ? '🚨 TAI NẠN' : (lvl === 1 ? '⚠️ CÓ KHẢ NĂNG TAI NẠN' : '✅ BÌNH THƯỜNG');
  document.getElementById('modal-info').innerHTML =
    `<b>${name}</b> &nbsp;|&nbsp; ${lvlText} &nbsp;|&nbsp; ${conf || ''}`;
  document.getElementById('modal').classList.add('open');
}

function closeModal(e) {
  if (e.target === document.getElementById('modal'))
    document.getElementById('modal').classList.remove('open');
}

// ══════════════════════════════════════════════════════════════════
// MAIN GO
// ══════════════════════════════════════════════════════════════════
async function go() {
  const src  = document.getElementById('src').value;
  const conf = document.getElementById('conf').value;
  const iou  = document.getElementById('iou').value;
  running = true;

  if (src === 'image') {
    const files = Array.from(document.getElementById('fi').files);
    if (!files.length) { alert('Chọn ít nhất 1 file ảnh!'); return; }
    initGrid(files);

    const CONCURRENCY = 3;
    const queue = files.map((_, i) => i);

    async function processOne(i) {
      if (!running) return;
      const f  = files[i];
      const fd = new FormData();
      fd.append('file', f);
      fd.append('conf', conf);
      fd.append('iou', iou);
      try {
        const r    = await fetch('/detect_image', { method: 'POST', body: fd });
        const data = await r.json();
        if (data.error) markCardError(i, f.name, data.error);
        else            updateCard(i, data, f.name);
      } catch (e) {
        markCardError(i, f.name, e.message);
      }
    }

    async function worker() {
      while (queue.length && running) {
        const i = queue.shift();
        await processOne(i);
      }
    }
    await Promise.all(Array.from({ length: Math.min(CONCURRENCY, files.length) }, () => worker()));

  } else if (src === 'video') {
    document.getElementById('single-wrap').style.display = 'block';
    document.getElementById('multi-wrap').style.display  = 'none';
    const f = document.getElementById('fi').files[0];
    if (!f) { alert('Chọn file video trước!'); return; }
    document.getElementById('vp').style.display = 'block';
    const fd = new FormData(); fd.append('file', f); fd.append('conf', conf); fd.append('iou', iou);
    const r      = await fetch('/detect_video', { method: 'POST', body: fd });
    const reader = r.body.getReader(); const dec = new TextDecoder(); let buf = '';
    while (running) {
      const { done, value } = await reader.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split('\n\n'); buf = parts.pop();
      for (const p of parts) {
        if (!p.startsWith('data:')) continue;
        try {
          const d = JSON.parse(p.slice(5));
          if (d.done) { document.getElementById('vi').textContent = '✅ Xử lý xong!'; running = false; break; }
          showRes(d, 'Video frame ' + d.frame);
          document.getElementById('pb').style.width = (d.progress || 0) + '%';
          document.getElementById('vi').textContent = `Frame ${d.frame} | ${d.progress || 0}%`;
        } catch (e) {}
      }
    }
    document.getElementById('vp').style.display = 'none';

  } else {
    document.getElementById('single-wrap').style.display = 'block';
    document.getElementById('multi-wrap').style.display  = 'none';
    document.getElementById('vi').textContent = '📹 Webcam đang chạy...';
    try { await fetch('/webcam_temporal_reset', { method: 'POST' }); } catch (e) {}
    while (running) {
      const r = await fetch(`/webcam_frame?conf=${conf}&iou=${iou}`);
      if (!r.ok) { alert('Không mở được webcam!'); break; }
      showRes(await r.json(), 'Webcam');
      await new Promise(x => setTimeout(x, 80));
    }
    document.getElementById('vi').textContent = '';
  }
}

function stop() {
  running = false;
  document.getElementById('vi').textContent = '⏹ Đã dừng.';
  document.querySelectorAll('.img-card.loading').forEach(card => {
    const confEl = card.querySelector('.card-conf');
    if (confEl) confEl.textContent = 'Đã dừng';
    card.classList.remove('loading');
    const spinner = card.querySelector('.spinner');
    if (spinner) spinner.remove();
  });
  const processed = Object.values(cardData).filter(d => d.level >= 0);
  if (processed.length > 0) {
    document.getElementById('accuracy-panel').classList.add('visible');
    document.getElementById('calc-acc-info').textContent =
      `Đã xử lý ${processed.length} ảnh (dừng sớm). Tích ô sai rồi nhấn tính.`;
  }
}
