// ══════════════════════════════════════════════════════════════════
// BENCHMARK SO SÁNH THẬT — thêm vào cuối main.js
// ══════════════════════════════════════════════════════════════════

// Chart instances để destroy trước khi vẽ lại
let _bSpeedChart = null, _bConfChart  = null, _bRadarChart = null;

// ── Load trạng thái model khi vào tab ──────────────────────────────
async function loadBenchStatus() {
  try {
    const res  = await fetch('/benchmark_status');
    const data = await res.json();
    const row  = document.getElementById('bench-status-row');
    if (!row) return;
    row.innerHTML = '';
    for (const [key, info] of Object.entries(data)) {
      const ok  = info.loaded;
      const dot = ok ? '🟢' : '🔴';
      const txt = ok ? `✅ ${info.path || info.name}` : (info.error ? `❌ ${info.error.slice(0,50)}` : '⚠️ Chưa load');
      row.innerHTML += `
        <div style="flex:1;min-width:180px;padding:10px 14px;border-radius:10px;border:1px solid ${ok?'#bbf7d0':'#fecaca'};background:${ok?'#f0fdf4':'#fff5f5'}">
          <div style="font-weight:700;font-size:.82rem;color:${info.color||'#374151'}">${dot} ${info.name}</div>
          <div style="font-size:.72rem;color:#64748b;margin-top:3px">${txt}</div>
          <div style="font-size:.72rem;color:#94a3b8;margin-top:2px">mAP@0.5: ${info.map50 == null ? 'N/A' : info.map50 + '%'} | FPS: ${info.fps == null ? 'N/A' : info.fps} | ${info.size}</div>
        </div>`;
    }
  } catch(e) {
    console.error('bench status error:', e);
  }
}

// ── Chạy benchmark ─────────────────────────────────────────────────
async function runBenchmark() {
  const fileEl = document.getElementById('bench-file');
  if (!fileEl.files.length) {
    alert('Vui lòng chọn 1 ảnh để so sánh!');
    return;
  }

  const conf = document.getElementById('bench-conf').value;
  const iou  = document.getElementById('bench-iou').value;

  // Show loading
  document.getElementById('bench-loading').style.display = 'block';
  document.getElementById('bench-results').style.display = 'none';
  document.getElementById('bench-static-charts').style.display = 'none';
  document.getElementById('bench-run-btn').disabled = true;

  // Animate progress
  let prog = 0;
  const msgs = ['Đang chạy SSD...', 'Đang chạy Faster R-CNN...', 'Đang inference YOLOv12...', 'Đang tổng hợp kết quả...'];
  const progBar  = document.getElementById('bench-prog-bar');
  const progMsg  = document.getElementById('bench-loading-msg');
  const progStep = document.getElementById('bench-prog-steps');
  let msgIdx = 0;
  const ticker = setInterval(() => {
    prog = Math.min(prog + 8, 90);
    progBar.style.width = prog + '%';
    if (prog % 25 === 0 && msgIdx < msgs.length) {
      progMsg.textContent = msgs[msgIdx++];
      progStep.textContent = `Bước ${msgIdx}/${msgs.length}`;
    }
  }, 300);

  try {
    const fd = new FormData();
    fd.append('file', fileEl.files[0]);
    fd.append('conf', conf);
    fd.append('iou', iou);

    const res  = await fetch('/benchmark_image', { method: 'POST', body: fd });
    const data = await res.json();

    clearInterval(ticker);
    progBar.style.width = '100%';

    if (data.error) {
      alert('Lỗi: ' + data.error);
      document.getElementById('bench-loading').style.display = 'none';
      document.getElementById('bench-run-btn').disabled = false;
      return;
    }

    await new Promise(r => setTimeout(r, 300)); // brief pause
    document.getElementById('bench-loading').style.display = 'none';
    document.getElementById('bench-results').style.display = 'block';

    renderBenchResults(data);

  } catch(e) {
    clearInterval(ticker);
    alert('Lỗi kết nối: ' + e.message);
    document.getElementById('bench-loading').style.display = 'none';
  }
  document.getElementById('bench-run-btn').disabled = false;
}

// ── Render toàn bộ kết quả ─────────────────────────────────────────
function renderBenchResults(data) {
  const results = data.results;
  const ORDER   = ['ssd', 'faster_rcnn', 'yolov12'];

  // ─ 1. Image grid ─
  const imgGrid = document.getElementById('bench-img-grid');
  imgGrid.innerHTML = '';
  ORDER.forEach(key => {
    const r = results[key];
    if (!r) return;
    const lvlBadge = !r.loaded ? '⚪ Chưa load'
      : r.error   ? '❌ Lỗi'
      : r.level === 2 ? '🚨 TAI NẠN'
      : r.level === 1 ? '⚠️ CẢNH BÁO'
      : '✅ BÌNH THƯỜNG';
    const lvlColor = r.level === 2 ? '#dc2626' : r.level === 1 ? '#d97706' : '#16a34a';
    const borderColor = r.color || '#e2e8f0';

    imgGrid.innerHTML += `
      <div style="border:2px solid ${borderColor};border-radius:12px;overflow:hidden;background:white;box-shadow:0 2px 8px rgba(0,0,0,.08)">
        <div style="padding:8px 12px;background:${borderColor}15;border-bottom:1px solid ${borderColor}40;display:flex;justify-content:space-between;align-items:center">
          <span style="font-weight:700;font-size:.82rem;color:${borderColor}">${r.name || key}</span>
          <span style="font-size:.75rem;font-weight:700;color:${lvlColor}">${lvlBadge}</span>
        </div>
        ${r.image
          ? `<img src="${r.image}" style="width:100%;display:block;max-height:220px;object-fit:contain;background:#f8fafc" alt="${r.name}">`
          : `<div style="height:180px;display:flex;align-items:center;justify-content:center;color:#94a3b8;font-size:.8rem">${r.error || 'Model chưa load'}</div>`
        }
        <div style="padding:8px 12px;font-size:.75rem;color:#64748b;display:flex;gap:10px;flex-wrap:wrap">
          ${r.timing_ms !== undefined ? `<span>⚡ <b>${r.timing_ms}ms</b></span>` : ''}
          ${r.fps       !== undefined ? `<span>📹 <b>${r.fps} FPS</b></span>` : ''}
          ${r.acc_conf  ? `<span>🎯 <b>${r.acc_conf}</b></span>` : ''}
          ${r.num_det   !== undefined ? `<span>📦 <b>${r.num_det} box</b></span>` : ''}
        </div>
      </div>`;
  });

  // ─ 2. Table ─
  const tbody = document.getElementById('bench-table-body');
  tbody.innerHTML = '';
  // Find fastest & most confident
  const loaded = ORDER.map(k => results[k]).filter(r => r && r.loaded && !r.error && r.timing_ms !== undefined);
  const fastest    = loaded.length ? loaded.reduce((a,b) => a.timing_ms < b.timing_ms ? a : b) : null;
  const mostConf   = loaded.length ? loaded.reduce((a,b) => (a.max_conf||0) > (b.max_conf||0) ? a : b) : null;

  ORDER.forEach(key => {
    const r = results[key];
    if (!r) return;
    const isErr     = !r.loaded || r.error;
    const isFastest = fastest    && r.name === fastest.name;
    const isTop     = mostConf   && r.name === mostConf.name && (r.max_conf||0) > 0;
    const lvlCell   = isErr ? '—'
      : r.level === 2 ? '<span style="color:#dc2626;font-weight:700">🚨 TAI NẠN</span>'
      : r.level === 1 ? '<span style="color:#d97706;font-weight:700">⚠️ CẢNH BÁO</span>'
      : '<span style="color:#16a34a;font-weight:700">✅ BÌNH THƯỜNG</span>';

    tbody.innerHTML += `
      <tr style="border-bottom:1px solid #f1f5f9;${isFastest?'background:#eff6ff':''}">
        <td style="padding:10px 14px;font-weight:700;color:${r.color||'#374151'}">${r.name || key}</td>
        <td style="padding:10px 14px;text-align:center">${lvlCell}</td>
        <td style="padding:10px 14px;text-align:center;font-weight:600;color:${isTop?'#2563eb':'#374151'}">${isErr?'—':(r.acc_conf||'—')} ${isTop?'🏆':''}</td>
        <td style="padding:10px 14px;text-align:center;font-weight:600;color:${isFastest?'#16a34a':'#374151'}">${isErr?'—':r.timing_ms+'ms'} ${isFastest?'⚡':''}</td>
        <td style="padding:10px 14px;text-align:center">${isErr?'—':r.fps}</td>
        <td style="padding:10px 14px;text-align:center">${isErr?'—':r.num_det}</td>
        <td style="padding:10px 14px;text-align:center">${r.map50_str || 'N/A'}</td>
        <td style="padding:10px 14px;text-align:center">${r.size||'—'}</td>
      </tr>`;
  });

  // ─ 3. Charts ─
  drawBenchCharts(results, ORDER);

  // ─ 4. Report ─
  generateReport(results, ORDER, data);
}

// ── Vẽ 3 biểu đồ ──────────────────────────────────────────────────
function drawBenchCharts(results, ORDER) {
  const names   = ORDER.map(k => results[k]?.name || k);
  const colors  = ORDER.map(k => results[k]?.color || '#64748b');
  const colorsA = colors.map(c => c + 'CC');

  const speedVals = ORDER.map(k => results[k]?.timing_ms || 0);
  const confVals  = ORDER.map(k => Number((results[k]?.max_conf || 0).toFixed(2)));
  const map50Vals = ORDER.map(k => results[k]?.map50 ?? 0);

  const fontOpts = { family: 'DM Sans', size: 11 };
  const baseOpts = {
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: '#64748b', font: fontOpts }, grid: { color: '#e2e8f0' } },
      y: { ticks: { color: '#64748b', font: fontOpts }, grid: { color: '#e2e8f0' } },
    }
  };

  // Destroy cũ
  if (_bSpeedChart) { _bSpeedChart.destroy(); _bSpeedChart = null; }
  if (_bConfChart)  { _bConfChart.destroy();  _bConfChart  = null; }
  if (_bRadarChart) { _bRadarChart.destroy(); _bRadarChart = null; }

  // Speed chart (bar, lower=better)
  _bSpeedChart = new Chart(document.getElementById('bench-speed-chart'), {
    type: 'bar',
    data: {
      labels: names,
      datasets: [{
        label: 'Inference (ms)',
        data: speedVals,
        backgroundColor: colorsA,
        borderColor: colors,
        borderWidth: 2,
        borderRadius: 6,
      }]
    },
    options: {
      ...baseOpts,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => `${ctx.parsed.y} ms` } }
      },
      scales: {
        ...baseOpts.scales,
        y: { ...baseOpts.scales.y, min: 0, title: { display: true, text: 'ms (thấp hơn = nhanh hơn)', color:'#94a3b8', font:{size:10} } }
      }
    }
  });

  // Confidence chart
  _bConfChart = new Chart(document.getElementById('bench-conf-chart'), {
    type: 'bar',
    data: {
      labels: names,
      datasets: [{
        label: 'Confidence (0-1)',
        data: confVals,
        backgroundColor: colorsA,
        borderColor: colors,
        borderWidth: 2,
        borderRadius: 6,
      }]
    },
    options: {
      ...baseOpts,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => `${ctx.parsed.y.toFixed(2)}` } }
      },
      scales: {
        ...baseOpts.scales,
        y: { ...baseOpts.scales.y, min: 0, max: 1, title: { display: true, text: '0-1 (cao hon = tu tin hon)', color:'#94a3b8', font:{size:10} } }
      }
    }
  });

  // Radar chart (multi-dim)
  // Normalize speed: faster = higher score
  const maxMs = Math.max(...speedVals.filter(v=>v>0)) || 1;
  const speedScore = speedVals.map(v => v > 0 ? Math.round((1 - v/maxMs) * 100 + 20) : 0);

  _bRadarChart = new Chart(document.getElementById('bench-radar-chart'), {
    type: 'radar',
    data: {
      labels: ['mAP@0.5', 'Confidence\n(lần này)', 'Tốc độ\n(inv)', 'Kích thước\n(inv)'],
      datasets: ORDER.map((k, i) => {
        const r = results[k];
        const sizeScore = k === 'faster_rcnn' ? 5 : 90; // inverse size score
        return {
          label: r?.name || k,
          data: [r?.map50 ?? 0, Math.round((r?.max_conf || 0) * 100), speedScore[i], sizeScore],
          borderColor: colors[i],
          backgroundColor: colorsA[i] + '44',
          pointBackgroundColor: colors[i],
          borderWidth: 2,
          pointRadius: 3,
        };
      })
    },
    options: {
      plugins: {
        legend: {
          display: true,
          labels: { color: '#475569', font: fontOpts, boxWidth: 12 }
        }
      },
      scales: {
        r: {
          min: 0, max: 100,
          ticks: { color: '#64748b', backdropColor: 'transparent', font: { size: 9 }, stepSize: 25 },
          grid:  { color: '#e2e8f0' },
          pointLabels: { color: '#475569', font: { size: 9 } }
        }
      }
    }
  });
}

// ── Sinh báo cáo tự động ───────────────────────────────────────────
function generateReport(results, ORDER, data) {
  const loaded = ORDER.map(k => results[k]).filter(r => r && r.loaded && !r.error && r.timing_ms !== undefined);
  if (!loaded.length) {
    document.getElementById('bench-report').innerHTML = '<p style="color:#94a3b8">Không có model nào load được để tổng hợp.</p>';
    return;
  }

  const fastest  = loaded.reduce((a,b) => a.timing_ms < b.timing_ms ? a : b);
  const slowest  = loaded.reduce((a,b) => a.timing_ms > b.timing_ms ? a : b);
  const mostConf = loaded.reduce((a,b) => (a.max_conf||0) > (b.max_conf||0) ? a : b);
  const loadedWithMap = loaded.filter(r => r.map50 != null);
  const bestMap  = loadedWithMap.length ? loadedWithMap.reduce((a,b) => (a.map50||0) > (b.map50||0) ? a : b) : null;
  const accCount = loaded.filter(r => r.level === 2).length;
  const warnCount= loaded.filter(r => r.level === 1).length;
  const okCount  = loaded.filter(r => r.level === 0).length;

  const consensus = accCount >= 2 ? '🚨 <b>ĐA SỐ MODEL PHÁT HIỆN TAI NẠN</b> — cần xử lý ngay!'
    : warnCount >= 2 ? '⚠️ <b>ĐA SỐ MODEL PHÁT HIỆN NGUY CƠ</b> — cần theo dõi'
    : okCount >= 2   ? '✅ <b>ĐA SỐ MODEL: BÌNH THƯỜNG</b>'
    : '🔀 Các model cho kết quả khác nhau — cần xem xét thêm';

  const speedDiff = slowest.timing_ms && fastest.timing_ms
    ? Math.round(slowest.timing_ms / fastest.timing_ms * 10) / 10
    : 1;

  const html = `
    <div style="margin-bottom:16px;padding:12px 16px;border-radius:8px;background:#f8fafc;border-left:4px solid #3b82f6">
      <b>🔎 Kết luận đồng thuận:</b> ${consensus}
    </div>

    <table style="width:100%;border-collapse:collapse;margin-bottom:16px;font-size:.82rem">
      <tr>
        <td style="padding:6px 0;color:#64748b;width:45%">⚡ Model nhanh nhất (lần này)</td>
        <td style="font-weight:700;color:#16a34a">${fastest.name} — ${fastest.timing_ms}ms (${fastest.fps} FPS)</td>
      </tr>
      <tr>
        <td style="padding:6px 0;color:#64748b">🐢 Model chậm nhất (lần này)</td>
        <td style="font-weight:700;color:#dc2626">${slowest.name} — ${slowest.timing_ms}ms (${slowest.fps} FPS)</td>
      </tr>
      <tr>
        <td style="padding:6px 0;color:#64748b">📏 Chênh lệch tốc độ</td>
        <td style="font-weight:700">${fastest.name} nhanh hơn ${slowest.name} <b>${speedDiff}×</b></td>
      </tr>
      <tr>
        <td style="padding:6px 0;color:#64748b">🎯 Confidence cao nhất (lần này)</td>
        <td style="font-weight:700;color:#2563eb">${mostConf.name} — ${mostConf.acc_conf || '—'}</td>
      </tr>
      <tr>
        <td style="padding:6px 0;color:#64748b">🏅 mAP@0.5 cao nhất (benchmark)</td>
        <td style="font-weight:700;color:#7c3aed">${bestMap ? `${bestMap.name} - ${bestMap.map50}%` : 'N/A'}</td>
      </tr>
    </table>

    <div style="margin-bottom:12px">
      <b>📌 Phân tích từng model:</b>
      <ul style="margin:8px 0 0 18px;line-height:2.2">
        ${loaded.map(r => {
          const lvlText = r.level === 2 ? '🚨 TAI NẠN' : r.level === 1 ? '⚠️ CẢNH BÁO' : '✅ Bình thường';
          const speedTag = r.name === fastest.name ? ' <span style="color:#16a34a;font-size:.75rem">[NHANH NHẤT]</span>' : r.name === slowest.name ? ' <span style="color:#dc2626;font-size:.75rem">[CHẬM NHẤT]</span>' : '';
          return `<li><b style="color:${r.color||'#374151'}">${r.name}</b>${speedTag}: ${lvlText}, confidence ${r.acc_conf||'—'}, inference ${r.timing_ms}ms, ${r.num_det} box phát hiện</li>`;
        }).join('')}
      </ul>
    </div>

    <div style="margin-bottom:10px">
      <b>💡 Khuyến nghị sử dụng:</b>
      <ul style="margin:8px 0 0 18px;line-height:2.2">
        <li>🚗 <b>Real-time / camera giao thông:</b> Dùng <b>SSD</b> hoặc <b>YOLOv12</b> — FPS cao, latency thấp</li>
        <li>🔍 <b>Phân tích hồi tố / video đã quay:</b> Dùng <b>Faster R-CNN</b> — chính xác nhất, tốc độ không cần thiết</li>
        <li>⚖️ <b>Cân bằng accuracy + speed:</b> Dùng <b>YOLOv12</b> — mới nhất, cải thiện cả 2 mặt</li>
      </ul>
    </div>

    <div style="font-size:.75rem;color:#94a3b8;border-top:1px solid #f1f5f9;padding-top:10px;margin-top:10px">
      📐 Ảnh test: ${data.img_size || '—'} &nbsp;|&nbsp; 
      🤖 Model đã chạy: ${data.num_models || loaded.length}/3 &nbsp;|&nbsp;
      ⚙️ Conf threshold: ${document.getElementById('bench-conf').value} &nbsp;|&nbsp;
      🔁 IoU threshold: ${document.getElementById('bench-iou').value}
    </div>`;

  document.getElementById('bench-report').innerHTML = html;
}

// ── Hook vào showTab để load status khi vào tab compare ───────────
const _origShowTab = window.showTab;
window.showTab = function(id, el) {
  _origShowTab(id, el);
  if (id === 'compare') {
    loadBenchStatus();
  }
};

// Tự load status lần đầu nếu tab compare đang active
if (document.getElementById('tab-compare')?.classList.contains('active')) {
  loadBenchStatus();
}
