async function loadIncidents() {
  const list = document.getElementById('incident-list');
  if (!list) return;

  try {
    const res = await fetch('/incidents');
    const incidents = await res.json();
    renderIncidents(incidents);
  } catch (err) {
    list.innerHTML = `<div class="incident-empty">Khong tai duoc danh sach su co: ${err.message}</div>`;
  }
}

async function loadFeedbackStats() {
  const el = document.getElementById('feedback-stats');
  if (!el) return;
  try {
    const res = await fetch('/feedback_stats');
    const data = await res.json();
    el.innerHTML = `
      Feedback dataset:
      <b>${data.confirmed_accident || 0}</b> confirmed ·
      <b>${data.false_alarm || 0}</b> false alarm
    `;
  } catch (err) {
    el.textContent = 'Feedback dataset: khong tai duoc thong ke.';
  }
}

function fmtNum(value, digits = 2) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : 'N/A';
}

function modelLabel(key) {
  if (key === 'faster_rcnn') return 'Faster R-CNN';
  if (key === 'yolov12') return 'YOLOv12';
  if (key === 'ssd') return 'SSD';
  return String(key || '').toUpperCase();
}

function unitBlock(title, icon, unit) {
  if (!unit || !unit.name) {
    return `
      <div class="dispatch-unit">
        <div class="dispatch-title">${icon} ${title}</div>
        <div class="incident-meta">Chua co don vi phu hop</div>
      </div>`;
  }

  return `
    <div class="dispatch-unit">
      <div class="dispatch-title">${icon} ${title}</div>
      <div class="incident-meta">
        <b>${unit.name}</b><br>
        ${unit.district || 'TP.HCM'} · ${fmtNum(unit.distance_km)} km · ETA ${unit.eta_str || 'N/A'}<br>
        Hotline: <b>${unit.hotline || 'N/A'}</b>
      </div>
    </div>`;
}

function dispatchPlanHtml(inc) {
  const plan = inc.dispatch_plan || {};
  return `
    <div class="dispatch-plan">
      <div class="dispatch-heading">Don vi phan ung nhanh</div>
      <div class="dispatch-grid">
        ${unitBlock('CSGT gan nhat', '🚓', plan.csgt)}
        ${unitBlock('Xe cuu thuong gan nhat', '🚑', plan.ambulance)}
      </div>
    </div>`;
}

function dispatchLogHtml(inc) {
  const logs = inc.dispatch_log || [];
  if (!logs.length) return '';
  return `
    <div class="dispatch-log">
      ${logs.map(log => `
        <div class="incident-meta">
          📡 ${log.dispatched_at || ''}: ${log.csgt || 'CSGT'} + ${log.ambulance || 'Ambulance'}
        </div>
      `).join('')}
    </div>`;
}

function cascadeHtml(inc) {
  const c = inc.cascade || {};
  if (!c.mode) return '';
  const verifier = c.verifier_ran ? 'Da xac minh bang Faster R-CNN' : 'Bo qua verifier vi model nhanh khong nghi ngo';
  return `
    <div class="cascade-box">
      <div><b>Cascade:</b> ${c.steps?.join(' -> ') || 'N/A'}</div>
      <div>${verifier}</div>
      <div>Latency: <b>${fmtNum(c.latency_ms, 1)} ms</b> · FPS: <b>${fmtNum(c.fps, 1)}</b></div>
    </div>`;
}

function incidentVideoHtml(inc) {
  if (!inc.video_url) return '';
  return `
    <div class="evidence-video-frame wide">
      <div class="evidence-video-head">
        <span>EVIDENCE VIDEO</span>
        <b>${inc.camera_id || 'CAMERA'}</b>
      </div>
      <video src="${inc.video_url}" controls muted autoplay loop playsinline></video>
      <div class="evidence-video-foot">Auto replay · confirmed incident evidence</div>
    </div>`;
}

function visibleIncidents(incidents) {
  const open = (incidents || [])
    .filter(inc => inc && !['RESOLVED', 'FALSE_ALARM'].includes(inc.status));
  const byCamera = new Map();
  open.forEach(inc => {
    const key = inc.camera_id || inc.id;
    if (!byCamera.has(key)) byCamera.set(key, inc);
  });
  return Array.from(byCamera.values()).sort((a, b) =>
    Number(b.priority_score || 0) - Number(a.priority_score || 0)
  );
}

function renderIncidents(incidents) {
  const list = document.getElementById('incident-list');
  const count = document.getElementById('map-incident-count');
  const activeIncidents = visibleIncidents(incidents);
  if (count) count.textContent = activeIncidents.length;
  if (!list) return;

  if (!activeIncidents.length) {
    list.innerHTML = '<div class="incident-empty">Chua co phieu su co khan cap.</div>';
    if (typeof window.renderIncidentRoutes === 'function') {
      window.renderIncidentRoutes([]);
    }
    return;
  }

  list.innerHTML = activeIncidents.map(inc => {
    const modelRows = Object.entries(inc.models || {}).map(([key, m]) => {
      if (m.skipped) return `${modelLabel(key)}: skipped by cascade`;
      const ok = m.accident ? 'Co tai nan' : 'Binh thuong';
      return `${modelLabel(key)}: ${ok} (${fmtNum(m.confidence)})`;
    }).join('<br>');

    const dispatched = inc.status === 'DISPATCHED';
    const verified = inc.status === 'VERIFIED' || dispatched;
    const verifyButton = verified
      ? '<button disabled>Da xac nhan</button>'
      : `<button onclick="setIncidentStatus('${inc.id}','VERIFIED')">Xac nhan</button>`;
    const dispatchButton = dispatched
      ? '<button disabled>📡 Da dieu phoi</button>'
      : `<button class="btn-dispatch" onclick="dispatchIncident('${inc.id}')">🚓 Bao CSGT + 🚑 Dieu cuu thuong</button>`;

    return `
      <div class="incident-card">
        <div class="incident-head">
          <div>
            <div class="incident-title">${inc.id} · ${inc.camera_name}</div>
            <div class="incident-meta">
              ${inc.time} · ${fmtNum(inc.lat, 4)}, ${fmtNum(inc.lng, 4)}<br>
              Score: <b>${fmtNum(inc.ensemble_score)}</b> · Votes: <b>${inc.votes}/3</b> · Priority: <b>${inc.priority_score || 0}/100</b><br>
              ${inc.reason || ''}<br>${modelRows}
            </div>
          </div>
          <div style="display:flex;flex-direction:column;gap:6px;align-items:flex-end">
            <span class="incident-priority">P${inc.priority_score || 0}</span>
            <span class="incident-status">${inc.status}</span>
          </div>
        </div>
        ${inc.image ? `<img src="${inc.image}" alt="Incident preview" style="width:100%;max-height:220px;object-fit:contain;background:#f8fafc;border-radius:8px;margin-top:4px">` : ''}
        ${incidentVideoHtml(inc)}
        ${cascadeHtml(inc)}
        ${dispatchPlanHtml(inc)}
        ${dispatchLogHtml(inc)}
        <div class="incident-actions">
          ${verifyButton}
          ${dispatchButton}
          <button onclick="setIncidentStatus('${inc.id}','RESOLVED')">Da xu ly</button>
          <button class="btn-stop" onclick="setIncidentStatus('${inc.id}','FALSE_ALARM')">Bao nham</button>
        </div>
      </div>`;
  }).join('');

  if (typeof window.renderIncidentRoutes === 'function') {
    window.renderIncidentRoutes(activeIncidents);
  }
}

async function setIncidentStatus(id, status) {
  const res = await fetch(`/incident/${id}/status`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ status }),
  });
  const data = await res.json();
  if (!res.ok || data.error) {
    alert(data.error || 'Khong cap nhat duoc trang thai');
    return;
  }
  await loadIncidents();
  await loadFeedbackStats();
}

async function dispatchIncident(id) {
  const res = await fetch(`/incident/${id}/dispatch`, { method: 'POST' });
  const data = await res.json();
  if (!res.ok || data.error) {
    alert(data.error || 'Khong dieu phoi duoc su co');
    return;
  }
  await loadIncidents();
}

async function loadEmergencyUnits() {
  const res = await fetch('/emergency_units');
  return res.json();
}

window.loadIncidents = loadIncidents;
window.renderIncidents = renderIncidents;
window.setIncidentStatus = setIncidentStatus;
window.dispatchIncident = dispatchIncident;
window.loadEmergencyUnits = loadEmergencyUnits;
window.loadFeedbackStats = loadFeedbackStats;

document.addEventListener('DOMContentLoaded', loadFeedbackStats);
