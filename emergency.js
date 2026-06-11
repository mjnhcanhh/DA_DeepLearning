/**
 * emergency.js — Cập nhật: hiển thị dispatch plan (CSGT + cứu thương)
 * Thay thế file emergency.js cũ hoàn toàn.
 */

async function loadIncidents() {
  const list = document.getElementById('incident-list');
  if (!list) return;
  try {
    const res = await fetch('/incidents');
    const incidents = await res.json();
    renderIncidents(incidents);
  } catch (err) {
    list.innerHTML = `<div class="incident-empty">Không tải được danh sách sự cố: ${err.message}</div>`;
  }
}

// ─── Render dispatch plan block ─────────────────────────────────────────────
function renderDispatchPlan(plan) {
  if (!plan || (!plan.csgt && !plan.ambulance)) return '';

  const csgt = plan.csgt || {};
  const amb  = plan.ambulance || {};

  return `
    <div class="dispatch-plan">
      <div class="dispatch-title">📡 Đơn vị phản ứng nhanh</div>
      <div class="dispatch-row">
        <div class="dispatch-unit csgt-unit">
          <span class="unit-icon">🚓</span>
          <div class="unit-info">
            <div class="unit-name">${csgt.name || '—'}</div>
            <div class="unit-meta">${csgt.district || ''} · ${csgt.distance_km || '?'} km · <b>${csgt.eta_str || '?'}</b></div>
            <div class="unit-hotline">☎ <a href="tel:${csgt.hotline}">${csgt.hotline || '—'}</a></div>
          </div>
        </div>
        <div class="dispatch-unit amb-unit">
          <span class="unit-icon">🚑</span>
          <div class="unit-info">
            <div class="unit-name">${amb.name || '—'}</div>
            <div class="unit-meta">${amb.district || ''} · ${amb.distance_km || '?'} km · <b>${amb.eta_str || '?'}</b></div>
            <div class="unit-hotline">☎ <a href="tel:${amb.hotline}">${amb.hotline || '—'}</a></div>
          </div>
        </div>
      </div>
    </div>`;
}

// ─── Render dispatch log ─────────────────────────────────────────────────────
function renderDispatchLog(log) {
  if (!log || !log.length) return '';
  const entries = log.map(e => `
    <div class="log-entry">
      ✅ ${e.dispatched_at} — <b>${e.csgt}</b> + <b>${e.ambulance}</b>
    </div>`).join('');
  return `<div class="dispatch-log"><b>Nhật ký điều phối:</b>${entries}</div>`;
}

// ─── Render toàn bộ danh sách incidents ─────────────────────────────────────
function renderIncidents(incidents) {
  const list = document.getElementById('incident-list');
  const count = document.getElementById('map-incident-count');
  if (count) count.textContent = incidents.length;
  if (!list) return;

  if (!incidents.length) {
    list.innerHTML = '<div class="incident-empty">Chưa có phiếu sự cố khẩn cấp.</div>';
    return;
  }

  list.innerHTML = incidents.map(inc => {
    const modelRows = Object.entries(inc.models || {}).map(([key, m]) => {
      const label = key === 'faster_rcnn' ? 'Faster R-CNN' : key.toUpperCase();
      const ok    = m.accident ? '🔴 Có tai nạn' : '🟢 Bình thường';
      return `${label}: ${ok} (${Number(m.confidence || 0).toFixed(2)})`;
    }).join('<br>');

    const isDispatched = ['DISPATCHED', 'RESOLVED'].includes(inc.status);

    return `
      <div class="incident-card">
        <div class="incident-head">
          <div>
            <div class="incident-title">${inc.id} · ${inc.camera_name}</div>
            <div class="incident-meta">
              ${inc.time} · ${Number(inc.lat).toFixed(4)}, ${Number(inc.lng).toFixed(4)}<br>
              Score: <b>${inc.ensemble_score}</b> · Votes: <b>${inc.votes}/3</b><br>
              ${inc.reason || ''}<br>${modelRows}
            </div>
          </div>
          <span class="incident-status ${inc.status.toLowerCase()}">${_statusLabel(inc.status)}</span>
        </div>

        ${inc.image ? `<img src="${inc.image}" alt="Incident preview"
            style="width:100%;max-height:220px;object-fit:contain;background:#f8fafc;border-radius:8px;margin-top:4px">` : ''}

        ${renderDispatchPlan(inc.dispatch_plan)}
        ${renderDispatchLog(inc.dispatch_log)}

        <div class="incident-actions">
          <button onclick="setIncidentStatus('${inc.id}','VERIFIED')">✅ Xác nhận</button>
          <button class="btn-dispatch ${isDispatched ? 'btn-dispatched' : ''}"
                  onclick="dispatchIncident('${inc.id}')"
                  ${isDispatched ? 'disabled' : ''}>
            ${isDispatched ? '📡 Đã điều phối' : '🚓 Báo CSGT + 🚑 Điều cứu thương'}
          </button>
          <button onclick="setIncidentStatus('${inc.id}','RESOLVED')">🏁 Đã xử lý</button>
          <button class="btn-stop" onclick="setIncidentStatus('${inc.id}','FALSE_ALARM')">❌ Báo nhầm</button>
        </div>
      </div>`;
  }).join('');
}

function _statusLabel(status) {
  const MAP = {
    NEW:        '🆕 Mới',
    VERIFIED:   '🔍 Đã xác nhận',
    DISPATCHED: '📡 Đang xử lý',
    RESOLVED:   '✅ Đã giải quyết',
    FALSE_ALARM:'❌ Báo nhầm',
  };
  return MAP[status] || status;
}

// ─── Actions ─────────────────────────────────────────────────────────────────
async function setIncidentStatus(id, status) {
  const res  = await fetch(`/incident/${id}/status`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ status }),
  });
  const data = await res.json();
  if (!res.ok || data.error) { alert(data.error || 'Không cập nhật được trạng thái'); return; }
  await loadIncidents();
}

async function dispatchIncident(id) {
  if (!confirm('Xác nhận điều phối CSGT và xe cứu thương đến hiện trường?')) return;

  const res  = await fetch(`/incident/${id}/dispatch`, { method: 'POST' });
  const data = await res.json();
  if (!res.ok || data.error) { alert(data.error || 'Không điều phối được'); return; }
  await loadIncidents();
}

// ─── CSS bổ sung (inject vào <head> nếu chưa có stylesheet riêng) ────────────
(function injectDispatchStyles() {
  if (document.getElementById('dispatch-style')) return;
  const style = document.createElement('style');
  style.id = 'dispatch-style';
  style.textContent = `
    .dispatch-plan {
      margin: 10px 0 6px;
      background: #f0f9ff;
      border: 1px solid #bae6fd;
      border-radius: 8px;
      padding: 10px 12px;
    }
    .dispatch-title {
      font-size: 0.78rem;
      font-weight: 700;
      color: #0369a1;
      margin-bottom: 8px;
      letter-spacing: .02em;
      text-transform: uppercase;
    }
    .dispatch-row {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .dispatch-unit {
      display: flex;
      align-items: flex-start;
      gap: 8px;
      flex: 1;
      min-width: 200px;
      background: #fff;
      border-radius: 6px;
      padding: 8px 10px;
      border: 1px solid #e2e8f0;
    }
    .csgt-unit { border-left: 3px solid #2563eb; }
    .amb-unit  { border-left: 3px solid #dc2626; }
    .unit-icon { font-size: 1.4rem; line-height: 1; }
    .unit-name { font-size: 0.82rem; font-weight: 600; color: #1e293b; }
    .unit-meta { font-size: 0.74rem; color: #64748b; margin: 2px 0; }
    .unit-hotline { font-size: 0.76rem; color: #0369a1; }
    .unit-hotline a { color: inherit; text-decoration: none; font-weight: 600; }
    .unit-hotline a:hover { text-decoration: underline; }

    .dispatch-log {
      font-size: 0.75rem;
      color: #475569;
      margin: 6px 0 0;
      padding: 6px 10px;
      background: #f8fafc;
      border-radius: 6px;
    }
    .log-entry { margin-top: 3px; }

    .btn-dispatch {
      background: linear-gradient(135deg, #1d4ed8, #dc2626);
      color: #fff !important;
      font-weight: 700;
      border: none;
      padding: 6px 14px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.82rem;
      transition: opacity .2s;
    }
    .btn-dispatch:hover:not(:disabled) { opacity: 0.88; }
    .btn-dispatch.btn-dispatched {
      background: #94a3b8;
      cursor: default;
    }
    .incident-status.dispatched { background: #dbeafe; color: #1d4ed8; }
    .incident-status.resolved   { background: #dcfce7; color: #166534; }
    .incident-status.false_alarm{ background: #fee2e2; color: #991b1b; }
    .incident-status.verified   { background: #fef9c3; color: #854d0e; }
    .incident-status.new        { background: #fce7f3; color: #9d174d; }
  `;
  document.head.appendChild(style);
})();

// ─── Exports ─────────────────────────────────────────────────────────────────
window.loadIncidents     = loadIncidents;
window.renderIncidents   = renderIncidents;
window.setIncidentStatus = setIncidentStatus;
window.dispatchIncident  = dispatchIncident;
