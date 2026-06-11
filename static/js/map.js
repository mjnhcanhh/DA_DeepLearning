let cameraMap = null;
let cameraMarkers = {};
let cameraMapBuilt = false;
let cameraList = [];
let emergencyUnitsCache = null;
let emergencyUnitLayer = null;
let incidentRouteLayer = null;
let roadRouteCache = {};

function markerIcon(status) {
  return L.divIcon({
    className: '',
    html: `<div class="camera-marker ${status || 'normal'}">📷</div>`,
    iconSize: [34, 34],
    iconAnchor: [17, 17],
  });
}

function cameraPopup(cam, statusText = 'Bình thường', extra = '') {
  return `
    <b>${cam.id || cam.camera_id}</b><br>
    ${cam.name || cam.camera_name}<br>
    Trạng thái: ${statusText}<br>
    ${extra}
  `;
}

function videoReplayHtml(url) {
  if (!url) return '';
  return `
    <div class="evidence-video-frame">
      <div class="evidence-video-head">
        <span>LIVE EVIDENCE REPLAY</span>
        <b>ACCIDENT</b>
      </div>
      <video src="${url}" controls muted autoplay loop playsinline
             style="width:100%;display:block;background:#000"></video>
      <div class="evidence-video-foot">Auto replay · muted for browser safety</div>
    </div>`;
}

function playCameraAlarm() {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return;
    const ctx = new AudioCtx();
    [0, 220, 520, 740, 1120, 1340].forEach((delay, idx) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'square';
      osc.frequency.value = idx % 2 ? 660 : 920;
      gain.gain.setValueAtTime(0.0001, ctx.currentTime + delay / 1000);
      gain.gain.exponentialRampToValueAtTime(0.075, ctx.currentTime + delay / 1000 + 0.03);
      gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + delay / 1000 + 0.24);
      osc.connect(gain).connect(ctx.destination);
      osc.start(ctx.currentTime + delay / 1000);
      osc.stop(ctx.currentTime + delay / 1000 + 0.28);
    });
    setTimeout(() => ctx.close(), 1900);
  } catch (err) {}
}

function unitIcon(type) {
  const icon = type === 'ambulance' ? '🚑' : '🚓';
  const cls = type === 'ambulance' ? 'ambulance' : 'csgt';
  return L.divIcon({
    className: '',
    html: `<div class="unit-marker ${cls}">${icon}</div>`,
    iconSize: [32, 32],
    iconAnchor: [16, 16],
  });
}

function incidentIcon(status) {
  return L.divIcon({
    className: '',
    html: `<div class="incident-map-marker ${status === 'DISPATCHED' ? 'dispatched' : 'new'}">🚨</div>`,
    iconSize: [36, 36],
    iconAnchor: [18, 18],
  });
}

function unitPopup(unit, label) {
  return `
    <b>${label}</b><br>
    ${unit.name}<br>
    ${unit.district || 'TP.HCM'}<br>
    Hotline: <b>${unit.hotline || 'N/A'}</b>
  `;
}

async function loadEmergencyUnitsOnMap() {
  if (!window.L || !cameraMap) return null;
  if (emergencyUnitsCache) return emergencyUnitsCache;
  const res = await fetch('/emergency_units');
  emergencyUnitsCache = await res.json();
  return emergencyUnitsCache;
}

async function renderEmergencyUnitMarkers() {
  if (!window.L || !cameraMap) return;
  const units = await loadEmergencyUnitsOnMap();
  if (!units) return;

  if (emergencyUnitLayer) emergencyUnitLayer.remove();
  emergencyUnitLayer = L.layerGroup().addTo(cameraMap);

  (units.csgt || []).forEach(unit => {
    L.marker([unit.lat, unit.lng], { icon: unitIcon('csgt') })
      .bindPopup(unitPopup(unit, 'CSGT'))
      .addTo(emergencyUnitLayer);
  });

  (units.ambulance || []).forEach(unit => {
    L.marker([unit.lat, unit.lng], { icon: unitIcon('ambulance') })
      .bindPopup(unitPopup(unit, 'Cuu thuong'))
      .addTo(emergencyUnitLayer);
  });
}

function routePopup(type, unit, inc) {
  const label = type === 'ambulance' ? 'Tuyen cuu thuong' : 'Tuyen CSGT';
  const icon = type === 'ambulance' ? '🚑' : '🚓';
  const distance = unit.route_distance_km || unit.distance_km || 'N/A';
  const eta = unit.route_eta_str || unit.eta_str || 'N/A';
  const mode = unit.route_mode || 'duong chim bay';
  return `
    <b>${icon} ${label}</b><br>
    Tu: ${unit.name}<br>
    Den: ${inc.camera_name}<br>
    Kieu tuyen: <b>${mode}</b><br>
    Khoang cach: <b>${distance} km</b><br>
    ETA: <b>${eta}</b>
  `;
}

function incidentPopup(inc) {
  const disabled = inc.status === 'DISPATCHED' ? 'disabled' : '';
  const text = inc.status === 'DISPATCHED' ? '📡 Da dieu phoi' : '🚓 Bao CSGT + 🚑 Dieu cuu thuong';
  return `
    <b>${inc.id}</b><br>
    ${inc.camera_name}<br>
    Trang thai: <b>${inc.status}</b><br>
    Score: ${inc.ensemble_score || 'N/A'} · Votes: ${inc.votes || 0}/3<br>
    Priority: <b>${inc.priority_score || 0}/100</b><br>
    ${inc.cascade ? `Cascade: ${inc.cascade.steps?.join(' -> ') || 'N/A'}<br>` : ''}
    ${videoReplayHtml(inc.video_url)}
    <button ${disabled} style="margin-top:8px;padding:6px 10px;border-radius:6px;border:1px solid #2563eb;background:#2563eb;color:#fff;cursor:pointer"
      onclick="dispatchIncidentFromMap('${inc.id}')">${text}</button>
  `;
}

function clearIncidentRoutes() {
  if (incidentRouteLayer) incidentRouteLayer.remove();
  if (cameraMap) incidentRouteLayer = L.layerGroup().addTo(cameraMap);
}

function routeCacheKey(inc, type, unit) {
  return [
    type,
    Number(unit.lat).toFixed(5),
    Number(unit.lng).toFixed(5),
    Number(inc.lat).toFixed(5),
    Number(inc.lng).toFixed(5),
  ].join(':');
}

function formatDuration(seconds) {
  const minutes = Math.max(1, Math.round(Number(seconds || 0) / 60));
  if (minutes < 60) return `${minutes} phut`;
  const hours = Math.floor(minutes / 60);
  const rest = minutes % 60;
  return rest ? `${hours} gio ${rest} phut` : `${hours} gio`;
}

async function fetchRoadRoute(inc, type, unit) {
  const key = routeCacheKey(inc, type, unit);
  if (roadRouteCache[key]) return roadRouteCache[key];

  const coords = `${unit.lng},${unit.lat};${inc.lng},${inc.lat}`;
  const url = `https://router.project-osrm.org/route/v1/driving/${coords}?overview=full&geometries=geojson&steps=false`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`OSRM ${res.status}`);
  const data = await res.json();
  const route = data.routes && data.routes[0];
  if (!route || !route.geometry || !route.geometry.coordinates) {
    throw new Error('Khong co tuyen duong');
  }

  const result = {
    latlngs: route.geometry.coordinates.map(([lng, lat]) => [lat, lng]),
    distance_km: (Number(route.distance || 0) / 1000).toFixed(2),
    eta_str: formatDuration(route.duration),
  };
  roadRouteCache[key] = result;
  return result;
}

async function drawDispatchRoute(inc, type, unit) {
  if (!unit || !unit.lat || !unit.lng || !inc.lat || !inc.lng) return;
  const color = type === 'ambulance' ? '#dc2626' : '#2563eb';
  let latlngs = [[unit.lat, unit.lng], [inc.lat, inc.lng]];
  let popupUnit = { ...unit, route_mode: 'duong chim bay' };

  try {
    const road = await fetchRoadRoute(inc, type, unit);
    latlngs = road.latlngs;
    popupUnit = {
      ...unit,
      route_distance_km: road.distance_km,
      route_eta_str: road.eta_str,
      route_mode: 'duong bo that',
    };
  } catch (err) {
    popupUnit = { ...popupUnit, route_error: err.message };
  }

  L.polyline(latlngs, {
    color,
    weight: 4,
    opacity: 0.85,
    dashArray: popupUnit.route_mode === 'duong bo that' ? null : (type === 'ambulance' ? '8 6' : '2 8'),
  }).bindPopup(routePopup(type, popupUnit, inc)).addTo(incidentRouteLayer);
}

async function renderIncidentRoutes(incidents = []) {
  if (!window.L || !cameraMap) return;
  clearIncidentRoutes();

  cameraList.forEach(cam => {
    const marker = cameraMarkers[cam.id];
    if (marker) {
      marker.setIcon(markerIcon('normal'));
      marker.bindPopup(cameraPopup(cam));
    }
  });

  const active = incidents
    .filter(inc => inc && inc.lat && inc.lng && !['RESOLVED', 'FALSE_ALARM'].includes(inc.status));

  active.forEach(inc => {
    L.marker([inc.lat, inc.lng], { icon: incidentIcon(inc.status) })
      .bindPopup(incidentPopup(inc))
      .addTo(incidentRouteLayer);

    const marker = cameraMarkers[inc.camera_id];
    if (marker) {
      marker.setIcon(markerIcon('accident'));
      marker.bindPopup(incidentPopup(inc));
    }
  });

  const jobs = [];
  active.filter(inc => inc.status === 'DISPATCHED').forEach(inc => {
    const plan = inc.dispatch_plan || {};
    jobs.push(drawDispatchRoute(inc, 'csgt', plan.csgt));
    jobs.push(drawDispatchRoute(inc, 'ambulance', plan.ambulance));
  });
  await Promise.allSettled(jobs);
}

async function dispatchIncidentFromMap(id) {
  if (typeof window.dispatchIncident === 'function') {
    await window.dispatchIncident(id);
  } else {
    await fetch(`/incident/${id}/dispatch`, { method: 'POST' });
    await loadIncidents();
  }
}

async function renderCameraHealth() {
  const list = document.getElementById('camera-health-list');
  if (!list) return;
  try {
    const res = await fetch('/camera_health');
    const health = await res.json();
    const scanned = health
      .filter(h => h.last_scan)
      .sort((a, b) => String(b.last_scan || '').localeCompare(String(a.last_scan || '')))
      .slice(0, 6);
    const online = health.filter(h => h.online).length;
    const count2 = document.getElementById('map-camera-count-2');
    if (count2) count2.textContent = `${online}/${health.length}`;

    if (!scanned.length) {
      list.innerHTML = '<div class="incident-empty">Chưa có dữ liệu health.</div>';
      return;
    }

    list.innerHTML = scanned.map(h => {
      const cls = h.status === 'accident' ? 'accident' : (h.online ? 'online' : '');
      const verifier = h.cascade?.verifier_ran ? 'Verifier: Faster R-CNN' : 'Verifier: skipped';
      return `
        <div class="health-card ${cls}">
          <div class="health-title">
            <span>${h.camera_id}</span>
            <span>${h.status}</span>
          </div>
          <div class="health-meta">
            ${h.camera_name}<br>
            FPS: <b>${h.fps ?? 'N/A'}</b> · Latency: <b>${h.latency_ms ?? 'N/A'} ms</b><br>
            ${verifier}<br>
            ${h.last_scan || ''}
          </div>
        </div>`;
    }).join('');
  } catch (err) {
    list.innerHTML = `<div class="incident-empty">Khong tai duoc health: ${err.message}</div>`;
  }
}

async function initCameraMap() {
  const el = document.getElementById('camera-map');
  if (!el) return;

  const res = await fetch('/cameras');
  const cameras = await res.json();
  cameraList = cameras;
  const count = document.getElementById('map-camera-count');
  if (count) count.textContent = cameras.length;

  if (!window.L) {
    renderOfflineMap(cameras);
    cameraMapBuilt = true;
    await renderCameraHealth();
    await loadIncidents();
    return;
  }

  if (!cameraMapBuilt) {
    cameraMap = L.map('camera-map').setView([10.7769, 106.7009], 12);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap',
    }).addTo(cameraMap);

    cameras.forEach(cam => {
      const marker = L.marker([cam.lat, cam.lng], { icon: markerIcon('normal') }).addTo(cameraMap);
      marker.bindPopup(cameraPopup(cam));
      cameraMarkers[cam.id] = marker;
    });
    cameraMapBuilt = true;
  }

  setTimeout(() => cameraMap.invalidateSize(), 80);
  await renderEmergencyUnitMarkers();
  await renderCameraHealth();
  await loadIncidents();
}

function renderOfflineMap(cameras) {
  const el = document.getElementById('camera-map');
  if (!el) return;

  el.innerHTML = `
    <div class="offline-map">
      <div class="offline-map-title">TP.HCM Camera Network</div>
      ${cameras.map(cam => `
        <button
          class="offline-camera-pin normal"
          id="offline-${cam.id}"
          style="left:${cameraX(cam.lng)}%;top:${cameraY(cam.lat)}%"
          title="${cam.id} - ${cam.name}">
          <span></span>
          📷
          <b>${cam.id.replace('CAM_', '')}</b>
        </button>
      `).join('')}
    </div>
  `;
}

function cameraX(lng) {
  const min = 106.62;
  const max = 106.75;
  return Math.max(7, Math.min(93, ((lng - min) / (max - min)) * 86 + 7));
}

function cameraY(lat) {
  const min = 10.74;
  const max = 10.83;
  return Math.max(8, Math.min(92, 92 - ((lat - min) / (max - min)) * 84));
}

function updateOfflinePin(result) {
  const pin = document.getElementById(`offline-${result.camera_id}`);
  if (!pin) return;
  pin.classList.remove('normal', 'accident', 'no-source');
  pin.classList.add(result.accident ? 'accident' : (result.status === 'no_source' ? 'no-source' : 'normal'));
  pin.title = `${result.camera_id} - ${result.camera_name} | ${result.status} | score ${result.ensemble_score}`;
}

function updateScanSummary(results) {
  const el = document.getElementById('scan-summary');
  if (!el) return;
  const accident = results.filter(r => r.accident).length;
  const missing = results.filter(r => r.status === 'no_source').length;
  el.innerHTML = `
    Đã quét <b>${results.length}</b> camera.<br>
    Tai nạn: <b style="color:#dc2626">${accident}</b><br>
    Chưa có ảnh demo: <b>${missing}</b>
  `;
}

async function scanCameras() {
  if (!cameraMapBuilt) await initCameraMap();

  const btn = document.getElementById('scan-cameras-btn');
  if (btn) {
    btn.disabled = true;
    btn.textContent = 'Đang quét...';
  }

  try {
    const res = await fetch('/scan_cameras', { method: 'POST' });
    const data = await res.json();
    const results = data.results || [];

    results.forEach(r => {
      const marker = cameraMarkers[r.camera_id];

      const statusClass = r.accident ? 'accident' : (r.status === 'no_source' ? 'no-source' : 'normal');
      const statusText = r.accident ? '🚨 Tai nạn' : (r.status === 'no_source' ? 'Chưa có ảnh demo' : '✅ Bình thường');
      const extra = `
        Ensemble score: ${r.ensemble_score}<br>
        Votes: ${r.votes}/3<br>
        Priority: ${r.priority_score || 0}/100<br>
        ${r.cascade ? `Cascade: ${r.cascade.steps?.join(' -> ') || 'N/A'}<br>` : ''}
        ${videoReplayHtml(r.video_url)}
        ${r.incident_id ? 'Mã sự cố: ' + r.incident_id + '<br>' : ''}
        ${r.message ? r.message + '<br>' : ''}
      `;
      if (marker) {
        marker.setIcon(markerIcon(statusClass));
        marker.bindPopup(cameraPopup(r, statusText, extra));
      }
      updateOfflinePin(r);
    });

    updateScanSummary(results);
    renderIncidents(data.incidents || []);
    await renderCameraHealth();
  } catch (err) {
    alert('Không quét được camera: ' + err.message);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = '🔍 Quét toàn bộ camera';
    }
  }
}

function applyScanResult(r) {
  const marker = cameraMarkers[r.camera_id];
  const statusClass = r.accident ? 'accident' : (r.status === 'no_source' ? 'no-source' : 'normal');
  const statusText = r.accident ? '🚨 Tai nạn' : (r.status === 'no_source' ? 'Chưa có ảnh demo' : '✅ Bình thường');
  const extra = `
    Ensemble score: ${r.ensemble_score}<br>
    Votes: ${r.votes}/3<br>
    Priority: ${r.priority_score || 0}/100<br>
    ${r.cascade ? `Cascade: ${r.cascade.steps?.join(' -> ') || 'N/A'}<br>` : ''}
    ${videoReplayHtml(r.video_url)}
    ${r.incident_id ? 'Mã sự cố: ' + r.incident_id + '<br>' : ''}
    ${r.message ? r.message + '<br>' : ''}
  `;
  if (marker) {
    marker.setIcon(markerIcon(statusClass));
    marker.bindPopup(cameraPopup(r, statusText, extra));
    marker.openPopup();
  }
  updateOfflinePin(r);
  if (r.accident) playCameraAlarm();
}

async function uploadRandomCameraImage(input) {
  const file = input.files && input.files[0];
  if (!file) return;
  if (!cameraMapBuilt) await initCameraMap();

  const summary = document.getElementById('scan-summary');
  if (summary) summary.innerHTML = 'Đang gán ảnh vào camera ngẫu nhiên và chạy ensemble...';

  const fd = new FormData();
  fd.append('file', file);

  try {
    const res = await fetch('/scan_random_camera_image', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || 'Upload thất bại');

    applyScanResult(data.result);
    if (summary) {
      summary.innerHTML = `
        Ảnh <b>${file.name}</b> đã được gán vào <b>${data.result.camera_id}</b>.<br>
        Kết quả: <b style="color:${data.result.accident ? '#dc2626' : '#16a34a'}">${data.result.accident ? 'Tai nạn' : 'Bình thường'}</b><br>
        Ensemble score: <b>${data.result.ensemble_score}</b> · Votes: <b>${data.result.votes}/3</b>
      `;
    }
    renderIncidents(data.incidents || []);
    await renderCameraHealth();
  } catch (err) {
    alert('Không xử lý được ảnh: ' + err.message);
  } finally {
    input.value = '';
  }
}

window.initCameraMap = initCameraMap;
window.scanCameras = scanCameras;
window.uploadRandomCameraImage = uploadRandomCameraImage;
window.renderIncidentRoutes = renderIncidentRoutes;
window.dispatchIncidentFromMap = dispatchIncidentFromMap;
window.renderEmergencyUnitMarkers = renderEmergencyUnitMarkers;
window.renderCameraHealth = renderCameraHealth;

document.addEventListener('DOMContentLoaded', () => {
  const activeMap = document.getElementById('tab-map')?.classList.contains('active');
  if (activeMap) initCameraMap();
});
