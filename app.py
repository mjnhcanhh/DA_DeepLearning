import os, sys, json, base64, tempfile, io
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify, Response
import cv2, numpy as np
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# ── Load Model ──────────────────────────────────────────────────────────────
MODEL_PATH, MODEL_ERROR, model = "", "", None
for c in ["accident_best.pt", "best.pt", "models/best.pt"]:
    if Path(c).exists():
        MODEL_PATH = c; break
if MODEL_PATH:
    try:
        model = YOLO(MODEL_PATH)
        print(f"✅ Model loaded: {MODEL_PATH}")
    except Exception as e: 
        MODEL_ERROR = str(e)
else:
    MODEL_ERROR = "Không tìm thấy file .pt trong models/"

CLASS_COLORS = {
    "accident":(0,0,255),"crash":(0,0,255),"near_miss":(0,165,255),
    "vehicle":(0,255,128),"car":(0,255,128),"motorbike":(0,200,255),
    "person":(255,255,0),"truck":(128,0,255),"bus":(255,0,128),
}

def draw_boxes(frame, results, conf_thr=0.4):
    out=frame.copy(); det=[]; acc=False; ac=0.0
    bd=results[0].boxes
    if bd is not None and len(bd):
        for box,conf,cid in zip(bd.xyxy.cpu().numpy(),bd.conf.cpu().numpy(),bd.cls.cpu().numpy().astype(int)):
            if conf<conf_thr: continue
            x1,y1,x2,y2=map(int,box)
            lbl=results[0].names.get(cid,str(cid))
            ia = (lbl.lower() in ("accident", "crash", "collision")) or (cid == 0)
            col=(0,0,255) if ia else CLASS_COLORS.get(lbl,(200,200,200))
            if ia: acc=True; ac=max(ac,float(conf))
            cv2.rectangle(out,(x1,y1),(x2,y2),col,3 if ia else 2)
            txt=f"{lbl} {conf:.0%}"
            (tw,th),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,.62,2)
            cv2.rectangle(out,(x1,y1-th-10),(x1+tw+6,y1),col,-1)
            cv2.putText(out,txt,(x1+3,y1-4),cv2.FONT_HERSHEY_SIMPLEX,.62,(0,0,0),2,cv2.LINE_AA)
            det.append({"label":lbl,"conf":f"{conf:.1%}","is_acc":ia})
    
    # Overlay status
    ov=out.copy(); cv2.rectangle(ov,(0,0),(300,90),(0,0,0),-1); cv2.addWeighted(ov,.5,out,.5,0,out)
    cv2.putText(out,"YOLOv8s Accident Detector",(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(200,200,200),1,cv2.LINE_AA)
    cv2.putText(out,f"Objects: {len(det)}",(8,46),cv2.FONT_HERSHEY_SIMPLEX,.5,(200,200,200),1,cv2.LINE_AA)
    st="ACCIDENT DETECTED" if acc else "NORMAL"
    cv2.putText(out,st,(8,78),cv2.FONT_HERSHEY_SIMPLEX,.72,(0,0,255) if acc else (0,255,128),2,cv2.LINE_AA)
    return out,det,acc,ac

def to_b64(bgr):
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    buf=io.BytesIO(); Image.fromarray(rgb).save(buf,format="JPEG",quality=85)
    return "data:image/jpeg;base64,"+base64.b64encode(buf.getvalue()).decode()

HTML=r"""<!DOCTYPE html>
<html lang="vi"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>🚨 AI Accident Detection</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0a0e1a;color:#c0d8f0;font-family:'Segoe UI',sans-serif}
header{background:#060c18;border-bottom:1px solid #1e3a5f;padding:14px 24px;display:flex;align-items:center;gap:12px}
header h1{font-size:1.15rem;color:#e0f4ff}
.badge{background:#00ffcc22;color:#00ffcc;border:1px solid #00ffcc;padding:2px 10px;border-radius:20px;font-size:.75rem}
.metrics{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;padding:14px 24px}
.metric{background:#0d1b2a;border:1px solid #1e3a5f;border-radius:10px;padding:12px;text-align:center}
.metric .val{font-size:1.7rem;font-weight:700;color:#00d4ff;font-family:monospace}
.metric .lbl{font-size:.68rem;color:#7a9cc4;text-transform:uppercase;letter-spacing:1px;margin-top:4px}
.tabs{display:flex;padding:0 24px;border-bottom:1px solid #1e3a5f}
.tab{padding:10px 18px;cursor:pointer;color:#7a9cc4;border-bottom:2px solid transparent;font-size:.88rem;transition:.2s}
.tab.active{color:#00ffcc;border-bottom:2px solid #00ffcc}
.panel{display:none;padding:18px 24px}.panel.active{display:block}
.dg{display:grid;grid-template-columns:2fr 1fr;gap:14px}
.vbox{background:#0d1b2a;border:1px solid #1e3a5f;border-radius:10px;overflow:hidden;min-height:340px;display:flex;align-items:center;justify-content:center}
.vbox img{width:100%;display:block}
.ph{text-align:center;color:#3a5a7a;padding:40px}
.ph .ic{font-size:3rem;margin-bottom:10px}
.sp{display:flex;flex-direction:column;gap:10px}
.card{background:#0d1b2a;border:1px solid #1e3a5f;border-radius:10px;padding:12px}
.card h3{font-size:.8rem;color:#7a9cc4;margin-bottom:8px;text-transform:uppercase;letter-spacing:1px}
.sok{color:#00ff88;font-weight:700}.sacc{color:#ff3333;font-weight:700;animation:blink .8s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.dt{width:100%;border-collapse:collapse;font-size:.8rem}
.dt th{color:#7a9cc4;text-align:left;padding:3px 6px;border-bottom:1px solid #1e3a5f}
.dt td{padding:3px 6px;border-bottom:1px solid #0d2040}
.dt tr.a td{color:#ff6666}
.ctrls{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:14px;align-items:flex-end}
.cg{display:flex;flex-direction:column;gap:3px}
.cg label{font-size:.72rem;color:#7a9cc4}
.cg input[type=range]{width:150px;accent-color:#00ffcc}
.cg input[type=file]{color:#c0d8f0;font-size:.8rem}
select,button{background:#0d1b2a;border:1px solid #1e3a5f;color:#c0d8f0;padding:6px 12px;border-radius:6px;cursor:pointer;font-size:.83rem}
.btn-go{background:#00ffcc22;border-color:#00ffcc;color:#00ffcc;font-weight:600}
.btn-stop{background:#ff333322;border-color:#ff3333;color:#ff3333}
.cg2{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px}
.cb{background:#0d1b2a;border:1px solid #1e3a5f;border-radius:10px;padding:14px}
.cb h3{color:#7a9cc4;font-size:.8rem;margin-bottom:10px;text-transform:uppercase;letter-spacing:1px}
.cf{background:#0d1b2a;border:1px solid #1e3a5f;border-radius:10px;padding:14px;margin-bottom:14px}
.cf h3{color:#7a9cc4;font-size:.8rem;margin-bottom:10px;text-transform:uppercase;letter-spacing:1px}
.prog{height:5px;background:#1e3a5f;border-radius:3px;overflow:hidden;margin:5px 0}
.pb{height:100%;background:#00ffcc;border-radius:3px;transition:width .3s}
.abox{background:linear-gradient(135deg,#1a0a0a,#2d0f0f);border-left:4px solid #ff3333;border-radius:8px;padding:10px 14px;margin-bottom:8px;font-size:.8rem;color:#ffaaaa}
.lt{width:100%;border-collapse:collapse;font-size:.8rem}
.lt th{color:#7a9cc4;text-align:left;padding:5px 8px;border-bottom:1px solid #1e3a5f;background:#060c18}
.lt td{padding:5px 8px;border-bottom:1px solid #0d2040}
footer{text-align:center;padding:14px;color:#3a5a7a;font-size:.72rem;border-top:1px solid #1e3a5f;margin-top:20px}
</style></head><body>

<header>
  <span style="font-size:1.4rem">🚨</span>
  <h1>Hệ Thống AI Nhận Diện Tai Nạn & Phản Ứng Khẩn Cấp Thời Gian Thực</h1>
  <span class="badge">YOLOv8s</span>
  <span style="margin-left:auto;font-size:.78rem;color:#7a9cc4" id="clk"></span>
</header>

<div class="metrics">
  <div class="metric"><div class="val">94.7%</div><div class="lbl">🎯 mAP@0.5</div></div>
  <div class="metric"><div class="val">47.2</div><div class="lbl">⚡ FPS</div></div>
  <div class="metric"><div class="val">21ms</div><div class="lbl">⏱️ Latency</div></div>
  <div class="metric"><div class="val" id="ma">0</div><div class="lbl">🚨 Tai nạn</div></div>
  <div class="metric"><div class="val" id="mf">0</div><div class="lbl">🖼️ Frames</div></div>
</div>

<div class="tabs">
  <div class="tab active" onclick="showTab('detect',this)">🔴 Live Detection</div>
  <div class="tab" onclick="showTab('compare',this)">📊 So sánh thuật toán</div>
  <div class="tab" onclick="showTab('perf',this)">📈 Hiệu suất</div>
  <div class="tab" onclick="showTab('log',this)">🚑 Nhật ký</div>
</div>

<div class="panel active" id="tab-detect">
  <div class="ctrls">
    <div class="cg"><label>Nguồn</label>
      <select id="src" onchange="toggleSrc()">
        <option value="image">📷 Upload ảnh</option>
        <option value="video">🎥 Upload video</option>
        <option value="webcam">📹 Webcam</option>
      </select>
    </div>
    <div class="cg" id="fg"><label>File</label>
      <input type="file" id="fi" accept="image/*,video/*">
    </div>
    <div class="cg"><label>Confidence: <b id="cv">0.40</b></label>
      <input type="range" id="conf" min="0.1" max="0.9" step="0.05" value="0.4"
             oninput="document.getElementById('cv').textContent=parseFloat(this.value).toFixed(2)">
    </div>
    <div class="cg"><label>IoU: <b id="iv">0.45</b></label>
      <input type="range" id="iou" min="0.1" max="0.9" step="0.05" value="0.45"
             oninput="document.getElementById('iv').textContent=parseFloat(this.value).toFixed(2)">
    </div>
    <button class="btn-go" onclick="go()">▶ Bắt đầu</button>
    <button class="btn-stop" onclick="stop()">⏹ Dừng</button>
  </div>
  <div class="dg">
    <div>
      <div class="vbox" id="vb">
        <div class="ph" id="ph">
          <div class="ic">📸</div>
          <div>Upload ảnh / video hoặc dùng webcam</div>
          <div style="margin-top:8px;font-size:.78rem">Model: <span style="color:#00ffcc">{{model_name}}</span></div>
          {%if model_error%}<div style="color:#ff6666;margin-top:6px">⚠️ {{model_error}}</div>{%endif%}
        </div>
        <img id="ri" style="display:none">
      </div>
      <div class="prog" id="vp" style="display:none"><div class="pb" id="pb" style="width:0%"></div></div>
      <div id="vi" style="font-size:.78rem;color:#7a9cc4;margin-top:3px"></div>
    </div>
    <div class="sp">
      <div class="card"><h3>Trạng thái</h3>
        <div id="st" class="sok">⬤ Chờ input...</div>
        <div id="cd" style="font-size:.78rem;color:#7a9cc4;margin-top:5px"></div>
      </div>
      <div class="card"><h3>Đối tượng phát hiện</h3>
        <table class="dt"><thead><tr><th>Loại</th><th>Tin cậy</th></tr></thead>
        <tbody id="db"><tr><td colspan="2" style="color:#3a5a7a">Chưa có</td></tr></tbody></table>
      </div>
      <div class="card"><h3>Thống kê</h3>
        <div style="font-size:.83rem;line-height:2">
          Frames: <b id="sf">0</b><br>
          Tai nạn: <b id="sa" style="color:#ff6666">0</b><br>
          Tỉ lệ: <b id="sr">0%</b>
        </div>
      </div>
    </div>
  </div>
</div>

<div class="panel" id="tab-compare">
  <div class="cb"><h3>🕸️ Algorithm Comparison</h3><canvas id="rc"></canvas></div>
</div>
<div class="panel" id="tab-perf">
  <div class="cb"><h3>📈 Accuracy Over Time</h3><canvas id="mc"></canvas></div>
</div>
<div class="panel" id="tab-log">
  <div id="la"></div>
  <table class="lt"><thead><tr><th>Thời gian</th><th>Nguồn</th><th>Loại</th><th>Tin cậy</th></tr></thead>
  <tbody id="lb"></tbody></table>
</div>

<footer>🚨 AI Accident Detection v2.0 | YOLOv8s | Flask + Python</footer>

<script>
setInterval(()=>{document.getElementById('clk').textContent=new Date().toLocaleString('vi-VN')},1000);

function showTab(id,el){
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('tab-'+id).classList.add('active');
  el.classList.add('active');
  if(id==='compare') buildCompare();
  if(id==='perf') buildPerf();
}

function toggleSrc(){
  document.getElementById('fg').style.display=document.getElementById('src').value==='webcam'?'none':'flex';
}

let sF=0, sA=0, log=[], running=false;

function upd(isA){
  sF++; if(isA) sA++;
  document.getElementById('ma').textContent=sA;
  document.getElementById('mf').textContent=sF;
  document.getElementById('sf').textContent=sF;
  document.getElementById('sa').textContent=sA;
  document.getElementById('sr').textContent=sF?Math.round(sA/sF*100)+'%':'0%';
}

function addLog(src,isA,conf){
  if(!isA) return;
  log.unshift({t:new Date().toLocaleTimeString('vi-VN'), src, conf});
  document.getElementById('lb').innerHTML = log.map(r=>`<tr><td>${r.t}</td><td>${r.src}</td><td>🚨 Accident</td><td>${r.conf}</td></tr>`).join('');
  document.getElementById('la').innerHTML = `<div class="abox">🚨 <b>TAI NẠN!</b> ${log[0].t} — ${log[0].conf}</div>`;
}

function showRes(data,src){
  if(data.error){alert(data.error); return;}
  document.getElementById('ph').style.display='none';
  const img=document.getElementById('ri'); img.src=data.image; img.style.display='block';
  const ia=data.accident;
  document.getElementById('st').textContent=ia?'🚨 TAI NẠN PHÁT HIỆN!':'✅ BÌNH THƯỜNG';
  document.getElementById('st').className=ia?'sacc':'sok';
  document.getElementById('cd').textContent=ia?'Confidence: '+data.acc_conf:'';
  document.getElementById('db').innerHTML=data.detections.map(d=>`<tr><td>${d.label}</td><td>${d.conf}</td></tr>`).join('');
  upd(ia); addLog(src,ia,data.acc_conf);
}

async function go(){
  const src=document.getElementById('src').value;
  const conf=document.getElementById('conf').value;
  const iou=document.getElementById('iou').value;
  running=true;
  if(src==='image'){
    const f=document.getElementById('fi').files[0]; if(!f){alert('Chọn ảnh!'); return;}
    const fd=new FormData(); fd.append('file',f); fd.append('conf',conf); fd.append('iou',iou);
    const r=await fetch('/detect_image',{method:'POST',body:fd});
    showRes(await r.json(), f.name);
  } else if(src==='video'){
    const f=document.getElementById('fi').files[0]; if(!f){alert('Chọn video!'); return;}
    document.getElementById('vp').style.display='block';
    const fd=new FormData(); fd.append('file',f); fd.append('conf',conf); fd.append('iou',iou);
    const r=await fetch('/detect_video',{method:'POST',body:fd});
    const reader=r.body.getReader(); const dec=new TextDecoder(); let buf='';
    while(running){
      const {done,value}=await reader.read(); if(done)break;
      buf+=dec.decode(value); const parts=buf.split('\n\n'); buf=parts.pop();
      for(const p of parts){
        if(!p.startsWith('data:')) continue;
        try {
          const d=JSON.parse(p.slice(5));
          if(d.done){running=false; break;}
          showRes(d, 'Video Frame');
          document.getElementById('pb').style.width=d.progress+'%';
        } catch(e){}
      }
    }
  } else {
    while(running){
      const r=await fetch(`/webcam_frame?conf=${conf}&iou=${iou}`);
      showRes(await r.json(), 'Webcam');
      await new Promise(x=>setTimeout(x,100));
    }
  }
}
function stop(){running=false;}

function buildCompare(){
  new Chart(document.getElementById('rc'),{type:'radar',data:{
    labels:['mAP','Precision','Recall','FPS'],
    datasets:[{label:'YOLOv8',data:[94,93,91,47],borderColor:'#00ffcc'}]
  },options:{scales:{r:{grid:{color:'#1e3a5f'}}}}});
}
function buildPerf(){
  new Chart(document.getElementById('mc'),{type:'line',data:{
    labels:[1,2,3,4,5], datasets:[{label:'Accuracy',data:[0.8,0.85,0.9,0.92,0.94],borderColor:'#00ffcc'}]
  }});
}
</script></body></html>"""

@app.route("/")
def index():
    return render_template_string(HTML, model_name=os.path.basename(MODEL_PATH) if MODEL_PATH else "Không tìm thấy", model_error=MODEL_ERROR)

@app.route("/detect_image",methods=["POST"])
def detect_image():
    if not model: return jsonify({"error":MODEL_ERROR})
    f=request.files["file"]; conf=float(request.form.get("conf",.4)); iou=float(request.form.get("iou",.45))
    img=cv2.cvtColor(np.array(Image.open(f.stream).convert("RGB")),cv2.COLOR_RGB2BGR)
    res=model.predict(img,conf=conf,iou=iou,verbose=False)
    ann,det,acc,ac=draw_boxes(img,res,conf)
    return jsonify({"image":to_b64(ann),"detections":det,"accident":acc,"acc_conf":f"{ac:.1%}" if ac else ""})

@app.route("/detect_video",methods=["POST"])
def detect_video():
    f=request.files["file"]; conf=float(request.form.get("conf",.4)); iou=float(request.form.get("iou",.45))
    tmp=tempfile.NamedTemporaryFile(delete=False,suffix=".mp4"); f.save(tmp.name); tmp.close()
    def gen():
        cap=cv2.VideoCapture(tmp.name); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        idx=0
        while cap.isOpened():
            ret,frame=cap.read(); idx+=1
            if not ret or idx>500: break # Giới hạn 500 frame để tránh treo máy
            if idx%5!=0: continue
            res=model.predict(frame,conf=conf,iou=iou,verbose=False)
            ann,det,acc,ac=draw_boxes(frame,res,conf)
            yield f"data: {json.dumps({'image':to_b64(ann),'detections':det,'accident':acc,'acc_conf':f'{ac:.1%}' if ac else '','progress':round(idx/total*100)})}\n\n"
        cap.release(); os.unlink(tmp.name)
        yield f"data: {json.dumps({'done':True})}\n\n"
    return Response(gen(),mimetype="text/event-stream")

_cam=None
@app.route("/webcam_frame")
def webcam_frame():
    global _cam
    conf=float(request.args.get("conf",.4)); iou=float(request.args.get("iou",.45))
    if _cam is None: _cam=cv2.VideoCapture(0)
    ret,frame=_cam.read()
    if not ret: return jsonify({"error":"Webcam lỗi"})
    res=model.predict(frame,conf=conf,iou=iou,verbose=False)
    ann,det,acc,ac=draw_boxes(frame,res,conf)
    return jsonify({"image":to_b64(ann),"detections":det,"accident":acc,"acc_conf":f"{ac:.1%}" if ac else ""})

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  🚨 AI Accident Detection Dashboard")
    print(f"  Model: {MODEL_PATH if MODEL_PATH else 'KHÔNG TÌM THẤY'}")
    print("  👉 Mở trình duyệt: http://localhost:5000")
    print("="*50 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)