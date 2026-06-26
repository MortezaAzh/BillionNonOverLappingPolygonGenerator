"""
Iran Polygon Generator — Web UI
Run: python app.py
Open: http://localhost:5000
"""

from flask import Flask, render_template_string, request, jsonify, send_file
import threading
import math
import time
import os
import zipfile
import numpy as np
import geopandas as gpd
from shapely.geometry import box, Polygon
from pyproj import Transformer

app = Flask(__name__)

progress = {"status": "idle", "done": 0, "total": 0, "message": "", "files": []}
progress_lock = threading.Lock()

to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

PROVINCES = {
    "Iran (All)":        (44.0, 25.0, 63.5, 39.8),
    "Tehran":            (50.8, 35.1, 52.8, 36.3),
    "Isfahan":           (49.5, 31.5, 52.8, 34.5),
    "Fars":              (50.5, 27.5, 55.0, 31.5),
    "Khorasan Razavi":   (56.5, 34.5, 61.5, 38.5),
    "Khuzestan":         (47.5, 29.5, 50.5, 33.0),
    "East Azerbaijan":   (44.5, 36.5, 48.0, 39.0),
    "West Azerbaijan":   (43.5, 36.0, 46.5, 38.5),
    "Kerman":            (54.5, 26.5, 59.5, 31.5),
    "Sistan-Baluchestan":(60.0, 25.5, 63.5, 31.5),
    "Hormozgan":         (54.5, 25.0, 59.5, 28.5),
    "Gilan":             (48.5, 36.5, 50.5, 38.0),
    "Mazandaran":        (50.5, 35.8, 56.5, 37.2),
    "Alborz":            (50.5, 35.6, 51.5, 36.2),
    "Markazi":           (49.0, 33.5, 51.5, 35.5),
}

OUTPUT_DIR = "output"

def iran_bbox(province):
    minlon, minlat, maxlon, maxlat = PROVINCES[province]
    minx, miny = to_mercator.transform(minlon, minlat)
    maxx, maxy = to_mercator.transform(maxlon, maxlat)
    return minx, miny, maxx, maxy

def save_chunk(polygons, ids, chunk_idx, mode, province_slug):
    gdf = gpd.GeoDataFrame(
        {"id": ids, "area_m2": [round(p.area, 2) for p in polygons]},
        geometry=polygons, crs="EPSG:3857"
    )
    fname = f"polygons_{province_slug}_{mode}_part{chunk_idx:03d}.shp"
    path = os.path.join(OUTPUT_DIR, fname)
    gdf.to_file(path)
    return fname

def run_grid(count, chunk_size, province, province_slug):
    minx, miny, maxx, maxy = iran_bbox(province)
    width, height = maxx - minx, maxy - miny
    cols = math.ceil(math.sqrt(count * width / height))
    rows = math.ceil(count / cols)
    cell_w, cell_h = width / cols, height / rows

    with progress_lock:
        progress["total"] = min(count, cols * rows)
        progress["message"] = f"Grid: {cols}×{rows} | Cell: {cell_w/1000:.1f}×{cell_h/1000:.1f} km"

    polygons, ids, chunk_idx, done, files = [], [], 1, 0, []

    for r in range(rows):
        for c in range(cols):
            if done >= count: break
            x0 = minx + c * cell_w
            y0 = miny + r * cell_h
            polygons.append(box(x0, y0, x0 + cell_w, y0 + cell_h))
            ids.append(done + 1)
            done += 1
            if len(polygons) >= chunk_size:
                fname = save_chunk(polygons, ids, chunk_idx, "grid", province_slug)
                files.append(fname)
                with progress_lock:
                    progress["done"] = done
                    progress["files"] = files[:]
                polygons, ids = [], []
                chunk_idx += 1
        if done >= count: break

    if polygons:
        fname = save_chunk(polygons, ids, chunk_idx, "grid", province_slug)
        files.append(fname)

    with progress_lock:
        progress["done"] = done
        progress["files"] = files[:]
        progress["status"] = "done"
        progress["message"] = f"Done! {done:,} polygons in {len(files)} shapefile(s)."

def run_voronoi(count, chunk_size, province, province_slug):
    from scipy.spatial import Voronoi
    minx, miny, maxx, maxy = iran_bbox(province)
    iran_poly = box(minx, miny, maxx, maxy)

    with progress_lock:
        progress["total"] = count
        progress["message"] = "Generating seed points..."

    np.random.seed(42)
    xs = np.random.uniform(minx, maxx, count)
    ys = np.random.uniform(miny, maxy, count)
    n = 100
    ex = np.concatenate([np.full(n,minx-1e6), np.full(n,maxx+1e6), np.random.uniform(minx,maxx,n), np.random.uniform(minx,maxx,n)])
    ey = np.concatenate([np.random.uniform(miny,maxy,n), np.random.uniform(miny,maxy,n), np.full(n,miny-1e6), np.full(n,maxy+1e6)])
    points = np.column_stack([np.concatenate([xs,ex]), np.concatenate([ys,ey])])

    with progress_lock:
        progress["message"] = "Computing Voronoi tessellation..."

    vor = Voronoi(points)
    polygons, ids, chunk_idx, done, files = [], [], 1, 0, []

    for i, region_idx in enumerate(vor.point_region[:count]):
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0: continue
        try:
            poly = Polygon([vor.vertices[v] for v in region]).intersection(iran_poly)
            if poly.is_empty or poly.area < 1: continue
            if poly.geom_type == 'MultiPolygon':
                poly = max(poly.geoms, key=lambda g: g.area)
            polygons.append(poly)
            ids.append(done + 1)
            done += 1
            if len(polygons) >= chunk_size:
                fname = save_chunk(polygons, ids, chunk_idx, "voronoi", province_slug)
                files.append(fname)
                with progress_lock:
                    progress["done"] = done
                    progress["files"] = files[:]
                    progress["message"] = f"Saved chunk {chunk_idx}..."
                polygons, ids = [], []
                chunk_idx += 1
        except: continue

    if polygons:
        fname = save_chunk(polygons, ids, chunk_idx, "voronoi", province_slug)
        files.append(fname)

    with progress_lock:
        progress["done"] = done
        progress["files"] = files[:]
        progress["status"] = "done"
        progress["message"] = f"Done! {done:,} polygons in {len(files)} shapefile(s)."

HTML = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Iran Polygon Generator</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',sans-serif;background:#0f1923;color:#e0e6ed;min-height:100vh;display:flex;align-items:center;justify-content:center;padding:2rem}
.card{background:#151f2e;border:1px solid #1e3a5f;border-radius:12px;padding:2rem;width:100%;max-width:580px}
h1{font-size:18px;font-weight:600;color:#4a9eff;margin-bottom:4px}
.sub{font-size:12px;color:#5a7a9e;margin-bottom:1.5rem}
.field{margin-bottom:1rem}
label{display:block;font-size:11px;color:#7a9ec0;margin-bottom:4px;text-transform:uppercase;letter-spacing:.05em}
input,select{width:100%;padding:8px 12px;border-radius:6px;background:#0f1923;border:1px solid #1e3a5f;color:#e0e6ed;font-size:13px}
input:focus,select:focus{outline:none;border-color:#4a9eff}
.row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.mode-cards{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:1rem}
.mode-card{border:1px solid #1e3a5f;border-radius:8px;padding:12px;cursor:pointer;transition:.15s}
.mode-card:hover{border-color:#4a9eff}
.mode-card.selected{border-color:#4a9eff;background:#0d2040}
.mode-title{font-size:13px;font-weight:600;color:#e0e6ed;margin-bottom:4px}
.mode-desc{font-size:11px;color:#5a7a9e}
.btn{width:100%;padding:10px;border-radius:7px;border:none;font-size:14px;font-weight:600;cursor:pointer;margin-top:4px;transition:.15s}
.btn-go{background:#4a9eff;color:#fff}.btn-go:hover{background:#2d85f0}
.btn-go:disabled{background:#2a4a6e;color:#5a7a9e;cursor:not-allowed}
.btn-dl{background:#2ecc71;color:#fff;margin-top:8px}.btn-dl:hover{background:#27ae60}
.progress-wrap{margin-top:1.2rem;display:none}
.progress-label{display:flex;justify-content:space-between;font-size:11px;color:#7a9ec0;margin-bottom:6px}
.progress-bar{background:#0f1923;border-radius:4px;height:8px;overflow:hidden;border:1px solid #1e3a5f}
.progress-fill{height:100%;background:#4a9eff;border-radius:4px;transition:width .3s}
.msg{font-size:12px;color:#7a9ec0;margin-top:8px;min-height:18px}
.files{margin-top:1rem;display:none}
.files-title{font-size:11px;color:#4a9eff;margin-bottom:6px;text-transform:uppercase;letter-spacing:.05em}
.file-item{font-size:11px;color:#5a8ab0;padding:3px 0;font-family:monospace}
.stat-row{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:1rem}
.stat{background:#0f1923;border-radius:6px;padding:8px 10px;text-align:center}
.stat-val{font-size:18px;font-weight:600;color:#4a9eff}
.stat-lbl{font-size:10px;color:#5a7a9e;margin-top:2px}
</style>
</head>
<body>
<div class="card">
  <h1>🗺 Iran Polygon Generator</h1>
  <p class="sub">Generate non-overlapping spatial polygons — output: Shapefile (EPSG:3857)</p>

  <div class="field">
    <label>Province / Region</label>
    <select id="province">
      ''' + ''.join(f'<option value="{p}">{p}</option>' for p in PROVINCES.keys()) + '''
    </select>
  </div>

  <div class="field">
    <label>Generation Mode</label>
    <div class="mode-cards">
      <label class="mode-card selected" id="card-grid" onclick="selectMode('grid')">
        <div class="mode-title">⬛ Grid</div>
        <div class="mode-desc">Uniform rectangular cells. Very fast. Ideal for load testing.</div>
      </label>
      <label class="mode-card" id="card-voronoi" onclick="selectMode('voronoi')">
        <div class="mode-title">🔷 Voronoi</div>
        <div class="mode-desc">Irregular shapes. Realistic, like real land parcels.</div>
      </label>
    </div>
  </div>

  <div class="row">
    <div class="field">
      <label>Total Polygons</label>
      <input type="number" id="count" value="10000" min="100" max="3000000" step="1000">
    </div>
    <div class="field">
      <label>Per File (chunk)</label>
      <input type="number" id="chunk" value="5000" min="100" max="500000" step="1000">
    </div>
  </div>

  <button class="btn btn-go" id="btn-go" onclick="startGen()">Generate Polygons</button>

  <div class="progress-wrap" id="progress-wrap">
    <div class="progress-label">
      <span id="prog-label">Processing...</span>
      <span id="prog-pct">0%</span>
    </div>
    <div class="progress-bar"><div class="progress-fill" id="prog-fill" style="width:0%"></div></div>
    <div class="msg" id="msg"></div>
  </div>

  <div class="stat-row" id="stats" style="display:none">
    <div class="stat"><div class="stat-val" id="s-total">0</div><div class="stat-lbl">Polygons</div></div>
    <div class="stat"><div class="stat-val" id="s-files">0</div><div class="stat-lbl">Files</div></div>
    <div class="stat"><div class="stat-val" id="s-time">0s</div><div class="stat-lbl">Time</div></div>
  </div>

  <div class="files" id="files-wrap">
    <div class="files-title">Output Files</div>
    <div id="files-list"></div>
    <button class="btn btn-dl" onclick="downloadZip()">⬇ Download All as ZIP</button>
  </div>
</div>

<script>
let mode = 'grid';
let polling = null;
let startTime = null;

function selectMode(m){
  mode = m;
  document.getElementById('card-grid').classList.toggle('selected', m==='grid');
  document.getElementById('card-voronoi').classList.toggle('selected', m==='voronoi');
}

function startGen(){
  const count    = parseInt(document.getElementById('count').value);
  const chunk    = parseInt(document.getElementById('chunk').value);
  const province = document.getElementById('province').value;
  if(!count||!chunk){ alert('Fill in all fields'); return; }

  document.getElementById('btn-go').disabled = true;
  document.getElementById('progress-wrap').style.display = 'block';
  document.getElementById('stats').style.display = 'none';
  document.getElementById('files-wrap').style.display = 'none';
  document.getElementById('files-list').innerHTML = '';
  document.getElementById('prog-fill').style.width = '0%';
  document.getElementById('prog-pct').textContent = '0%';
  startTime = Date.now();

  fetch('/generate', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({mode, count, chunk, province})
  });

  polling = setInterval(checkProgress, 800);
}

function checkProgress(){
  fetch('/progress').then(r=>r.json()).then(p=>{
    const pct = p.total > 0 ? Math.round(p.done / p.total * 100) : 0;
    document.getElementById('prog-fill').style.width = pct + '%';
    document.getElementById('prog-pct').textContent = pct + '%';
    document.getElementById('prog-label').textContent =
      `${p.done.toLocaleString()} / ${p.total.toLocaleString()} polygons`;
    document.getElementById('msg').textContent = p.message;

    if(p.files && p.files.length){
      document.getElementById('files-list').innerHTML =
        p.files.map(f => `<div class="file-item">📄 ${f}</div>`).join('');
      document.getElementById('files-wrap').style.display = 'block';
    }

    if(p.status === 'done'){
      clearInterval(polling);
      const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
      document.getElementById('s-total').textContent = p.done.toLocaleString();
      document.getElementById('s-files').textContent = p.files.length;
      document.getElementById('s-time').textContent  = elapsed + 's';
      document.getElementById('stats').style.display = 'grid';
      document.getElementById('btn-go').disabled = false;
      document.getElementById('prog-fill').style.width = '100%';
      document.getElementById('prog-pct').textContent = '100%';
    }
  });
}

function downloadZip(){
  window.location.href = '/download';
}
</script>
</body>
</html>'''

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/generate", methods=["POST"])
def generate():
    data     = request.json
    mode     = data.get("mode", "grid")
    count    = int(data.get("count", 10000))
    chunk    = int(data.get("chunk", 5000))
    province = data.get("province", "Iran (All)")
    province_slug = province.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-","_")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for f in os.listdir(OUTPUT_DIR):
        os.remove(os.path.join(OUTPUT_DIR, f))

    with progress_lock:
        progress.update({"status":"running","done":0,"total":count,"message":"Starting...","files":[]})

    fn = run_grid if mode == "grid" else run_voronoi
    t  = threading.Thread(target=fn, args=(count, chunk, province, province_slug))
    t.daemon = True
    t.start()

    return jsonify({"ok": True})

@app.route("/progress")
def get_progress():
    with progress_lock:
        return jsonify(dict(progress))

@app.route("/download")
def download():
    zip_path = "polygons_output.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in os.listdir(OUTPUT_DIR):
            zf.write(os.path.join(OUTPUT_DIR, f), f)
    return send_file(zip_path, as_attachment=True)

if __name__ == "__main__":
    print("Open http://localhost:5000")
    app.run(debug=False, port=5000)
