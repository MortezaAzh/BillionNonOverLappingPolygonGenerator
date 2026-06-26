![BillionPolGenerator Screenshot](screenshot.jpg)

# BillionPolGenerator

A Python-based spatial polygon generator for WebGIS load testing.
Generates millions of non-overlapping polygons covering Iran provinces.

## Features

- Select province or all of Iran
- Two modes: **Grid** (fast, uniform) and **Voronoi** (realistic shapes)
- Output: Shapefiles in **Web Mercator (EPSG:3857)**
- Auto-splits output into chunks
- Web UI with progress bar

## Usage

```bash
pip install flask geopandas shapely scipy numpy pyproj
python app.py
```

Open http://localhost:5000

## Provinces Supported

Iran (All), Tehran, Isfahan, Fars, Khorasan Razavi, Khuzestan,
East Azerbaijan, West Azerbaijan, Kerman, Sistan-Baluchestan,
Hormozgan, Gilan, Mazandaran, Alborz, Markazi

## Tech Stack

- Python, Flask, GeoPandas, Shapely, SciPy, NumPy, PyProj
