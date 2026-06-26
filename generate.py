"""
Iran Spatial Polygon Generator
Generates millions of non-overlapping polygons covering Iran
Output: Shapefiles in Web Mercator (EPSG:3857)

Modes:
  grid    - Regular grid polygons (fast, uniform)
  voronoi - Voronoi tessellation (realistic, irregular shapes)

Usage:
  python generate.py --mode grid    --count 3000000 --chunk 500000
  python generate.py --mode voronoi --count 500000  --chunk 100000
"""

import argparse
import math
import time
import os
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from pyproj import Transformer

# ── Iran bounding box (WGS84) ─────────────────────────────────────────────────
IRAN_MINLON, IRAN_MAXLON = 44.0, 63.5
IRAN_MINLAT, IRAN_MAXLAT = 25.0, 39.8

# ── Coordinate transformer WGS84 → Web Mercator ───────────────────────────────
to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

def iran_bbox_mercator():
    minx, miny = to_mercator.transform(IRAN_MINLON, IRAN_MINLAT)
    maxx, maxy = to_mercator.transform(IRAN_MAXLON, IRAN_MAXLAT)
    return minx, miny, maxx, maxy

def save_chunk(polygons, ids, chunk_idx, output_dir, mode):
    gdf = gpd.GeoDataFrame(
        {"id": ids, "area_m2": [round(p.area, 2) for p in polygons]},
        geometry=polygons,
        crs="EPSG:3857"
    )
    path = os.path.join(output_dir, f"polygons_{mode}_part{chunk_idx:03d}.shp")
    gdf.to_file(path)
    return path

# ── GRID MODE ─────────────────────────────────────────────────────────────────
def generate_grid(total, chunk_size, output_dir):
    minx, miny, maxx, maxy = iran_bbox_mercator()
    width  = maxx - minx
    height = maxy - miny

    cols = math.ceil(math.sqrt(total * width / height))
    rows = math.ceil(total / cols)
    actual = cols * rows

    cell_w = width  / cols
    cell_h = height / rows

    print(f"Grid: {cols} cols × {rows} rows = {actual:,} polygons")
    print(f"Cell size: {cell_w/1000:.2f} km × {cell_h/1000:.2f} km")

    polygons, ids = [], []
    chunk_idx = 1
    saved = 0
    t0 = time.time()

    for r in range(rows):
        for c in range(cols):
            x0 = minx + c * cell_w
            y0 = miny + r * cell_h
            x1 = x0 + cell_w
            y1 = y0 + cell_h
            polygons.append(box(x0, y0, x1, y1))
            ids.append(saved + len(polygons))

            if len(polygons) >= chunk_size:
                path = save_chunk(polygons, ids, chunk_idx, output_dir, "grid")
                print(f"  Saved chunk {chunk_idx}: {len(polygons):,} polygons → {path}")
                polygons, ids = [], []
                chunk_idx += 1

            saved_total = saved + len(polygons) + (chunk_idx-1)*chunk_size
            if saved_total >= total:
                break
        if saved + len(polygons) + (chunk_idx-1)*chunk_size >= total:
            break

    if polygons:
        path = save_chunk(polygons, ids, chunk_idx, output_dir, "grid")
        print(f"  Saved chunk {chunk_idx}: {len(polygons):,} polygons → {path}")

    elapsed = time.time() - t0
    total_saved = (chunk_idx-1)*chunk_size + len(polygons) if polygons else chunk_idx*chunk_size
    print(f"\nDone: {total_saved:,} polygons in {elapsed:.1f}s")

# ── VORONOI MODE ──────────────────────────────────────────────────────────────
def generate_voronoi(total, chunk_size, output_dir):
    from scipy.spatial import Voronoi

    minx, miny, maxx, maxy = iran_bbox_mercator()
    iran_poly = box(minx, miny, maxx, maxy)

    print(f"Generating {total:,} random seed points...")
    t0 = time.time()

    # Random points inside bbox
    np.random.seed(42)
    xs = np.random.uniform(minx, maxx, total)
    ys = np.random.uniform(miny, maxy, total)

    # Add mirror points around edges to close boundary cells
    margin = max(maxx-minx, maxy-miny) * 0.1
    n = 100
    ex = np.concatenate([np.full(n, minx-margin), np.full(n, maxx+margin), np.random.uniform(minx,maxx,n), np.random.uniform(minx,maxx,n)])
    ey = np.concatenate([np.random.uniform(miny,maxy,n), np.random.uniform(miny,maxy,n), np.full(n, miny-margin), np.full(n, maxy+margin)])
    points = np.column_stack([np.concatenate([xs,ex]), np.concatenate([ys,ey])])

    print(f"Computing Voronoi tessellation for {len(points):,} points...")
    vor = Voronoi(points)
    print(f"Voronoi done in {time.time()-t0:.1f}s. Building polygons...")

    polygons, ids = [], []
    chunk_idx = 1
    count = 0

    for i, region_idx in enumerate(vor.point_region[:total]):
        region = vor.regions[region_idx]
        if -1 in region or len(region) == 0:
            continue
        try:
            verts = [vor.vertices[v] for v in region]
            poly = Polygon(verts)
            clipped = poly.intersection(iran_poly)
            if clipped.is_empty or clipped.area < 1:
                continue
            if clipped.geom_type == 'MultiPolygon':
                clipped = max(clipped.geoms, key=lambda g: g.area)
            polygons.append(clipped)
            ids.append(count + 1)
            count += 1

            if len(polygons) >= chunk_size:
                path = save_chunk(polygons, ids, chunk_idx, output_dir, "voronoi")
                print(f"  Saved chunk {chunk_idx}: {len(polygons):,} polygons → {path}")
                polygons, ids = [], []
                chunk_idx += 1

        except Exception:
            continue

    if polygons:
        path = save_chunk(polygons, ids, chunk_idx, output_dir, "voronoi")
        print(f"  Saved chunk {chunk_idx}: {len(polygons):,} polygons → {path}")

    elapsed = time.time() - t0
    print(f"\nDone: {count:,} polygons in {elapsed:.1f}s")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Iran Polygon Generator")
    parser.add_argument("--mode",  choices=["grid","voronoi"], default="grid")
    parser.add_argument("--count", type=int, default=100000, help="Number of polygons")
    parser.add_argument("--chunk", type=int, default=50000,  help="Polygons per shapefile")
    parser.add_argument("--out",   default="output",         help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print(f"\nMode: {args.mode} | Count: {args.count:,} | Chunk: {args.chunk:,}")
    print(f"Output: {args.out}/\n")

    if args.mode == "grid":
        generate_grid(args.count, args.chunk, args.out)
    else:
        generate_voronoi(args.count, args.chunk, args.out)

if __name__ == "__main__":
    main()
