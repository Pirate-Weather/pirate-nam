"""Time WMS map tiles against a running server, cold and at several concurrencies. Usage: load_tiles.py BASE_URL"""
import concurrent.futures as cf, datetime as dt, random, statistics as st, sys, time, urllib.request as u
BASE = sys.argv[1]; R = 20037508.34; CYCLE = dt.datetime(2026, 11, 5, 12)
RANGES = {"t2m": "-40,40", "mslp": "980,1040", "refc": "0,60", "pwat": "0,70"}
def one(args):
    z, _ = args; w = 2 * R / 2 ** z
    # Tiles over the contiguous US at each zoom level.
    x = random.randint(int(2 ** z * 0.19), int(2 ** z * 0.32)); y = random.randint(int(2 ** z * 0.33), int(2 ** z * 0.42)); h = random.randint(0, 84); v = random.choice(list(RANGES))
    t = (CYCLE + dt.timedelta(hours=h)).strftime("%Y-%m-%dT%H:00:00Z"); b = f"{-R + x * w},{R - (y + 1) * w},{-R + (x + 1) * w},{R - y * w}"
    url = f"{BASE}?service=WMS&version=1.3.0&request=GetMap&format=image/png&transparent=true&styles=raster/default&crs=EPSG:3857&width=256&height=256&colorscalerange={RANGES[v]}&autoscale=false&layers={v}&time={t}&bbox={b}"
    t0 = time.time()
    with u.urlopen(url, timeout=180) as r: body = r.read(); ok = r.status == 200 and body[:4] == b"\x89PNG"
    return time.time() - t0, ok
for z in (3, 5, 7):
    for c, n in ((1, 12), (4, 40), (16, 80)):
        t0 = time.time()
        with cf.ThreadPoolExecutor(c) as e: res = list(e.map(one, [(z, i) for i in range(n)]))
        d = sorted(r[0] for r in res); bad = sum(not r[1] for r in res)
        print(f"zoom {z} concurrency {c}: {n / (time.time() - t0):.1f} tiles/s, median {st.median(d):.2f} s, p95 {d[int(.95 * n)]:.2f} s, bad {bad}", flush=True)
