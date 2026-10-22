"""Download GNS GM3 geomorphological-unit polygons covering the PRJ-2937 soundings.

Source: GNS Science, 1:50 000 Geomorphological Map of Eastern Canterbury
(Begg, Jones & Barrell 2015, GNS Science Geological Map 3), public ArcGIS feature service
NZL_GNS_GM3_geomorphology_FeatureService, layer 1 (geomorphological units).
Copyright: GNS Science, Land Information New Zealand.

Usage: python -I fetch_gm3_units.py <soundings.parquet> <out_geojson>
"""
import json
import sys
import time
import urllib.parse
import urllib.request

import pandas as pd

URL = ("https://gis.gns.cri.nz/server/rest/services/NZL_GNS_GM3_Eastern_Canterbury_Geomorphology/"
       "NZL_GNS_GM3_geomorphology_FeatureService/FeatureServer/1/query")


def main(snd_path, out_path):
    s = pd.read_parquet(snd_path)
    pad = 0.01
    env = {"xmin": s["lon"].min() - pad, "ymin": s["lat"].min() - pad,
           "xmax": s["lon"].max() + pad, "ymax": s["lat"].max() + pad, "spatialReference": {"wkid": 4326}}
    feats = []
    offset = 0
    while True:
        params = {"where": "1=1", "geometry": json.dumps(env), "geometryType": "esriGeometryEnvelope",
                  "inSR": "4326", "spatialRel": "esriSpatialRelIntersects", "outFields": "*",
                  "outSR": "4326", "f": "geojson", "resultOffset": str(offset), "resultRecordCount": "1000"}
        url = URL + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=120) as r:
            d = json.load(r)
        batch = d.get("features", [])
        feats.extend(batch)
        print(f"offset {offset}: {len(batch)} features", flush=True)
        if len(batch) < 1000:
            break
        offset += 1000
        time.sleep(0.5)
    out = {"type": "FeatureCollection", "features": feats,
           "source": URL, "copyright": "GNS Science, Land Information New Zealand",
           "citation": "Begg, Jones & Barrell (2015) GNS Science Geological Map 3"}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print("total features:", len(feats))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
