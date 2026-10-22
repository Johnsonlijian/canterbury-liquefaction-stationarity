"""Find candidate DOIs by bibliographic query against Crossref and print the top matches, so that
new references are taken from the registry rather than from memory.

Usage: python -I find_dois.py
"""
import json
import time
import urllib.parse
import urllib.request

QUERIES = {
    "robertsonwride1998": "Robertson Wride 1998 Evaluating cyclic liquefaction potential using the cone penetration test Canadian Geotechnical Journal",
    "mitchellsolymar1984": "Mitchell Solymar 1984 Time-dependent strength gain in recently deposited sands Journal of Geotechnical Engineering",
    "schmertmann1991": "Schmertmann 1991 The mechanical aging of soils Journal of Geotechnical Engineering",
    "boulangerdejong2018": "Boulanger DeJong 2018 Inverse filtering procedure to correct cone penetration data for thin-layer and transition effects",
    "wang2019cpt": "Wang Xiao Evans Qiu Salam 2019 Time-dependent cone penetration resistance of a postliquefaction sand deposit at shallow depth",
    "maurer2019ic": "Maurer Green van Ballegooy Wotherspoon 2019 Development of region-specific soil behavior type index correlations Christchurch",
    "lees2015": "Lees Ballagh Orense van Ballegooy 2015 CPT-based analysis of liquefaction and re-liquefaction following the Canterbury earthquake sequence",
    "baxtermitchell2004": "Baxter Mitchell 2004 Experimental study on the aging of sands Journal of Geotechnical and Geoenvironmental Engineering",
}


def search(q, rows=3):
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode({"query.bibliographic": q, "rows": rows})
    req = urllib.request.Request(url, headers={"User-Agent": "IMUT-reference-check/1.0 (mailto:renlijian@imut.edu.cn)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        items = json.load(r)["message"]["items"]
    out = []
    for m in items:
        year = None
        for k in ["published-print", "published-online", "issued"]:
            if k in m and m[k].get("date-parts"):
                year = m[k]["date-parts"][0][0]
                break
        auth = m.get("author", [{}])
        out.append({"doi": m.get("DOI"), "title": (m.get("title") or [""])[0], "container": (m.get("container-title") or [""])[0],
                    "year": year, "first": (auth[0].get("family") if auth else ""), "volume": m.get("volume"), "page": m.get("page"),
                    "type": m.get("type")})
    return out


def main():
    for key, q in QUERIES.items():
        try:
            res = search(q)
        except Exception as e:  # noqa: BLE001
            print(key, "ERROR", e)
            continue
        print("==", key)
        for r in res:
            print("   ", r["doi"], "|", r["first"], r["year"], "|", str(r["title"])[:100], "|", r["container"], r["volume"], r["page"], r["type"])
        time.sleep(0.5)


if __name__ == "__main__":
    main()
