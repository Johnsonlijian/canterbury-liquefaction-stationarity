"""Verify candidate references by DOI against Crossref (and DataCite for datasets).

Writes a CSV with the registry title, container, year, first author and a status flag.
Usage: python -I verify_dois.py <out_csv>
"""
import csv
import json
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from build_bib import DOIS  # noqa: E402  (single source of truth for the DOIs)

CANDIDATES = dict(DOIS)
for _k in ["boulanger2014report", "tt2015ilv", "lees2013", "begg2015", "idriss1999", "idrissboulanger2008", "robertsoncabal2010"]:
    CANDIDATES[_k] = None


def crossref(doi):
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    req = urllib.request.Request(url, headers={"User-Agent": "IMUT-reference-check/1.0 (mailto:renlijian@imut.edu.cn)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        m = json.load(r)["message"]
    year = None
    for k in ["published-print", "published-online", "issued"]:
        if k in m and m[k].get("date-parts"):
            year = m[k]["date-parts"][0][0]
            break
    auth = m.get("author", [{}])
    first = (auth[0].get("family") or auth[0].get("name") or "") if auth else ""
    return {"title": (m.get("title") or [""])[0], "container": (m.get("container-title") or [""])[0],
            "year": year, "first_author": first, "volume": m.get("volume"), "page": m.get("page"), "registry": "crossref"}


def datacite(doi):
    url = "https://api.datacite.org/dois/" + urllib.parse.quote(doi)
    with urllib.request.urlopen(url, timeout=60) as r:
        a = json.load(r)["data"]["attributes"]
    return {"title": a["titles"][0]["title"], "container": a.get("publisher"), "year": a.get("publicationYear"),
            "first_author": (a.get("creators") or [{}])[0].get("name", ""), "volume": None, "page": None, "registry": "datacite"}


def main(out_csv):
    rows = []
    for key, doi in CANDIDATES.items():
        if not doi:
            rows.append({"key": key, "doi": "", "status": "NO_DOI_MANUAL_CHECK"})
            continue
        rec = {"key": key, "doi": doi}
        try:
            rec.update(crossref(doi))
            rec["status"] = "VERIFIED"
        except Exception as e1:  # noqa: BLE001
            try:
                rec.update(datacite(doi))
                rec["status"] = "VERIFIED"
            except Exception as e2:  # noqa: BLE001
                rec["status"] = f"NOT_FOUND ({type(e1).__name__}/{type(e2).__name__})"
        rows.append(rec)
        print(key, rec.get("status"), "|", rec.get("first_author"), rec.get("year"), "|", str(rec.get("title"))[:90], "|", rec.get("container"), flush=True)
        time.sleep(0.3)
    keys = ["key", "doi", "status", "first_author", "year", "title", "container", "volume", "page", "registry"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


if __name__ == "__main__":
    main(sys.argv[1])
