"""Build a BibTeX file from DOI content negotiation (registry-sourced entries only),
plus a small set of manually curated grey-literature entries that have no DOI.

Usage: python -X utf8 build_bib.py <out_bib>
"""
import re
import sys
import time
import urllib.request

DOIS = {
    "geyin2021": "10.1177/8755293021996367",
    "prj2937": "10.17603/ds2-tygh-ht91",
    "boulanger2016": "10.1061/(ASCE)GT.1943-5606.0001388",
    "vanballegooy2014": "10.1193/031813EQS070M",
    "maurer2015lpiish": "10.1016/j.sandf.2015.06.010",
    "zhang2002": "10.1139/t02-047",
    "robertson2009": "10.1139/T09-065",
    "bradley2014": "10.1016/j.soildyn.2014.01.025",
    "maurer2014": "10.1061/(ASCE)GT.1943-5606.0001117",
    "green2014": "10.1193/030713EQS066M",
    "quigley2013": "10.1130/G33944.1",
    "cubrinovski2019": "10.1016/j.soildyn.2018.05.013",
    "beyzaei2018": "10.1016/j.soildyn.2018.01.035",
    "alatik2010": "10.1785/gssrl.81.5.794",
    "lin2011": "10.1785/0120090312",
    "anderson1999": "10.1785/gssrl.70.1.19",
    "lavrentiadis2023": "10.1007/s10518-022-01485-x",
    "maurer2023": "10.1177/87552930231173711",
    "geyin2020": "10.1061/(ASCE)GT.1943-5606.0002416",
    "hutabarat2022": "10.1061/(ASCE)GT.1943-5606.0002744",
    "wooldridge2005": "10.1002/jae.770",
    "baker2008": "10.1061/(ASCE)1090-0241(2008)134:1(14)",
    "zhu2017": "10.1785/0120160198",
    "roberts2017": "10.1111/ecog.02881",
    "selfliang1987": "10.1080/01621459.1987.10478472",
    "iwasaki1984": "10.1016/0261-7277(84)90027-5",
    "townsend2016": "10.1080/00288306.2016.1182929",
    "kayen2013": "10.1061/(ASCE)GT.1943-5606.0000743",
    "cetin2018": "10.1016/j.soildyn.2018.09.012",
    "bradley2012": "10.1080/00288306.2012.674049",
    "geyin2022": "10.1016/j.soildyn.2022.107348",
    "delong1988": "10.2307/2531595",
    "brandenberg2020": "10.1177/8755293020902477",
    "yang2026": "10.1016/j.soildyn.2026.110425",
    "robertsonwride1998": "10.1139/t98-017",
    "mitchellsolymar1984": "10.1061/(ASCE)0733-9410(1984)110:11(1559)",
    "schmertmann1991": "10.1061/(ASCE)0733-9410(1991)117:9(1288)",
    "baxtermitchell2004": "10.1061/(ASCE)1090-0241(2004)130:10(1051)",
    "wang2019cpt": "10.1061/(ASCE)GT.1943-5606.0002049",
    "lees2015": "10.1016/j.soildyn.2015.02.004",
    "maurer2019ic": "10.1016/j.soildyn.2018.04.059",
    "heckmansinger1984": "10.2307/1911491",
    "honorekyriazidou2000": "10.1111/1468-0262.00139",
}

MANUAL = r"""
@techreport{boulanger2014report,
  author = {Boulanger, R. W. and Idriss, I. M.},
  title = {{CPT} and {SPT} based liquefaction triggering procedures},
  institution = {Center for Geotechnical Modeling, University of California, Davis},
  number = {UCD/CGM-14/01},
  year = {2014},
  note = {Manual entry: verify against the UC Davis report PDF before submission}
}
@techreport{tt2015ilv,
  author = {{Tonkin \& Taylor Ltd}},
  title = {Canterbury Earthquake Sequence: Increased Liquefaction Vulnerability Assessment Methodology. Appendix E: Comparison of adjacent {CPT} traces before, during and after the {CES}},
  institution = {Report prepared for the Earthquake Commission (now Natural Hazards Commission), Wellington},
  year = {2015},
  url = {https://naturalhazards.govt.nz/assets/Canterbury-earthquake-page-documents/ILV/Appendix-E.pdf},
  note = {Manual entry: confirm report number and date from the main ILV report}
}
@inproceedings{lees2013,
  author = {Lees, J. and Ballagh, R. and Orense, R. P.},
  title = {Evaluation of liquefaction and re-liquefaction observed in {Christchurch} using {CPT} data},
  booktitle = {Proceedings of the 19th New Zealand Geotechnical Society Geotechnical Symposium},
  editor = {Chin, C. Y.},
  address = {Queenstown, New Zealand},
  year = {2013},
  note = {Manual entry from the local PDF; confirm pagination}
}
@misc{begg2015,
  author = {Begg, J. G. and Jones, K. E. and Barrell, D. J. A.},
  title = {Geology and geomorphology of urban {Christchurch} and eastern {Canterbury}},
  howpublished = {GNS Science Geological Map 3, 1:50 000, GNS Science, Lower Hutt, New Zealand},
  year = {2015},
  note = {Accessed via the GNS ArcGIS feature service NZL\_GNS\_GM3\_geomorphology\_FeatureService, October 2026}
}
@techreport{idriss1999,
  author = {Idriss, I. M.},
  title = {An update to the {Seed--Idriss} simplified procedure for evaluating liquefaction potential},
  institution = {Federal Highway Administration},
  type = {Proceedings, TRB Workshop on New Approaches to Liquefaction},
  number = {FHWA-RD-99-165},
  year = {1999},
  note = {Manual entry: verify publication details}
}
@book{idrissboulanger2008,
  author = {Idriss, I. M. and Boulanger, R. W.},
  title = {Soil Liquefaction during Earthquakes},
  series = {Monograph MNO-12},
  publisher = {Earthquake Engineering Research Institute},
  address = {Oakland, California},
  year = {2008}
}
@inproceedings{robertsoncabal2010,
  author = {Robertson, P. K. and Cabal, K. L.},
  title = {Estimating soil unit weight from {CPT}},
  booktitle = {Proceedings of the 2nd International Symposium on Cone Penetration Testing},
  address = {Huntington Beach, California},
  year = {2010},
  note = {Manual entry: verify paper number}
}
"""


def fetch(doi):
    req = urllib.request.Request("https://doi.org/" + doi, headers={"Accept": "application/x-bibtex; charset=utf-8",
                                                                   "User-Agent": "IMUT-bib/1.0 (mailto:renlijian@imut.edu.cn)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8")


def main(out_bib):
    out = []
    for key, doi in DOIS.items():
        for attempt in range(3):
            try:
                b = fetch(doi).strip()
                break
            except Exception as e:  # noqa: BLE001
                b = None
                time.sleep(2)
        if not b:
            print("FAILED", key, doi)
            continue
        b = re.sub(r"^@(\w+)\{[^,]*,", lambda m: "@" + m.group(1) + "{" + key + ",", b, count=1)
        out.append(b)
        print("ok", key)
        time.sleep(0.3)
    with open(out_bib, "w", encoding="utf-8") as f:
        f.write("% Registry-sourced entries (DOI content negotiation), generated by build_bib.py\n")
        f.write("\n\n".join(out))
        f.write("\n\n% Manually curated grey literature (no DOI): verify before submission\n")
        f.write(MANUAL)


if __name__ == "__main__":
    main(sys.argv[1])
