"""Replace Unicode punctuation/accents in a BibTeX file with LaTeX-safe equivalents.

Usage: python -X utf8 sanitize_bib.py <bib>
"""
import re
import sys

REP = {
    "‐": "-", "‑": "-", "‒": "-", "–": "--", "—": "---",
    "‘": "`", "’": "'", "“": "``", "”": "''", " ": " ", " ": " ",
    "é": r"{\'e}", "ö": r'{\"o}', "ü": r'{\"u}', "ç": r"{\c{c}}", "İ": r"{\.I}",
    "ş": r"{\c{s}}", "ğ": r"{\u{g}}", "á": r"{\'a}", "ı": r"{\i}", "ō": r"{\=o}",
    "ā": r"{\=a}", "Ç": r"{\c{C}}", "Ö": r'{\"O}', "č": r"{\v{c}}", "š": r"{\v{s}}",
    "ž": r"{\v{z}}",
}


def main(p):
    s = open(p, encoding="utf-8").read()
    for k, v in REP.items():
        s = s.replace(k, v)

    def protect(m):
        inner = re.sub(r"\b(CPT|SPT|LPI|LSN|AI|ASCE|NZ|GIS)\b", r"{\1}", m.group(2))
        return m.group(1) + inner + "}"

    s = re.sub(r"(\btitle\s*=\s*\{)([^{}]*)\}", protect, s)
    bad = sorted({c for c in s if ord(c) > 127})
    open(p, "w", encoding="utf-8").write(s)
    print("remaining non-ASCII characters:", [hex(ord(c)) for c in bad])


if __name__ == "__main__":
    main(sys.argv[1])
