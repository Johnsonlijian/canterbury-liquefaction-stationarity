"""List the contents of the NZGD export zip files (no extraction, no execution).

Usage: python -I list_nzgd_zips.py <zip_dir> [<zip_dir2> ...]
"""
import sys
import zipfile
from collections import Counter
from pathlib import Path


def main(dirs):
    ext = Counter()
    for d in dirs:
        for z in sorted(Path(d).glob("*.zip")):
            with zipfile.ZipFile(z) as zf:
                names = zf.namelist()
                for n in names:
                    ext[Path(n).suffix.lower()] += 1
                print(f"{z.name}: {len(names)} entries; e.g. {names[:4]}")
    print("extensions:", dict(ext))


if __name__ == "__main__":
    main(sys.argv[1:])
