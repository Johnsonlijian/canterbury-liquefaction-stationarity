"""Print the head of selected text members of NZGD export zips (read in memory, nothing executed).

Usage: python -I peek_nzgd_zip.py <zip_dir> <n_lines> [--group GRP] [<member_substring> ...]
With --group, prints <n_lines> lines starting at the "GROUP","GRP" line of the first AGS file.
"""
import sys
import zipfile
from pathlib import Path


def main(zip_dir, n_lines, args):
    group = None
    if args and args[0] == "--group":
        group, args = args[1], args[2:]
    zips = sorted(Path(zip_dir).glob("*.zip"))
    print("n zips:", len(zips))
    shown = 0
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                if args and not any(s.lower() in name.lower() for s in args):
                    continue
                if not name.lower().endswith((".ags", ".csv", ".txt")):
                    continue
                txt = zf.read(name).decode("utf-8", errors="replace").splitlines()
                start = 0
                if group:
                    hits = [i for i, l in enumerate(txt) if l.startswith(f'"GROUP","{group}"')]
                    if not hits:
                        continue
                    start = hits[0]
                print(f"===== {z.name} :: {name} ({len(txt)} lines)")
                for line in txt[start:start + n_lines]:
                    print(line[:300])
                shown += 1
                if shown >= 2:
                    return


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3:])
