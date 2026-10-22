"""Run the sounding-epoch robustness check of Paper A on its own and write
outputs/paperA_robustness_epoch.json (see part_epoch in paperA_robustness.py).

Usage: python -I paperA_epoch.py <outputs_dir>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paperA_robustness import part_epoch  # noqa: E402


def main(out_dir):
    out = Path(out_dir)
    res = part_epoch(out)
    (out / "paperA_robustness_epoch.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1])
