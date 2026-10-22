"""Inspect the field structure of the PRJ-2937 CANTERBURYDATASET.mat file.

Usage: python -I inspect_mat.py <path-to-CANTERBURYDATASET.mat>
"""
import sys

import numpy as np
import scipy.io as sio


def describe(name, v, indent="  "):
    if hasattr(v, "_fieldnames"):
        parts = []
        for g in v._fieldnames:
            gv = getattr(v, g)
            if isinstance(gv, np.ndarray):
                parts.append(f"{g}=ndarray{gv.shape}")
            else:
                parts.append(f"{g}={gv!r}"[:60])
        print(f"{indent}{name}: struct -> " + "; ".join(parts))
    elif isinstance(v, np.ndarray):
        flat = v.ravel()
        print(f"{indent}{name}: ndarray shape={v.shape} dtype={v.dtype} head={flat[:4]}")
    else:
        print(f"{indent}{name}: {type(v).__name__} = {v!r}"[:220])


def main(path):
    d = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = [k for k in d if not k.startswith("__")]
    print("top-level keys:", keys)
    for k in keys:
        v = d[k]
        print(f"{k}: type={type(v).__name__} shape={getattr(v, 'shape', None)} dtype={getattr(v, 'dtype', None)}")
    arr = d[keys[0]]
    recs = arr if isinstance(arr, np.ndarray) else np.array([arr])
    print("n records:", recs.size)
    first = recs.ravel()[0]
    print("record fields:", first._fieldnames)
    for f in first._fieldnames:
        describe(f, getattr(first, f))
    # Show a second and a late record to see variability of scalar/text fields
    for idx in (1, recs.size // 2, recs.size - 1):
        r = recs.ravel()[idx]
        print(f"--- record {idx}")
        for f in r._fieldnames:
            v = getattr(r, f)
            if not isinstance(v, np.ndarray) or v.size < 5:
                describe(f, v, indent="    ")


if __name__ == "__main__":
    main(sys.argv[1])
