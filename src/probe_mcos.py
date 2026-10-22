"""Probe the MATLAB MCOS subsystem (__function_workspace__) of CANTERBURYDATASET.mat.

MATLAB `datetime` objects saved in a v5/v7 MAT-file are stored as opaque MCOS
objects. Their property values live in the hidden __function_workspace__
variable, which is itself a MAT-5 stream without the 128-byte text header.
This probe re-wraps that stream with a synthetic header so that SciPy can parse
it, then prints the structure so a decoder can be written.

Usage: python -I probe_mcos.py <path-to-CANTERBURYDATASET.mat>
"""
import io
import sys

import numpy as np
import scipy.io as sio


def main(path):
    d = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    print("keys:", list(d.keys()))
    fws = d.get("__function_workspace__")
    if fws is None:
        print("no __function_workspace__")
        return
    raw = fws.tobytes()
    print("function workspace bytes:", len(raw), "head:", raw[:16])
    # Synthetic 128-byte MAT-5 header: 116 text + 8 subsys offset + version + endian
    header = b"MATLAB 5.0 MAT-file, synthetic header for MCOS probe".ljust(116, b" ")
    header += b"\x00" * 8
    stream = header + raw[:2] + raw[2:4] + raw[8:] if False else None
    # The workspace stream starts with 2-byte version, 2-byte endian, 4 bytes padding
    version, endian = raw[0:2], raw[2:4]
    print("version bytes:", version, "endian:", endian)
    stream = header + version + endian + raw[8:]
    sub = sio.loadmat(io.BytesIO(stream), squeeze_me=False, struct_as_record=False)
    print("subsystem keys:", [k for k in sub.keys()])
    for k, v in sub.items():
        if k.startswith("__"):
            continue
        print(k, type(v), getattr(v, "shape", None), getattr(v, "dtype", None))
        obj = v
        try:
            obj = v[0, 0]
        except Exception:
            pass
        if hasattr(obj, "_fieldnames"):
            print(" fields:", obj._fieldnames)
            for f in obj._fieldnames:
                fv = getattr(obj, f)
                print("  ", f, type(fv), getattr(fv, "shape", None), getattr(fv, "dtype", None))
                # MCOS payload is usually an opaque FileWrapper__ with a cell array
                try:
                    inner = fv[0]
                    print("   inner type:", type(inner), getattr(inner, "dtype", None))
                    if hasattr(inner, "dtype") and inner.dtype.names:
                        for nm in inner.dtype.names:
                            val = inner[nm]
                            print("     ", nm, type(val), getattr(val, "shape", None))
                            if nm == "arr":
                                arr = val
                                cells = arr.ravel()[0] if arr.dtype == object else arr
                                print("      arr dtype/shape", getattr(cells, "dtype", None), getattr(cells, "shape", None))
                                cl = cells.ravel() if hasattr(cells, "ravel") else cells
                                for i, c in enumerate(cl[:12]):
                                    cc = c
                                    print(f"       cell {i}: type={type(cc).__name__} dtype={getattr(cc,'dtype',None)} shape={getattr(cc,'shape',None)}")
                                    if isinstance(cc, np.ndarray) and cc.dtype != object and cc.size < 40:
                                        print("          values:", cc.ravel()[:40])
                                print("       total cells:", len(cl))
                except Exception as exc:  # noqa: BLE001
                    print("   inner probe failed:", exc)


if __name__ == "__main__":
    main(sys.argv[1])
