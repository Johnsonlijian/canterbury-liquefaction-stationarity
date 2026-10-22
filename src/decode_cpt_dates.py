"""Decode the MATLAB `datetime` CPT test dates stored in CANTERBURYDATASET.mat.

PRJ-2937 stores each sounding's test date as an opaque MATLAB MCOS `datetime`
object. SciPy exposes only an object reference; the property values live in the
hidden ``__function_workspace__`` subsystem. This script parses that subsystem
(MCOS ``FileWrapper__`` metadata + property cells), maps every record's object
id to its ``data`` property (milliseconds since 1970-01-01, possibly complex for
extra precision) and writes one row per sounding.

Usage:
    python -I decode_cpt_dates.py <CANTERBURYDATASET.mat> <out_csv>
"""
import io
import struct
import sys

import numpy as np
import pandas as pd
import scipy.io as sio


def load_subsystem(mat):
    from scipy.io.matlab._mio5 import MatFile5Reader

    raw = mat["__function_workspace__"].tobytes()
    header = b"MATLAB 5.0 MAT-file, synthetic header".ljust(116, b" ") + b"\x00" * 8
    stream = header + raw[0:2] + raw[2:4] + raw[8:]
    # The subsystem variable has an empty name, which loadmat keeps as raw bytes;
    # read it with the low-level reader so that it is processed as a struct.
    reader = MatFile5Reader(io.BytesIO(stream), squeeze_me=False, struct_as_record=False)
    reader.mat_stream.seek(0)
    reader.initialize_read()
    reader.read_file_header()
    hdr, _next = reader.read_var_header()
    ws = reader.read_var_array(hdr, process=True)
    mcos = ws[0, 0].MCOS
    # mcos is a MatlabOpaque with fields s0,s1,s2,arr ; arr holds the cell array
    cells = mcos[0]["arr"]
    if isinstance(cells, np.ndarray) and cells.dtype == object and cells.size == 1 and isinstance(cells.ravel()[0], np.ndarray):
        inner = cells.ravel()[0]
        if inner.dtype == object:
            cells = inner
    return cells.ravel()


def parse_metadata(blob):
    b = blob.tobytes() if isinstance(blob, np.ndarray) else bytes(blob)
    version, nstr = struct.unpack_from("<2i", b, 0)
    offsets = struct.unpack_from("<8i", b, 8)
    # names: null-terminated strings starting at byte 40, nstr of them
    pos = 40
    names = []
    for _ in range(nstr):
        end = b.index(b"\x00", pos)
        names.append(b[pos:end].decode("latin1"))
        pos = end + 1
    # segment 1: class info, 4 int32 each
    classes = []
    s, e = offsets[0], offsets[1]
    for p in range(s, e, 16):
        classes.append(struct.unpack_from("<4i", b, p))
    # segment 2: type1 object properties (may be empty)
    seg2 = (offsets[1], offsets[2])
    # segment 3: object info, 6 int32 each
    objects = []
    s, e = offsets[2], offsets[3]
    for p in range(s, e, 24):
        objects.append(struct.unpack_from("<6i", b, p))
    # segment 4: type2 object properties
    seg4 = (offsets[3], offsets[4])
    return version, names, offsets, classes, seg2, objects, seg4, b


def parse_prop_segment(b, start, end):
    """Each entry: n_props, then n_props*(name_idx, flag, value); padded to 8 bytes."""
    entries = []
    p = start
    # first block is a dummy all-zero entry of 8 bytes
    p += 8
    while p < end:
        (n,) = struct.unpack_from("<i", b, p)
        p += 4
        props = []
        for _ in range(n):
            props.append(struct.unpack_from("<3i", b, p))
            p += 12
        # pad to 8-byte boundary
        if (p - start) % 8:
            p += 8 - ((p - start) % 8)
        entries.append(props)
    return entries


def main(mat_path, out_csv):
    mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    recs = mat["CANTERBURYDATASET"]
    cells = load_subsystem(mat)
    print("MCOS cells:", len(cells))
    version, names, offsets, classes, seg2, objects, seg4, b = parse_metadata(cells[0])
    print("version", version, "n names", len(names), "offsets", offsets)
    print("names (first 30):", names[:30])
    print("classes:", classes[:6])
    print("n objects:", len(objects), "first objects:", objects[:4])
    t1 = parse_prop_segment(b, *seg2) if seg2[1] > seg2[0] else []
    t2 = parse_prop_segment(b, *seg4) if seg4[1] > seg4[0] else []
    print("type1 entries:", len(t1), "type2 entries:", len(t2))
    print("type2 first entries:", t2[:3])
    print("type1 first entries:", t1[:3])

    def name_of(i):
        # name indices in property blocks are 1-based into names
        return names[i - 1] if 1 <= i <= len(names) else f"#{i}"

    # Build object -> props mapping (object ids are 1-based; objects[0] is dummy)
    rows = []
    for obj_id in range(1, len(objects)):
        cls, _a, _b, t1_id, t2_id, dep = objects[obj_id]
        props = []
        if t2_id > 0 and t2_id - 1 < len(t2):
            props = t2[t2_id - 1]
        elif t1_id > 0 and t1_id - 1 < len(t1):
            props = t1[t1_id - 1]
        rec = {"object_id": obj_id, "class_index": cls}
        for name_idx, flag, val in props:
            pname = name_of(name_idx)
            if flag == 1:
                # value is an index into the cell array (offset by 2: cell0 metadata, cell1 empty)
                cell = cells[val + 2]
                rec[pname] = cell
            elif flag == 0:
                rec[pname] = name_of(val)
            else:
                rec[pname] = val
        rows.append(rec)
    print("decoded objects:", len(rows))
    if rows:
        print("example object props:", {k: (v if not isinstance(v, np.ndarray) else (v.dtype, v.shape, v.ravel()[:2])) for k, v in rows[0].items()})

    # Map each record's Date reference (5th element of arr = object id) to data
    out = []
    for i, r in enumerate(recs):
        ref = r.Date
        arr = ref["arr"] if hasattr(ref, "dtype") and ref.dtype.names else None
        if arr is None:
            obj_id = None
        else:
            a = np.asarray(arr.ravel()[0]).ravel()
            obj_id = int(a[4])
        ms = np.nan
        if obj_id is not None and 1 <= obj_id <= len(rows):
            dval = rows[obj_id - 1].get("data")
            if isinstance(dval, np.ndarray) and dval.size >= 1:
                v = dval.ravel()[0]
                ms = float(np.real(v)) + (float(np.imag(v)) if np.iscomplexobj(v) else 0.0)
        out.append({
            "rec_index": i,
            "ID": r.ID,
            "CPTname": r.CPTname,
            "object_id": obj_id,
            "date_ms_since_1970": ms,
        })
    df = pd.DataFrame(out)
    df["date"] = pd.to_datetime(df["date_ms_since_1970"], unit="ms", errors="coerce")
    df.to_csv(out_csv, index=False)
    print(df["date"].describe())
    print("missing:", df["date"].isna().sum())
    print(df["date"].dt.year.value_counts().sort_index())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
