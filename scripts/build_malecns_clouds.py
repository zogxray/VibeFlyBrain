#!/usr/bin/env python3
"""Build MaleCNS soma + process clouds for genome_shell.

Somata: Janelia body-annotations somaLocation + lee-lab superclass labels.
Processes: stratified SWC sample in native maleCNS space (same coords as somata).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "malecns"
PROC = ROOT / "data" / "processed"
SWC_CACHE = RAW / "swc_sample"

META_URL = (
    "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/"
    "compiled_data/malecns_09/malecns_09_meta.feather"
)
ANN_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/"
    "flat-connectome/body-annotations-male-cns-v1.0-minconf-0.5.feather"
)
SWC_URL = (
    "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/"
    "compiled_data/malecns_09/malecns_malecns_space_swc/{rid}.swc"
)
SYN_URL = (
    "https://storage.googleapis.com/lee-lab_brain-and-nerve-cord-fly-connectome/"
    "compiled_data/malecns_09/malecns_09_synapses.parquet"
)
WEIGHTS_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/"
    "flat-connectome/connectome-weights-male-cns-v1.0-minconf-0.5.feather"
)
WEIGHTS_NAME = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

# Compact atlas edge recipe written for the growth engine.
EDGE_MIN_WEIGHT = 5  # keep edges with ≥ this many synapses
EDGE_MAX_KEEP = 500_000  # hard cap on stored recipe edges (top by weight)

# MaleCNS superclass → legend class (includes VNC)
SUPER_MAP = {
    "ol_intrinsic": "optic",
    "cb_intrinsic": "central",
    "vnc_intrinsic": "vnc",
    "visual_projection": "visual_projection",
    "visual_centrifugal": "visual_centrifugal",
    "descending_neuron": "descending",
    "ascending_neuron": "ascending",
    "vnc_motor": "motor",
    "cb_motor": "motor",
    "vnc_efferent": "motor",
    "cb_endocrine": "endocrine",
    "vnc_endocrine": "endocrine",
    "vnc_sensory": "sensory",
    "ol_sensory": "sensory",
    "cb_sensory": "sensory",
    "sensory_ascending": "ascending",
    "vnc_sensory_tbc": "sensory",
    "cb_sensory_tbc": "sensory",
}

# Prefer lee-lab meta names when richer
META_SUPER_MAP = {
    "optic_lobe_intrinsic": "optic",
    "central_brain_intrinsic": "central",
    "ventral_nerve_cord_intrinsic": "vnc",
    "sensory": "sensory",
    "visual_projection": "visual_projection",
    "visual_centrifugal": "visual_centrifugal",
    "descending": "descending",
    "ascending": "ascending",
    "motor": "motor",
    "visceral_circulatory": "endocrine",
    "sensory_ascending": "ascending",
    "sensory_descending": "descending",
    "ascending_visceral_circulatory": "endocrine",
}

CLASS_ORDER = [
    "optic",
    "central",
    "vnc",
    "sensory",
    "visual_projection",
    "visual_centrifugal",
    "descending",
    "ascending",
    "motor",
    "endocrine",
]

NEURONS_PER_CLASS = {
    "optic": 220,
    "central": 180,
    "vnc": 220,
    "sensory": 180,
    "visual_projection": 140,
    "visual_centrifugal": 100,
    "descending": 140,
    "ascending": 140,
    "motor": 160,
    "endocrine": 80,
}
POINTS_PER_NEURON = 160
TARGET_PROC = 350_000
RNG_SEED = 42
WORKERS = 12


def _download(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 1000:
        return path
    print(f"Downloading {url}\n  → {path}")
    urllib.request.urlretrieve(url, path)
    return path


def _parse_swc(text: str) -> np.ndarray:
    pts: list[list[float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        pts.append([float(parts[2]), float(parts[3]), float(parts[4])])
    if not pts:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(pts, dtype=np.float64)


def _fetch_swc(rid: int) -> np.ndarray | None:
    path = SWC_CACHE / f"{rid}.swc"
    if path.is_file() and path.stat().st_size > 32:
        return _parse_swc(path.read_text(encoding="utf-8", errors="ignore"))
    url = SWC_URL.format(rid=rid)
    raw = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=90) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            break
        except Exception:
            if attempt == 3:
                return None
            time.sleep(0.35 * (attempt + 1))
    if raw is None or not any(
        ln.strip() and not ln.startswith("#") for ln in raw.splitlines()
    ):
        return None
    path.write_text(raw, encoding="utf-8")
    return _parse_swc(raw)


def _sample_nodes(xyz: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(xyz) == 0:
        return xyz
    if len(xyz) <= n:
        return xyz
    step = max(1, len(xyz) // n)
    taken = xyz[::step][:n]
    if len(taken) < n:
        extra = rng.choice(len(xyz), size=n - len(taken), replace=False)
        taken = np.concatenate([taken, xyz[extra]], axis=0)
    return taken[:n]


def _normalize(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    center = xyz.mean(axis=0)
    span = float(np.ptp(xyz, axis=0).max())
    scale = span / 20.0 if span > 0 else 1.0
    out = ((xyz - center) / scale).astype(np.float32)
    return out, center.astype(np.float64), float(scale)


def build_somas() -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    meta_path = _download(META_URL, RAW / "malecns_09_meta.feather")
    ann_path = _download(ANN_URL, RAW / "body-annotations.feather")
    meta = pd.read_feather(meta_path)
    ann = pd.read_feather(ann_path)

    meta = meta.copy()
    meta["bodyId"] = pd.to_numeric(meta["malecns_09_id"], errors="coerce")
    meta = meta.dropna(subset=["bodyId"])
    meta["bodyId"] = meta["bodyId"].astype(np.int64)
    meta["cls_from_meta"] = meta["super_class"].map(
        lambda s: META_SUPER_MAP.get(str(s))
    )

    ann = ann.copy()
    ann["bodyId"] = pd.to_numeric(ann["bodyId"], errors="coerce")
    ann = ann.dropna(subset=["bodyId"])
    ann["bodyId"] = ann["bodyId"].astype(np.int64)
    ann["cls_from_ann"] = ann["superclass"].map(lambda s: SUPER_MAP.get(str(s)))

    # Prefer annotation somaLocation; class: meta first, else ann
    df = ann[["bodyId", "somaLocation", "cls_from_ann"]].merge(
        meta[["bodyId", "cls_from_meta"]], on="bodyId", how="inner"
    )
    df["cls_name"] = df["cls_from_meta"].fillna(df["cls_from_ann"])
    df = df[df["cls_name"].isin(CLASS_ORDER)].copy()
    df = df[df["somaLocation"].notna()].copy()

    xyz = np.stack(
        [np.asarray(v, dtype=np.float64) for v in df["somaLocation"].to_numpy()]
    )
    name_to_i = {n: i for i, n in enumerate(CLASS_ORDER)}
    cls = np.array([name_to_i[n] for n in df["cls_name"]], dtype=np.int32)

    xyz_n, center, scale = _normalize(xyz)
    counts = np.bincount(cls, minlength=len(CLASS_ORDER)).astype(np.int32)
    print(f"somata: {len(xyz_n)}  classes={CLASS_ORDER}")
    for i, n in enumerate(CLASS_ORDER):
        print(f"  {n}: {counts[i]}")

    out = PROC / "malecns_somas.npz"
    PROC.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        xyz=xyz_n,
        cls=cls,
        body_id=df["bodyId"].to_numpy(dtype=np.int64),
        class_names=np.array(CLASS_ORDER, dtype=object),
        counts=counts,
        center=center,
        scale=np.array(scale, dtype=np.float64),
        dataset=np.array("malecns"),
    )
    print(f"wrote {out}")
    return xyz_n, cls, CLASS_ORDER, counts


def build_processes_swc(
    class_names: list[str], center: np.ndarray, scale: float
) -> None:
    meta = pd.read_feather(RAW / "malecns_09_meta.feather")
    meta = meta.copy()
    meta["cls_name"] = meta["super_class"].map(lambda s: META_SUPER_MAP.get(str(s)))
    meta = meta[meta["cls_name"].isin(class_names)].copy()
    meta["rid"] = pd.to_numeric(meta["malecns_09_id"], errors="coerce")
    meta = meta.dropna(subset=["rid"])
    meta["rid"] = meta["rid"].astype(np.int64)

    rng = np.random.default_rng(RNG_SEED)
    SWC_CACHE.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[int, int]] = []
    for name in class_names:
        budget = NEURONS_PER_CLASS.get(name, 80)
        sub = meta[meta["cls_name"] == name]
        if sub.empty:
            print(f"  warn: no neurons for {name}")
            continue
        take = sub.sample(n=min(budget, len(sub)), random_state=RNG_SEED)
        cls_i = class_names.index(name)
        for rid in take["rid"].tolist():
            jobs.append((int(rid), cls_i))
        print(f"  {name}: schedule {len(take)} SWCs")

    xyz_chunks: list[np.ndarray] = []
    cls_chunks: list[np.ndarray] = []
    ok = fail = 0

    def work(item: tuple[int, int]) -> tuple[np.ndarray, int] | None:
        rid, cls_i = item
        pts = _fetch_swc(rid)
        if pts is None or len(pts) == 0:
            return None
        return _sample_nodes(pts, POINTS_PER_NEURON, rng), cls_i

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for k, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            res = fut.result()
            if res is None:
                fail += 1
            else:
                pts, cls_i = res
                xyz_chunks.append(pts)
                cls_chunks.append(np.full(len(pts), cls_i, dtype=np.int32))
                ok += 1
            if k % 80 == 0 or k == len(futs):
                print(f"  fetched {k}/{len(futs)} ok={ok} fail={fail}")

    if not xyz_chunks:
        raise RuntimeError("No MaleCNS SWC points")
    xyz = np.concatenate(xyz_chunks, axis=0)
    cls = np.concatenate(cls_chunks, axis=0)
    if len(xyz) > TARGET_PROC:
        pick = rng.choice(len(xyz), size=TARGET_PROC, replace=False)
        xyz, cls = xyz[pick], cls[pick]
    xyz = ((xyz - center) / scale).astype(np.float32)
    counts = np.bincount(cls, minlength=len(class_names)).astype(np.int32)
    out = PROC / "malecns_processes.npz"
    np.savez_compressed(
        out,
        xyz=xyz,
        cls=cls.astype(np.int32),
        class_names=np.array(class_names, dtype=object),
        counts=counts,
        kind=np.array("processes_swc"),
        dataset=np.array("malecns"),
    )
    print(f"wrote {out} points={len(xyz)}")
    for i, n in enumerate(class_names):
        print(f"  {n}: {int(counts[i])}")


def build_edges(min_weight: int = EDGE_MIN_WEIGHT, max_keep: int = EDGE_MAX_KEEP) -> Path:
    """Build compact pre→post recipe from MaleCNS connectome weights."""
    somas = np.load(PROC / "malecns_somas.npz", allow_pickle=True)
    if "body_id" not in somas:
        raise RuntimeError("malecns_somas.npz missing body_id — rebuild somas first")
    body = somas["body_id"].astype(np.int64)
    body_set = set(int(x) for x in body.tolist())
    print(f"soma bodies: {len(body_set)}")

    weights_path = _download(WEIGHTS_URL, RAW / WEIGHTS_NAME)
    print(f"loading {weights_path} …")
    df = pd.read_feather(weights_path)
    cols = {c.lower(): c for c in df.columns}
    print("weight columns:", list(df.columns))
    # Tolerate naming variants across releases.
    pre_c = next(
        cols[k]
        for k in (
            "body_pre",
            "pre",
            "pre_id",
            "bodyid_pre",
            "upstream",
            "source",
        )
        if k in cols
    )
    post_c = next(
        cols[k]
        for k in (
            "body_post",
            "post",
            "post_id",
            "bodyid_post",
            "downstream",
            "target",
        )
        if k in cols
    )
    w_c = next(
        (
            cols[k]
            for k in ("weight", "count", "n", "synapses", "roiweight", "strength")
            if k in cols
        ),
        None,
    )
    pre = pd.to_numeric(df[pre_c], errors="coerce")
    post = pd.to_numeric(df[post_c], errors="coerce")
    if w_c is None:
        weight = pd.Series(np.ones(len(df), dtype=np.float64))
    else:
        weight = pd.to_numeric(df[w_c], errors="coerce").fillna(1.0)

    ok = pre.notna() & post.notna() & weight.notna()
    pre_i = pre[ok].astype(np.int64).to_numpy()
    post_i = post[ok].astype(np.int64).to_numpy()
    w_i = weight[ok].astype(np.float64).to_numpy()
    print(f"raw edges: {len(pre_i):,}")

    # Keep only edges whose endpoints exist in our soma cloud.
    body_sorted = np.sort(body)
    pre_ok = np.isin(pre_i, body_sorted, assume_unique=False)
    post_ok = np.isin(post_i, body_sorted, assume_unique=False)
    both = pre_ok & post_ok & (pre_i != post_i) & (w_i >= float(min_weight))
    pre_i, post_i, w_i = pre_i[both], post_i[both], w_i[both]
    print(f"after soma∩ weight≥{min_weight}: {len(pre_i):,}")

    if len(pre_i) > max_keep:
        # Keep strongest edges.
        order = np.argpartition(-w_i, max_keep)[:max_keep]
        order = order[np.argsort(-w_i[order])]
        pre_i, post_i, w_i = pre_i[order], post_i[order], w_i[order]
        print(f"capped to top {max_keep:,} by weight")

    out = PROC / "malecns_edges.npz"
    np.savez_compressed(
        out,
        pre=pre_i.astype(np.int64),
        post=post_i.astype(np.int64),
        weight=w_i.astype(np.float32),
        min_weight=np.array(min_weight, dtype=np.int32),
        dataset=np.array("malecns"),
        source=np.array(WEIGHTS_NAME),
    )
    print(f"wrote {out} edges={len(pre_i):,} weight sum={float(w_i.sum()):,.0f}")
    return out


def build_processes_synapses(
    class_names: list[str], center: np.ndarray, scale: float
) -> None:
    import pyarrow.parquet as pq

    syn_path = RAW / "malecns_09_synapses.parquet"
    if not syn_path.is_file() or syn_path.stat().st_size < 1_000_000_000:
        print("Downloading MaleCNS synapses (~8GB) — may take a while…")
        _download(SYN_URL, syn_path)

    meta = pd.read_feather(RAW / "malecns_09_meta.feather")
    name_to_i = {n: i for i, n in enumerate(class_names)}
    id_map: dict[int, int] = {}
    for rid, sc in zip(meta["malecns_09_id"].to_numpy(), meta["super_class"].to_numpy()):
        mapped = META_SUPER_MAP.get(str(sc))
        if mapped in name_to_i:
            try:
                id_map[int(rid)] = name_to_i[mapped]
            except (TypeError, ValueError):
                continue

    pf = pq.ParquetFile(syn_path)
    names = [f.name for f in pf.schema_arrow]
    print("synapse columns:", names)
    rid_col = next(c for c in ("pre", "pre_pt_root_id", "pre_id") if c in names)
    x_col, y_col, z_col = next(
        t
        for t in (("x", "y", "z"), ("pre_x", "pre_y", "pre_z"))
        if all(c in names for c in t)
    )
    rng = np.random.default_rng(RNG_SEED)
    n_rows = pf.metadata.num_rows
    rate = min(1.0, TARGET_PROC * 2.5 / max(1, n_rows))
    chunks_xyz: list[np.ndarray] = []
    chunks_cls: list[np.ndarray] = []
    kept = 0
    for i in range(pf.metadata.num_row_groups):
        df = pf.read_row_group(i, columns=[rid_col, x_col, y_col, z_col]).to_pandas()
        rid_num = pd.to_numeric(df[rid_col], errors="coerce")
        mapped = rid_num.map(id_map)
        ok_m = mapped.notna().to_numpy()
        if not np.any(ok_m):
            continue
        xyz = df.loc[ok_m, [x_col, y_col, z_col]].to_numpy(dtype=np.float64)
        cls = mapped[ok_m].to_numpy(dtype=np.int32)
        finite = np.isfinite(xyz).all(axis=1)
        xyz, cls = xyz[finite], cls[finite]
        if rate < 1.0 and len(xyz):
            take = rng.random(len(xyz)) < rate
            xyz, cls = xyz[take], cls[take]
        if len(xyz) == 0:
            continue
        chunks_xyz.append(xyz)
        chunks_cls.append(cls)
        kept += len(xyz)
        if (i + 1) % 40 == 0:
            print(f"  rg {i+1}/{pf.metadata.num_row_groups} kept={kept}")
        if kept >= TARGET_PROC * 1.3:
            break
    xyz = np.concatenate(chunks_xyz, axis=0)
    cls = np.concatenate(chunks_cls, axis=0)
    if len(xyz) > TARGET_PROC:
        pick = rng.choice(len(xyz), size=TARGET_PROC, replace=False)
        xyz, cls = xyz[pick], cls[pick]
    xyz = ((xyz - center) / scale).astype(np.float32)
    counts = np.bincount(cls, minlength=len(class_names)).astype(np.int32)
    out = PROC / "malecns_processes.npz"
    np.savez_compressed(
        out,
        xyz=xyz,
        cls=cls.astype(np.int32),
        class_names=np.array(class_names, dtype=object),
        counts=counts,
        kind=np.array("processes_synapses"),
        dataset=np.array("malecns"),
    )
    print(f"wrote {out} points={len(xyz)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--from-synapses",
        action="store_true",
        help="Build process cloud from malecns_09_synapses.parquet",
    )
    ap.add_argument("--somas-only", action="store_true")
    ap.add_argument(
        "--edges",
        action="store_true",
        help="Build compact atlas edge recipe from connectome weights",
    )
    ap.add_argument("--edges-only", action="store_true")
    args = ap.parse_args()

    if args.edges_only:
        build_edges()
        return

    _, _, class_names, _ = build_somas()
    if args.edges:
        build_edges()
    if args.somas_only:
        return
    # Reload transform from saved somas
    data = np.load(PROC / "malecns_somas.npz", allow_pickle=True)
    center = data["center"].astype(np.float64)
    scale = float(data["scale"])
    if args.from_synapses:
        build_processes_synapses(list(class_names), center, scale)
    else:
        print("Building MaleCNS process cloud from SWC sample…")
        build_processes_swc(list(class_names), center, scale)


if __name__ == "__main__":
    main()
