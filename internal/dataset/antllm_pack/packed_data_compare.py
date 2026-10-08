#!/usr/bin/env python3
"""antllm pre-pack comparison utility — GPU vs NPU.

Compares two pre-pack directories' meta + shards. Outputs JSON or diff.
Usage:
    # single directory → summary
    python3 pack_compare.py /path/to/save_path

    # two directories → diff
    python3 pack_compare.py /path/to/gpu_save_path /path/to/npu_save_path
"""
import argparse, json, os, pickle, sys
from pathlib import Path

from antllm_prepacked_reader import (
    _PACK_CONFIG_FILE,
    _PACK_DIR_NAME,
    _PACK_INDEX_FILE,
    _SHARD_INDEX_FILE,
    shard_dir_name,
)

IGNORE_META_KEYS = {"save_path", "pack_save_path", "encode_save_path",
                     "tokenizer_path", "node_rank", "node_num",
                     "remove_encode_file"}


def summarize_save_path(save_path):
    sp = Path(save_path)
    r = {"save_path": str(sp)}
    mf = sp / _PACK_CONFIG_FILE
    if mf.exists():
        meta = json.load(open(mf))
        keys = ["data_path", "max_length", "use_true_multiturn", "use_compatible_mode",
                "rotary_type", "context_parallel_size", "max_pack_sample_num",
                "skip_abnormal_data", "encode_process_per_node", "pack_process_per_node",
                "encode_batch_num", "encode_target_part_size", "pack_target_part_size",
                "total_encode_data_size", "total_pack_data_size",
                "think_enable_datasets"]
        r["meta"] = {k: meta.get(k) for k in keys if k in meta}
        r["meta"]["think_enable_count"] = len(meta.get("think_enable_datasets", []))
        r["meta"]["think_enable_sorted"] = sorted(meta.get("think_enable_datasets", []))
    else:
        r["meta"] = "MISSING"

    pr = sp / _PACK_DIR_NAME
    if not pr.exists():
        r["pack_data"] = "MISSING"
        return r

    pmf = pr / _PACK_INDEX_FILE
    if pmf.exists():
        pm = json.load(open(pmf))
        r["pack_meta_dict"] = {e["relative_path"]: e["shard_num"] for e in pm}
    else:
        r["pack_meta_dict"] = {}

    ds_info = {}
    for d in sorted(os.listdir(pr)):
        dd = pr / d
        if not dd.is_dir() or d.startswith("."):
            continue
        smf = dd / "shard_meta.json"
        if not smf.exists():
            ds_info[d] = {"shard_meta": "MISSING"}
            continue
        sm = json.load(open(smf))
        sn = sm.get("shard_num", 0)
        shards = []
        for si in range(sn):
            sd = dd / shard_dir_name(si, sn)
            if not sd.exists():
                shards.append({"exists": False})
                continue
            ss = {"shard": sd.name}
            mpkl = sd / _SHARD_INDEX_FILE
            if mpkl.exists():
                pkl = pickle.load(open(mpkl, "rb"))
                ss["data_dtype"] = pkl.get("data_dtype", [])
                ss["data_size"] = pkl.get("data_size", 0)
                ss["sample_count"] = len(pkl.get("data_meta", []))
                # first sample shapes
                dm = pkl.get("data_meta", [])
                if dm and isinstance(dm[0], dict) and "sample_meta" in dm[0]:
                    shapes = []
                    sm = dm[0]["sample_meta"]
                    if isinstance(sm, dict):
                        for dt_key, fields in sm.items():
                            if isinstance(fields, dict):
                                shapes.append({"key": dt_key, "shape": fields.get("shape"), "dtype": fields.get("dtype")})
                    elif isinstance(sm, list):
                        for item in sm:
                            if isinstance(item, dict):
                                for dt_key, fields in item.items():
                                    if isinstance(fields, dict):
                                        shapes.append({"key": dt_key, "shape": fields.get("shape"), "dtype": fields.get("dtype")})
                    ss["first_sample_shapes"] = shapes
            npy_files = sorted(sd.glob("data.*.npy"))
            ss["np_d_files"] = {str(f.name): f.stat().st_size for f in npy_files}
            shards.append(ss)
        ds_info[d] = {"shard_num": sn, "shards": shards,
                      "total_npy_mb": round(sum(sz for s in shards if "np_d_files" in s for sz in s["np_d_files"].values()) / 1048576, 2)}
    r["datasets"] = ds_info
    return r


def compare(gpu_path, npu_path):
    g = summarize_save_path(gpu_path)
    n = summarize_save_path(npu_path)
    diffs = []
    gm = g.get("meta", {})
    nm = n.get("meta", {})
    if isinstance(gm, dict) and isinstance(nm, dict):
        for k in sorted(set(list(gm.keys()) + list(nm.keys()))):
            if k in IGNORE_META_KEYS or k == "think_enable_sorted":
                continue
            if gm.get(k) != nm.get(k):
                diffs.append({"type": "meta", "key": k, "gpu": gm.get(k), "npu": nm.get(k)})
        if gm.get("think_enable_sorted") != nm.get("think_enable_sorted"):
            g_set = set(gm.get("think_enable_sorted", []))
            n_set = set(nm.get("think_enable_sorted", []))
            diffs.append({"type": "think_enable_diff",
                         "gpu_only": sorted(g_set - n_set), "npu_only": sorted(n_set - g_set),
                         "common": len(g_set & n_set), "gpu_count": len(g_set), "npu_count": len(n_set)})

    gpmd = g.get("pack_meta_dict", {})
    npmd = n.get("pack_meta_dict", {})
    g_only = sorted(set(gpmd.keys()) - set(npmd.keys()))
    n_only = sorted(set(npmd.keys()) - set(gpmd.keys()))
    if g_only:
        diffs.append({"type": "ds_gpu_only", "count": len(g_only), "datasets": g_only})
    if n_only:
        diffs.append({"type": "ds_npu_only", "count": len(n_only), "datasets": n_only})
    for ds in sorted(set(gpmd.keys()) & set(npmd.keys())):
        if gpmd[ds] != npmd[ds]:
            diffs.append({"type": "shard_num", "dataset": ds, "gpu": gpmd[ds], "npu": npmd[ds]})

    gds = g.get("datasets", {})
    nds = n.get("datasets", {})
    for ds in sorted(set(gds.keys()) & set(nds.keys())):
        gs = gds[ds].get("shards", [])
        ns = nds[ds].get("shards", [])
        if len(gs) != len(ns):
            diffs.append({"type": "shard_count", "dataset": ds, "gpu_shards": len(gs), "npu_shards": len(ns)})
            continue
        for i in range(len(gs)):
            if isinstance(gs[i], dict) and isinstance(ns[i], dict):
                if gs[i].get("data_size", 0) != ns[i].get("data_size", 0):
                    diffs.append({"type": "data_size", "dataset": ds, "shard": i,
                                 "gpu": gs[i].get("data_size"), "npu": ns[i].get("data_size")})
                if gs[i].get("sample_count", 0) != ns[i].get("sample_count", 0):
                    diffs.append({"type": "sample_count", "dataset": ds, "shard": i,
                                 "gpu": gs[i].get("sample_count"), "npu": ns[i].get("sample_count")})
                gn = gs[i].get("np_d_files", {})
                nn = ns[i].get("np_d_files", {})
                for fk in sorted(set(gn.keys()) | set(nn.keys())):
                    if gn.get(fk, 0) != nn.get(fk, 0):
                        diffs.append({"type": "npy_size", "dataset": ds, "shard": i, "file": fk,
                                     "gpu": gn.get(fk, 0), "npu": nn.get(fk, 0)})

    if not diffs:
        print("NO DIFFERENCES — GPU and NPU pre-pack are structurally identical.")
    else:
        print(f"{len(diffs)} differences:")
        for d in diffs:
            print(json.dumps(d, ensure_ascii=False, default=str))


def main():
    parser = argparse.ArgumentParser(description="antllm pre-pack comparison — GPU vs NPU")
    parser.add_argument("gpu_save_path")
    parser.add_argument("npu_save_path", nargs="?",
                       help="If omitted, outputs single-path summary JSON.")
    args = parser.parse_args()
    if args.npu_save_path:
        compare(args.gpu_save_path, args.npu_save_path)
    else:
        print(json.dumps(summarize_save_path(args.gpu_save_path), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
