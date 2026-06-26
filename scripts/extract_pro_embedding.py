#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def fasta_ids(fasta_path):
    ids = []
    with open(fasta_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                ids.append(line[1:].split()[0])
    if not ids:
        raise ValueError(f"No FASTA records found in {fasta_path}")
    return ids


def run_esm_extract(esm_dir, fasta_path, tmp_dir, model, repr_layer, python_bin):
    extract_py = esm_dir / "scripts" / "extract.py"
    if not extract_py.exists():
        raise FileNotFoundError(f"Cannot find ESM extract script: {extract_py}")
    command = [
        python_bin,
        str(extract_py),
        model,
        str(fasta_path),
        str(tmp_dir),
        "--repr_layers",
        str(repr_layer),
        "--include",
        "mean",
        "per_tok",
    ]
    env = os.environ.copy()
    env.setdefault("MKL_THREADING_LAYER", "GNU")
    subprocess.run(command, cwd=str(esm_dir), env=env, check=True)


def convert_pt_files(record_ids, tmp_dir, save_path, repr_layer):
    save_path.mkdir(parents=True, exist_ok=True)
    written = []
    for record_id in record_ids:
        pt_path = tmp_dir / f"{record_id}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"ESM output not found: {pt_path}")
        obj = torch.load(pt_path, map_location="cpu")
        arr = obj["representations"][repr_layer]
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        out_path = save_path / f"{record_id}.npy"
        np.save(out_path, np.asarray(arr, dtype=np.float32))
        written.append(out_path)
    return written


def main():
    parser = argparse.ArgumentParser(description="Extract ESM-2 protein embeddings and save per-protein .npy files.")
    parser.add_argument("--seqs_path", required=True, help="Protein FASTA file.")
    parser.add_argument("--save_path", required=True, help="Output directory for per-protein .npy embeddings.")
    parser.add_argument("--esm_dir", required=True, help="Path to a local ESM repository containing scripts/extract.py.")
    parser.add_argument("--model", default="esm2_t48_15B_UR50D", help="ESM model name passed to scripts/extract.py.")
    parser.add_argument("--repr_layer", type=int, default=48, help="Representation layer to save.")
    parser.add_argument("--python", default=sys.executable, help="Python executable for running ESM scripts/extract.py.")
    parser.add_argument("--keep_pt", action="store_true", help="Keep intermediate ESM .pt files next to the .npy outputs.")
    args = parser.parse_args()

    fasta_path = Path(args.seqs_path).resolve()
    save_path = Path(args.save_path).resolve()
    esm_dir = Path(args.esm_dir).resolve()
    record_ids = fasta_ids(fasta_path)

    if args.keep_pt:
        tmp_dir = save_path / "_esm_pt"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        run_esm_extract(esm_dir, fasta_path, tmp_dir, args.model, args.repr_layer, args.python)
        written = convert_pt_files(record_ids, tmp_dir, save_path, args.repr_layer)
    else:
        with tempfile.TemporaryDirectory(prefix="rpcontact_esm2_") as tmp:
            tmp_dir = Path(tmp)
            run_esm_extract(esm_dir, fasta_path, tmp_dir, args.model, args.repr_layer, args.python)
            written = convert_pt_files(record_ids, tmp_dir, save_path, args.repr_layer)

    print(f"Wrote {len(written)} protein embedding file(s) to {save_path}")


if __name__ == "__main__":
    main()
