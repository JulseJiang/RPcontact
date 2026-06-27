#!/usr/bin/env python3
import argparse
import heapq
import json
import math
import os
import pickle
import sys
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_DIR = Path(__file__).resolve().parent / "models"
EMBEDDING_CHECKPOINT = MODEL_DIR / "rpcontact_embedding_state_dict.pt"
ONEHOT_CHECKPOINT = MODEL_DIR / "rpcontact_onehot_state_dict.pt"
DEFAULT_N_HEAD = 4
DEFAULT_SYNC = True
EMBEDDING_PRO_DIM = 5140
EMBEDDING_RNA_DIM = 772


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)


class ESM2StyleAttention(nn.Module):
    def __init__(self, d_attn, n_head, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_attn // n_head
        self.scale = math.sqrt(self.d_head)
        self.q_proj = nn.Linear(d_attn, d_attn)
        self.k_proj = nn.Linear(d_attn, d_attn)
        self.v_proj = nn.Linear(d_attn, d_attn)
        self.out_proj = nn.Linear(d_attn, d_attn)
        self.dropout = nn.Dropout(dropout)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_head, 2).float() / self.d_head))
        self.register_buffer("inv_freq", inv_freq)

    def _get_rotary_emb(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]

    def forward(self, query, key, value):
        batch_size, lq, _ = query.shape
        lk = key.shape[1]
        q = self.q_proj(query).view(batch_size, lq, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, lk, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, lk, self.n_head, self.d_head).transpose(1, 2)
        cos_q, sin_q = self._get_rotary_emb(lq, query.device)
        cos_k, sin_k = self._get_rotary_emb(lk, query.device)
        q = apply_rotary_pos_emb(q, cos_q, sin_q)
        k = apply_rotary_pos_emb(k, cos_k, sin_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        context = torch.matmul(self.dropout(F.softmax(scores, dim=-1)), v)
        context = context.transpose(1, 2).contiguous().view(batch_size, lq, -1)
        return self.out_proj(context)


class TwoTrackAttention(nn.Module):
    def __init__(self, d_attn, n_head, d_ff=512, dropout=0.1):
        super().__init__()
        self.self_attn = ESM2StyleAttention(d_attn, n_head, dropout)
        self.cross_attn = ESM2StyleAttention(d_attn, n_head, dropout)
        self.dropout_self = nn.Dropout(dropout)
        self.dropout_cross = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_attn)
        self.ff = nn.Sequential(nn.Linear(d_attn, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_attn))
        self.norm2 = nn.LayerNorm(d_attn)
        self.dropout_final = nn.Dropout(dropout)

    def forward(self, obj_update, obj_message):
        s_out = self.self_attn(obj_update, obj_update, obj_update)
        c_out = self.cross_attn(obj_update, obj_message, obj_message)
        obj_update = self.norm1(obj_update + self.dropout_self(s_out) + self.dropout_cross(c_out))
        return self.norm2(obj_update + self.dropout_final(self.ff(obj_update)))


class SymertricTwoTrackAttention(nn.Module):
    def __init__(self, d_attn, n_head, d_ff=512, dropout=0.1, sync=False):
        super().__init__()
        self.tta1 = TwoTrackAttention(d_attn, n_head, d_ff, dropout)
        self.tta2 = TwoTrackAttention(d_attn, n_head, d_ff, dropout)
        self.sync = sync

    def forward(self, obj_1, obj_2):
        if self.sync:
            return self.tta1(obj_1, obj_2), self.tta2(obj_2, obj_1)
        obj_1 = self.tta1(obj_1, obj_2)
        obj_2 = self.tta2(obj_2, obj_1)
        return obj_1, obj_2


class LinearFF(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.1):
        super().__init__()
        self.emb = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, f_in):
        return self.norm(self.dropout(self.activation(self.emb(f_in))))


class ProteinRNAInteraction_rela(nn.Module):
    def __init__(self, d_pro, d_rna, n_layers, d_attn, n_head=4, d_ff=512, dropout=0.1, sync=False):
        super().__init__()
        self.pro_emb = LinearFF(d_pro, d_attn)
        self.rna_emb = LinearFF(d_rna, d_attn)
        self.layers = nn.ModuleList(
            [SymertricTwoTrackAttention(d_attn, n_head, d_ff, dropout, sync=sync) for _ in range(n_layers)]
        )
        self.feature_fuse = nn.LayerNorm(d_attn)
        self.pred = nn.Linear(d_attn, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_pro, f_rna):
        f_pro = self.pro_emb(f_pro.permute(0, 2, 1))
        f_rna = self.rna_emb(f_rna.permute(0, 2, 1))
        for layer in self.layers:
            f_pro, f_rna = layer(f_pro, f_rna)
        interaction = self.feature_fuse(f_pro.unsqueeze(2) * f_rna.unsqueeze(1))
        return self.sigmoid(self.pred(interaction))


def normalize_embedding_array(arr, name):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def load_embedding_array(data, name):
    if name.lower().endswith((".pickle", ".pkl")):
        arr = pickle.load(BytesIO(data))
    else:
        arr = np.load(BytesIO(data), allow_pickle=False)
    return normalize_embedding_array(arr, name)


def load_embedding_file(path):
    path = Path(path)
    if path.suffix.lower() in {".pickle", ".pkl"}:
        with path.open("rb") as handle:
            arr = pickle.load(handle)
    else:
        arr = np.load(path, allow_pickle=False)
    return normalize_embedding_array(arr, path.name)


def find_embedding_pair(files_or_members):
    for pro_name, rna_name in [
        ("pro.npy", "rna.npy"),
        ("pro.pickle", "rna.pickle"),
        ("pro.pkl", "rna.pkl"),
    ]:
        if pro_name in files_or_members and rna_name in files_or_members:
            return pro_name, rna_name
    return None, None


def make_embedding_warning(mode, note):
    if mode == "embedding":
        return None
    if not note:
        return None
    prefix = "embedding fallback: "
    reason = note[len(prefix):].strip() if note.startswith(prefix) else note.strip()
    if "not found" in reason:
        return None
    return (
        "Embedding files were detected but were not used. "
        f"Reason: {reason}. RPcontact automatically used the bundled one-hot checkpoint instead."
    )


def read_fasta(text):
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        lines.append(line)
    return "".join(lines)


def sequence_to_onehot(sequence, alphabet):
    cleaned = "".join(ch for ch in sequence.upper() if ch.isalpha())
    if not cleaned:
        raise ValueError("pro.fasta and rna.fasta must contain non-empty sequences.")
    lookup = {ch: idx for idx, ch in enumerate(alphabet)}
    arr = np.zeros((len(cleaned), len(alphabet)), dtype=np.float32)
    for i, ch in enumerate(cleaned):
        if ch in lookup:
            arr[i, lookup[ch]] = 1.0
    return arr


def looks_like_onehot(block):
    if block.ndim != 2 or block.size == 0:
        return False
    if not np.all((block >= -1e-6) & (block <= 1.0 + 1e-6)):
        return False
    row_sums = block.sum(axis=1)
    return bool(np.mean(np.isclose(row_sums, 1.0, atol=1e-4) | np.isclose(row_sums, 0.0, atol=1e-4)) > 0.98)


def standardize_feature_order(features, sequence, alphabet, raw_dim, full_dim, name):
    if features.shape[1] == raw_dim:
        return np.concatenate([sequence_to_onehot(sequence, alphabet), features], axis=1).astype(np.float32)
    if features.shape[1] != full_dim:
        raise ValueError(f"{name} feature dim {features.shape[1]} does not match expected {raw_dim} or {full_dim}")

    oh_dim = len(alphabet)
    first = features[:, :oh_dim]
    last = features[:, -oh_dim:]
    if looks_like_onehot(first):
        return features.astype(np.float32, copy=False)
    if looks_like_onehot(last):
        return np.concatenate([last, features[:, :-oh_dim]], axis=1).astype(np.float32)
    expected = sequence_to_onehot(sequence, alphabet)
    if np.mean(np.isclose(first, expected, atol=1e-4)) > 0.98:
        return features.astype(np.float32, copy=False)
    if np.mean(np.isclose(last, expected, atol=1e-4)) > 0.98:
        return np.concatenate([last, features[:, :-oh_dim]], axis=1).astype(np.float32)
    return features.astype(np.float32, copy=False)


def load_zip_inputs(zip_path):
    with zipfile.ZipFile(zip_path) as archive:
        members = {Path(name).name.lower(): name for name in archive.namelist() if not name.endswith("/")}
        missing = [name for name in ["pro.fasta", "rna.fasta"] if name not in members]
        if missing:
            raise ValueError(f"Input ZIP is missing required file(s): {', '.join(missing)}")

        pro_seq = read_fasta(archive.read(members["pro.fasta"]).decode("utf-8", errors="ignore"))
        rna_seq = read_fasta(archive.read(members["rna.fasta"]).decode("utf-8", errors="ignore")).replace("T", "U")
        if not pro_seq or not rna_seq:
            raise ValueError("pro.fasta and rna.fasta must contain non-empty sequences.")

        note = ""
        pro_embed_name, rna_embed_name = find_embedding_pair(members)
        if pro_embed_name and rna_embed_name:
            try:
                protein = load_embedding_array(archive.read(members[pro_embed_name]), pro_embed_name)
                rna = load_embedding_array(archive.read(members[rna_embed_name]), rna_embed_name)
                if protein.shape[0] != len(pro_seq):
                    raise ValueError(
                        f"{pro_embed_name} length {protein.shape[0]} does not match pro.fasta length {len(pro_seq)}"
                    )
                if rna.shape[0] != len(rna_seq):
                    raise ValueError(
                        f"{rna_embed_name} length {rna.shape[0]} does not match rna.fasta length {len(rna_seq)}"
                    )
                protein = standardize_feature_order(
                    protein, pro_seq, "ACDEFGHIKLMNPQRSTVWY", 5120, EMBEDDING_PRO_DIM, pro_embed_name
                )
                rna = standardize_feature_order(rna, rna_seq, "ACGU", 768, EMBEDDING_RNA_DIM, rna_embed_name)
                return protein, rna, pro_seq, rna_seq, "embedding", f"{pro_embed_name}/{rna_embed_name} loaded"
            except Exception as exc:
                note = f"embedding fallback: {exc}"
        elif any(name in members for name in ["pro.npy", "rna.npy", "pro.pickle", "rna.pickle", "pro.pkl", "rna.pkl"]):
            note = "embedding fallback: protein and RNA embeddings must be provided together with the same format"
        else:
            note = "embedding fallback: pro/rna embedding files not found"

        protein = sequence_to_onehot(pro_seq, "ACDEFGHIKLMNPQRSTVWY")
        rna = sequence_to_onehot(rna_seq, "ACGU")
        return protein, rna, pro_seq, rna_seq, "one-hot", note


def load_directory_inputs(input_dir):
    input_dir = Path(input_dir)
    files = {path.name.lower(): path for path in input_dir.iterdir() if path.is_file()}
    missing = [name for name in ["pro.fasta", "rna.fasta"] if name not in files]
    if missing:
        raise ValueError(f"Input directory is missing required file(s): {', '.join(missing)}")

    pro_seq = read_fasta(files["pro.fasta"].read_text(errors="ignore"))
    rna_seq = read_fasta(files["rna.fasta"].read_text(errors="ignore")).replace("T", "U")
    if not pro_seq or not rna_seq:
        raise ValueError("pro.fasta and rna.fasta must contain non-empty sequences.")

    note = ""
    pro_embed_name, rna_embed_name = find_embedding_pair(files)
    if pro_embed_name and rna_embed_name:
        try:
            protein = load_embedding_file(files[pro_embed_name])
            rna = load_embedding_file(files[rna_embed_name])
            if protein.shape[0] != len(pro_seq):
                raise ValueError(
                    f"{pro_embed_name} length {protein.shape[0]} does not match pro.fasta length {len(pro_seq)}"
                )
            if rna.shape[0] != len(rna_seq):
                raise ValueError(
                    f"{rna_embed_name} length {rna.shape[0]} does not match rna.fasta length {len(rna_seq)}"
                )
            protein = standardize_feature_order(
                protein, pro_seq, "ACDEFGHIKLMNPQRSTVWY", 5120, EMBEDDING_PRO_DIM, pro_embed_name
            )
            rna = standardize_feature_order(rna, rna_seq, "ACGU", 768, EMBEDDING_RNA_DIM, rna_embed_name)
            return protein, rna, pro_seq, rna_seq, "embedding", f"{pro_embed_name}/{rna_embed_name} loaded"
        except Exception as exc:
            note = f"embedding fallback: {exc}"
    elif any(name in files for name in ["pro.npy", "rna.npy", "pro.pickle", "rna.pickle", "pro.pkl", "rna.pkl"]):
        note = "embedding fallback: protein and RNA embeddings must be provided together with the same format"
    else:
        note = "embedding fallback: pro/rna embedding files not found"

    protein = sequence_to_onehot(pro_seq, "ACDEFGHIKLMNPQRSTVWY")
    rna = sequence_to_onehot(rna_seq, "ACGU")
    return protein, rna, pro_seq, rna_seq, "one-hot", note


def load_inputs(input_path):
    input_path = Path(input_path)
    if input_path.is_dir():
        return load_directory_inputs(input_path)
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        return load_zip_inputs(input_path)
    raise ValueError(
        "Input must be a directory or a .zip file containing pro.fasta, rna.fasta, and optional pro/rna embeddings."
    )


def infer_checkpoint_config(state, fallback_d_pro, fallback_d_rna):
    d_attn = int(state.get("pro_emb.emb.weight", np.empty((64, fallback_d_pro))).shape[0])
    d_pro = int(state.get("pro_emb.emb.weight", np.empty((d_attn, fallback_d_pro))).shape[1])
    d_rna = int(state.get("rna_emb.emb.weight", np.empty((d_attn, fallback_d_rna))).shape[1])
    layer_ids = set()
    for key in state:
        if key.startswith("layers."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                layer_ids.add(int(parts[1]))
    n_layers = max(layer_ids) + 1 if layer_ids else 3
    return d_pro, d_rna, n_layers, d_attn, DEFAULT_N_HEAD


def checkpoint_contact(protein, rna, checkpoint_path, device="cpu"):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        obj = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        obj = torch.load(checkpoint_path, map_location=device)
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if isinstance(obj, dict) and "model_state_dict" in obj:
        obj = obj["model_state_dict"]
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint object: {type(obj)}")

    state = {k.replace("module.", "", 1): v for k, v in obj.items()}
    d_pro, d_rna, n_layers, d_attn, n_head = infer_checkpoint_config(state, protein.shape[1], rna.shape[1])
    if d_pro != protein.shape[1] or d_rna != rna.shape[1]:
        raise ValueError(
            f"Checkpoint expects protein/RNA dims {d_pro}/{d_rna}, got {protein.shape[1]}/{rna.shape[1]}"
        )

    model = ProteinRNAInteraction_rela(
        d_pro=d_pro, d_rna=d_rna, n_layers=n_layers, d_attn=d_attn, n_head=n_head, sync=DEFAULT_SYNC
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected checkpoint keys: {unexpected[:5]}")
    if len(missing) > 8:
        raise ValueError(f"Too many missing checkpoint keys; first keys: {missing[:8]}")

    model.to(device)
    model.eval()
    with torch.no_grad():
        f_pro = torch.from_numpy(protein.T[None, :, :].astype(np.float32)).to(device)
        f_rna = torch.from_numpy(rna.T[None, :, :].astype(np.float32)).to(device)
        pred = model(f_pro, f_rna)
        scores = pred.detach().cpu().numpy()
    scores = np.squeeze(scores)
    if scores.ndim == 3 and scores.shape[-1] == 1:
        scores = scores[..., 0]
    if scores.ndim != 2:
        raise ValueError(f"Checkpoint output must resolve to 2D, got {scores.shape}")
    return np.clip(scores.astype(np.float32), 0.0, 1.0)


def load_checkpoint_model(protein_dim, rna_dim, checkpoint_path, device="cpu"):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        obj = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        obj = torch.load(checkpoint_path, map_location=device)
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if isinstance(obj, dict) and "model_state_dict" in obj:
        obj = obj["model_state_dict"]
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported checkpoint object: {type(obj)}")

    state = {k.replace("module.", "", 1): v for k, v in obj.items()}
    d_pro, d_rna, n_layers, d_attn, n_head = infer_checkpoint_config(state, protein_dim, rna_dim)
    if d_pro != protein_dim or d_rna != rna_dim:
        raise ValueError(f"Checkpoint expects protein/RNA dims {d_pro}/{d_rna}, got {protein_dim}/{rna_dim}")

    model = ProteinRNAInteraction_rela(
        d_pro=d_pro, d_rna=d_rna, n_layers=n_layers, d_attn=d_attn, n_head=n_head, sync=DEFAULT_SYNC
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected checkpoint keys: {unexpected[:5]}")
    if len(missing) > 8:
        raise ValueError(f"Too many missing checkpoint keys; first keys: {missing[:8]}")
    model.to(device)
    model.eval()
    return model


def encode_tracks(model, protein, rna, device="cpu"):
    with torch.no_grad():
        f_pro = torch.from_numpy(protein.T[None, :, :].astype(np.float32)).to(device)
        f_rna = torch.from_numpy(rna.T[None, :, :].astype(np.float32)).to(device)
        f_pro = model.pro_emb(f_pro.permute(0, 2, 1))
        f_rna = model.rna_emb(f_rna.permute(0, 2, 1))
        for layer in model.layers:
            f_pro, f_rna = layer(f_pro, f_rna)
    return f_pro, f_rna


def score_block(model, f_pro, f_rna, pro_start, pro_end, rna_start, rna_end):
    with torch.no_grad():
        pro_block = f_pro[:, pro_start:pro_end, :]
        rna_block = f_rna[:, rna_start:rna_end, :]
        interaction = model.feature_fuse(pro_block.unsqueeze(2) * rna_block.unsqueeze(1))
        pred = model.sigmoid(model.pred(interaction))
        scores = pred.detach().cpu().numpy()
    scores = np.squeeze(scores)
    if scores.ndim == 3 and scores.shape[-1] == 1:
        scores = scores[..., 0]
    if scores.ndim != 2:
        raise ValueError(f"Score block must resolve to 2D, got {scores.shape}")
    return np.clip(scores.astype(np.float32), 0.0, 1.0)


def label_at(sequence, index, fallback_prefix):
    if sequence and index < len(sequence):
        return f"{sequence[index].upper()}{index + 1}"
    return f"{fallback_prefix}{index + 1}"


def save_empty_distribution_plot(title, threshold, out_path):
    plt.figure(figsize=(6.4, 3.6), dpi=180)
    plt.axis("off")
    plt.title(title)
    plt.text(
        0.5,
        0.5,
        f"No contacts pass threshold >= {float(threshold):.2f}",
        ha="center",
        va="center",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_contact_bar_plot(labels, scores, threshold, out_path):
    scores = np.asarray(scores, dtype=np.float32)[:10]
    labels = labels[: len(scores)]
    if scores.size == 0:
        save_empty_distribution_plot("Top 10 contact scores", threshold, out_path)
        return

    plt.figure(figsize=(max(6.4, len(scores) * 0.68), 3.8), dpi=180)
    positions = np.arange(len(scores))
    plt.bar(positions, scores, color="#2563eb", edgecolor="#1e3a8a", linewidth=0.8)
    plt.axhline(float(threshold), color="#f97316", linestyle="--", linewidth=1.1, label=f"threshold={float(threshold):.2f}")
    plt.xticks(positions, labels, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("contact score")
    plt.title("Top 10 contact scores")
    plt.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_violin_strip_plot(groups, labels, title, threshold, out_path):
    if not groups:
        save_empty_distribution_plot(title, threshold, out_path)
        return

    plt.figure(figsize=(max(6.4, len(groups) * 0.62), 4.2), dpi=180)
    positions = np.arange(1, len(groups) + 1)
    violin_groups = []
    for values in groups:
        values = np.asarray(values, dtype=np.float32)
        violin_groups.append(np.asarray([values[0], values[0]], dtype=np.float32) if len(values) == 1 else values)
    violin = plt.violinplot(violin_groups, positions=positions, showmeans=False, showmedians=True, showextrema=False)
    for body in violin["bodies"]:
        body.set_facecolor("none")
        body.set_edgecolor("#2563eb")
        body.set_linewidth(1.2)
        body.set_alpha(1.0)
    if "cmedians" in violin:
        violin["cmedians"].set_color("#ef4444")
        violin["cmedians"].set_linewidth(1.4)

    rng = np.random.default_rng(7)
    for pos, values in zip(positions, groups):
        values = np.asarray(values, dtype=np.float32)
        jitter = rng.uniform(-0.12, 0.12, size=len(values))
        plt.scatter(np.full(len(values), pos) + jitter, values, s=15, color="#111827", alpha=0.62, linewidths=0)

    plt.axhline(float(threshold), color="#f97316", linestyle="--", linewidth=1.1, label=f"threshold={float(threshold):.2f}")
    plt.xticks(positions, labels, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("contact score")
    plt.title(title)
    plt.legend(loc="upper right", frameon=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_distribution_plots(out_dir, threshold, rows, cols, top_scores, pro_seq="", rna_seq=""):
    pair_path = out_dir / "top10_contact_pair_scores.png"
    nucleotide_path = out_dir / "top10_nucleotide_scores.png"
    residue_path = out_dir / "top10_residue_scores.png"

    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    top_scores = np.asarray(top_scores, dtype=np.float32)
    if top_scores.size == 0:
        save_empty_distribution_plot("Top 10 contact scores", threshold, pair_path)
        save_empty_distribution_plot("Top 10 nucleotide score distributions", threshold, nucleotide_path)
        save_empty_distribution_plot("Top 10 residue score distributions", threshold, residue_path)
        return pair_path, nucleotide_path, residue_path

    pair_labels = [
        f"{label_at(rna_seq, int(col), 'N')}-{label_at(pro_seq, int(row), 'X')}"
        for row, col in zip(rows[:10], cols[:10])
    ]
    save_contact_bar_plot(pair_labels, top_scores[:10], threshold, pair_path)

    nucleotide_groups = []
    for col in np.unique(cols):
        values = top_scores[cols == col]
        nucleotide_groups.append((float(values.max()), int(col), values))
    nucleotide_groups.sort(reverse=True, key=lambda item: item[0])
    nucleotide_groups = nucleotide_groups[:10]
    save_violin_strip_plot(
        [item[2] for item in nucleotide_groups],
        [label_at(rna_seq, item[1], "N") for item in nucleotide_groups],
        "Top 10 nucleotide score distributions",
        threshold,
        nucleotide_path,
    )

    residue_groups = []
    for row in np.unique(rows):
        values = top_scores[rows == row]
        residue_groups.append((float(values.max()), int(row), values))
    residue_groups.sort(reverse=True, key=lambda item: item[0])
    residue_groups = residue_groups[:10]
    save_violin_strip_plot(
        [item[2] for item in residue_groups],
        [label_at(pro_seq, item[1], "X") for item in residue_groups],
        "Top 10 residue score distributions",
        threshold,
        residue_path,
    )

    return pair_path, nucleotide_path, residue_path


def save_heatmap_from_npy(scores_path, shape, out_path, threshold):
    scores = np.load(scores_path, mmap_mode="r")
    shown = np.ma.masked_less(scores, threshold)
    fig_w = max(5, min(14, shape[1] / 25))
    fig_h = max(5, min(14, shape[0] / 25))
    plt.figure(figsize=(fig_w, fig_h), dpi=180)
    extent = [1, shape[1], 1, shape[0]]
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#f8fafc")
    plt.imshow(shown, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0, origin="lower", extent=extent)
    plt.colorbar(label="contact score")
    plt.xlabel("RNA nucleotide index (1-based)")
    plt.ylabel("Protein residue index (1-based)")
    plt.title(f"threshold={float(threshold):.2f}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def update_top_heap(heap, block, row_offset, col_offset, threshold, top_k):
    passing = np.argwhere(block >= threshold)
    for local_row, local_col in passing:
        score = float(block[local_row, local_col])
        row = int(row_offset + local_row)
        col = int(col_offset + local_col)
        item = (score, row, col)
        if len(heap) < top_k:
            heapq.heappush(heap, item)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, item)
    return int(len(passing))


def write_top_tables(heap, out_dir, pro_seq, rna_seq):
    top_path = out_dir / "rpcontact_top_pairs.csv"
    ordered = sorted(heap, reverse=True)
    rows = np.asarray([item[1] for item in ordered], dtype=np.int64)
    cols = np.asarray([item[2] for item in ordered], dtype=np.int64)
    scores = np.asarray([item[0] for item in ordered], dtype=np.float32)
    top_df = pd.DataFrame(
        {
            "rank": np.arange(1, len(ordered) + 1),
            "Nucleotide": [label_at(rna_seq, int(col), "N") for col in cols],
            "residue": [label_at(pro_seq, int(row), "X") for row in rows],
            "score": [round(float(score), 6) for score in scores],
        }
    )
    top_df.to_csv(top_path, index=False)
    return top_path, rows, cols, scores


def write_full_map_txt(scores_path, out_path, pro_seq, rna_seq):
    scores = np.load(scores_path, mmap_mode="r")
    with Path(out_path).open("w", encoding="utf-8") as handle:
        handle.write("#full_map\n")
        handle.write(f"# row =rna:{rna_seq}\n")
        handle.write(f"# col=protein:{pro_seq}\n")
        pro_labels = [label_at(pro_seq, i, "X") for i in range(scores.shape[0])]
        handle.write("\t" + "\t".join(pro_labels) + "\n")
        for rna_idx in range(scores.shape[1]):
            row_label = label_at(rna_seq, rna_idx, "N")
            values = scores[:, rna_idx]
            handle.write(row_label + "\t" + "\t".join(f"{float(v):.5f}" for v in values) + "\n")


def write_summary(out_dir, summary):
    meta_path = out_dir / "rpcontact_summary.json"
    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return meta_path


def package_outputs(out_dir, paths):
    zip_path = out_dir / "rpcontact_results.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path is not None and Path(path).exists():
                archive.write(path, arcname=Path(path).name)
    return zip_path


def predict(
    input_zip,
    out_dir,
    threshold=0.0,
    top_k=100,
    device="cpu",
    table_limit=100,
    pro_chunk=256,
    rna_chunk=256,
    plot_heatmap=False,
    save_npy=False,
    plots=False,
):
    input_zip = Path(input_zip)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    protein, rna, pro_seq, rna_seq, mode, note = load_inputs(input_zip)
    checkpoint_path = EMBEDDING_CHECKPOINT if mode == "embedding" else ONEHOT_CHECKPOINT
    embedding_warning = make_embedding_warning(mode, note)
    if embedding_warning:
        print(f"WARNING: {embedding_warning}", file=sys.stderr)

    cells = protein.shape[0] * rna.shape[0]
    top_k = max(1, int(top_k))
    threshold = float(threshold)
    pro_chunk = max(1, int(pro_chunk))
    rna_chunk = max(1, int(rna_chunk))

    model = load_checkpoint_model(protein.shape[1], rna.shape[1], checkpoint_path, device=device)
    f_pro, f_rna = encode_tracks(model, protein, rna, device=device)

    scores_path = out_dir / "rpcontact_scores.npy"
    scores = np.lib.format.open_memmap(scores_path, mode="w+", dtype=np.float32, shape=(protein.shape[0], rna.shape[0]))
    heap = []
    contacts_at_threshold = 0
    score_sum = 0.0
    max_score = 0.0

    for pro_start in range(0, protein.shape[0], pro_chunk):
        pro_end = min(pro_start + pro_chunk, protein.shape[0])
        for rna_start in range(0, rna.shape[0], rna_chunk):
            rna_end = min(rna_start + rna_chunk, rna.shape[0])
            block = score_block(model, f_pro, f_rna, pro_start, pro_end, rna_start, rna_end)
            scores[pro_start:pro_end, rna_start:rna_end] = block
            contacts_at_threshold += update_top_heap(heap, block, pro_start, rna_start, threshold, top_k)
            score_sum += float(block.sum())
            max_score = max(max_score, float(block.max()))
    scores.flush()

    top_path, rows, cols, top_scores = write_top_tables(heap, out_dir, pro_seq, rna_seq)

    pair_plot = nucleotide_plot = residue_plot = None
    if plots:
        pair_plot, nucleotide_plot, residue_plot = save_distribution_plots(
            out_dir, threshold, rows, cols, top_scores, pro_seq, rna_seq
        )

    heatmap_path = None
    if plot_heatmap:
        heatmap_path = out_dir / "rpcontact_contact_map.png"
        save_heatmap_from_npy(scores_path, scores.shape, heatmap_path, threshold)

    summary = {
        "mode": mode,
        "model": f"bundled checkpoint: {checkpoint_path.name}",
        "input_note": note,
        "embedding_warning": embedding_warning,
        "score_shape": [int(protein.shape[0]), int(rna.shape[0])],
        "threshold": threshold,
        "top_k": int(len(heap)),
        "requested_top_k": int(top_k),
        "contacts_at_threshold": int(contacts_at_threshold),
        "max_score": float(max_score),
        "mean_score": float(score_sum / max(cells, 1)),
        "chunks": {"protein": int(pro_chunk), "rna": int(rna_chunk)},
        "outputs": {
            "scores_npy": scores_path.name,
            "full_map_txt": "rpcontact_full_map.txt",
            "top_pairs_csv": top_path.name,
            "contact_map_png": heatmap_path.name if heatmap_path else None,
            "top10_contact_pair_scores_png": Path(pair_plot).name if pair_plot else None,
            "top10_nucleotide_scores_png": Path(nucleotide_plot).name if nucleotide_plot else None,
            "top10_residue_scores_png": Path(residue_plot).name if residue_plot else None,
            "results_zip": "rpcontact_results.zip",
        },
        "index_note": "Nucleotide and residue labels use 1-based indexing.",
        "note": "Use --save-npy to keep the full NumPy score matrix and --plot-heatmap to generate a full-resolution PNG.",
    }
    full_map_path = out_dir / "rpcontact_full_map.txt"
    write_full_map_txt(scores_path, full_map_path, pro_seq, rna_seq)
    if not save_npy:
        Path(scores_path).unlink(missing_ok=True)
        summary["outputs"]["scores_npy"] = None
    meta_path = write_summary(out_dir, summary)
    zip_path = package_outputs(
        out_dir,
        [
            scores_path if save_npy else None,
            full_map_path,
            top_path,
            meta_path,
            heatmap_path,
            pair_plot,
            nucleotide_plot,
            residue_plot,
        ],
    )
    summary["outputs"]["results_zip"] = zip_path.name
    write_summary(out_dir, summary)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description="Run RPcontact contact-map prediction without Gradio, with chunked score export.")
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input directory or ZIP with pro.fasta, rna.fasta, and optional pro/rna embeddings (.npy, .pickle, or .pkl).",
    )
    parser.add_argument("-o", "--out-dir", default="rpcontact_output", help="Output directory.")
    parser.add_argument("--threshold", type=float, default=0.0, help="Contact threshold used for top-pair filtering.")
    parser.add_argument("--top-k", type=int, default=100, help="Number of top contacts to export after threshold filtering.")
    parser.add_argument("--pro-chunk", type=int, default=256, help="Protein chunk size for writing scores.")
    parser.add_argument("--rna-chunk", type=int, default=256, help="RNA chunk size for writing scores.")
    parser.add_argument("--plot-heatmap", action="store_true", help="Generate a full-resolution heatmap PNG. Disabled by default.")
    parser.add_argument("--save-npy", action="store_true", help="Keep rpcontact_scores.npy in the output directory and result ZIP.")
    parser.add_argument("--plots", action="store_true", help="Generate diagnostic top-score distribution plots.")
    parser.add_argument("--device", default="cpu", help="Torch device, e.g. cpu or cuda:0.")
    return parser


def main():
    args = build_parser().parse_args()
    summary = predict(
        input_zip=args.input,
        out_dir=args.out_dir,
        threshold=args.threshold,
        top_k=args.top_k,
        device=args.device,
        pro_chunk=args.pro_chunk,
        rna_chunk=args.rna_chunk,
        plot_heatmap=args.plot_heatmap,
        save_npy=args.save_npy,
        plots=args.plots,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
