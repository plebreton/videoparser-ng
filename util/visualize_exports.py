#!/usr/bin/env python3
"""
Visualize binary exports from the modified HEVC decoder.

Supported formats (little-endian) based on the export code discussed:

1) QP export
   Header per frame:
       int32 poc
       int32 width
       int32 height
   Payload:
       int16 matrix[height][width]

2) Motion-vector export
   Header per frame:
       int32 poc
       int32 width
       int32 height
   Payload:
       MVCellOut matrix[height][width], where each cell is:
           int16 mv_l0_x
           int16 mv_l0_y
           int16 mv_l1_x
           int16 mv_l1_y
           int8  ref_idx_l0
           int8  ref_idx_l1
           int8  pred_flag
           int8  reserved

   pred_flag values typically map as:
       0 = intra / none
       1 = L0
       2 = L1
       3 = BI

3) CTU bits export
   Header per frame:
       int32 poc
       int32 ctb_width
       int32 ctb_height
       int32 ctu_size
   Payload:
       CTUBitsCell matrix[ctb_height][ctb_width], where each cell is:
           uint32 total_bits
           uint32 motion_bits
           uint32 coeff_bits

4) CU-size export (optional, if you added it)
   Header per frame:
       int32 poc
       int32 width
       int32 height
   Payload:
       uint8 matrix[height][width]

Examples:
    python visualize_exports.py qp qp.bin --frame 0
    python visualize_exports.py mv mv.bin --frame 0 --list L0
    python visualize_exports.py mv mv.bin --frame 0 --mode magnitude
    python visualize_exports.py ctu-bits ctu_bits.bin --frame 0 --field total
    python visualize_exports.py cu-size cu_size.bin --frame 0
"""

from __future__ import annotations

import argparse
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ----------------------------
# Data containers
# ----------------------------

@dataclass
class QPFrame:
    poc: int
    width: int
    height: int
    matrix: np.ndarray  # int16 [H, W]


@dataclass
class MVFrame:
    poc: int
    width: int
    height: int
    mv_l0_x: np.ndarray
    mv_l0_y: np.ndarray
    mv_l1_x: np.ndarray
    mv_l1_y: np.ndarray
    ref_idx_l0: np.ndarray
    ref_idx_l1: np.ndarray
    pred_flag: np.ndarray


@dataclass
class CTUBitsFrame:
    poc: int
    ctb_width: int
    ctb_height: int
    ctu_size: int
    total_bits: np.ndarray
    motion_bits: np.ndarray
    coeff_bits: np.ndarray


@dataclass
class CUSizeFrame:
    poc: int
    width: int
    height: int
    matrix: np.ndarray  # uint8 [H, W]


# ----------------------------
# Binary readers
# ----------------------------

def _read_exact(f, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise EOFError("Unexpected EOF while reading binary file")
    return data


def _maybe_read_exact(f, n: int) -> bytes | None:
    data = f.read(n)
    if not data:
        return None
    if len(data) != n:
        raise EOFError("Unexpected EOF while reading binary file")
    return data


def load_qp_file(path: str | Path) -> List[QPFrame]:
    frames: List[QPFrame] = []
    with open(path, "rb") as f:
        while True:
            header = _maybe_read_exact(f, 12)
            if header is None:
                break
            poc, width, height = struct.unpack("<iii", header)
            count = width * height
            payload = _read_exact(f, count * 2)
            matrix = np.frombuffer(payload, dtype="<i2").reshape(height, width).copy()
            frames.append(QPFrame(poc, width, height, matrix))
    return frames


def load_mv_file(path: str | Path) -> List[MVFrame]:
    frames: List[MVFrame] = []
    cell_dtype = np.dtype([
        ("mv_l0_x", "<i2"),
        ("mv_l0_y", "<i2"),
        ("mv_l1_x", "<i2"),
        ("mv_l1_y", "<i2"),
        ("ref_idx_l0", "i1"),
        ("ref_idx_l1", "i1"),
        ("pred_flag", "i1"),
        ("reserved", "i1"),
    ])

    with open(path, "rb") as f:
        while True:
            header = _maybe_read_exact(f, 12)
            if header is None:
                break
            poc, width, height = struct.unpack("<iii", header)
            count = width * height
            payload = _read_exact(f, count * cell_dtype.itemsize)
            arr = np.frombuffer(payload, dtype=cell_dtype).reshape(height, width).copy()
            frames.append(
                MVFrame(
                    poc=poc,
                    width=width,
                    height=height,
                    mv_l0_x=arr["mv_l0_x"],
                    mv_l0_y=arr["mv_l0_y"],
                    mv_l1_x=arr["mv_l1_x"],
                    mv_l1_y=arr["mv_l1_y"],
                    ref_idx_l0=arr["ref_idx_l0"],
                    ref_idx_l1=arr["ref_idx_l1"],
                    pred_flag=arr["pred_flag"],
                )
            )
    return frames


def load_ctu_bits_file(path: str | Path) -> List[CTUBitsFrame]:
    frames: List[CTUBitsFrame] = []
    cell_dtype = np.dtype([
        ("total_bits", "<u4"),
        ("motion_bits", "<u4"),
        ("coeff_bits", "<u4"),
    ])

    with open(path, "rb") as f:
        while True:
            header = _maybe_read_exact(f, 16)
            if header is None:
                break
            poc, ctb_width, ctb_height, ctu_size = struct.unpack("<iiii", header)
            count = ctb_width * ctb_height
            payload = _read_exact(f, count * cell_dtype.itemsize)
            arr = np.frombuffer(payload, dtype=cell_dtype).reshape(ctb_height, ctb_width).copy()
            frames.append(
                CTUBitsFrame(
                    poc=poc,
                    ctb_width=ctb_width,
                    ctb_height=ctb_height,
                    ctu_size=ctu_size,
                    total_bits=arr["total_bits"],
                    motion_bits=arr["motion_bits"],
                    coeff_bits=arr["coeff_bits"],
                )
            )
    return frames


def load_cu_size_file(path: str | Path) -> List[CUSizeFrame]:
    frames: List[CUSizeFrame] = []
    with open(path, "rb") as f:
        while True:
            header = _maybe_read_exact(f, 12)
            if header is None:
                break
            poc, width, height = struct.unpack("<iii", header)
            count = width * height
            payload = _read_exact(f, count)
            matrix = np.frombuffer(payload, dtype=np.uint8).reshape(height, width).copy()
            frames.append(CUSizeFrame(poc, width, height, matrix))
    return frames


# ----------------------------
# Plot helpers
# ----------------------------

def choose_frame(frames, frame_index: int):
    if not frames:
        raise ValueError("No frames found in file")
    if frame_index < 0 or frame_index >= len(frames):
        raise IndexError(f"Frame index {frame_index} out of range [0, {len(frames)-1}]")
    return frames[frame_index]


def list_frames(frames, max_rows: int = 20):
    print(f"Frames found: {len(frames)}")
    print("idx\tpoc\tshape")
    for i, fr in enumerate(frames[:max_rows]):
        if isinstance(fr, QPFrame):
            shape = f"{fr.height}x{fr.width}"
        elif isinstance(fr, MVFrame):
            shape = f"{fr.height}x{fr.width}"
        elif isinstance(fr, CTUBitsFrame):
            shape = f"{fr.ctb_height}x{fr.ctb_width} (ctu_size={fr.ctu_size})"
        elif isinstance(fr, CUSizeFrame):
            shape = f"{fr.height}x{fr.width}"
        else:
            shape = "?"
        print(f"{i}\t{fr.poc}\t{shape}")
    if len(frames) > max_rows:
        print(f"... ({len(frames)-max_rows} more)")


def visualize_qp(frame: QPFrame, out: str | None = None, title_prefix: str = "QP"):
    mat = frame.matrix.astype(float)
    masked = np.ma.masked_where(mat < 0, mat)

    plt.figure(figsize=(8, 6))
    im = plt.imshow(masked, interpolation="nearest", aspect="auto")
    plt.colorbar(im, label="QP")
    plt.title(f"{title_prefix} | frame idx unknown | POC={frame.poc} | shape={frame.height}x{frame.width}")
    plt.xlabel("x")
    plt.ylabel("y")
    if out:
        plt.savefig(out, dpi=160, bbox_inches="tight")
    else:
        plt.show()


def _sample_grid(h: int, w: int, stride: int):
    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    X, Y = np.meshgrid(xs, ys)
    return X, Y


def visualize_mv(frame: MVFrame, list_name: str = "L0", mode: str = "quiver",
                 stride: int = 4, out: str | None = None):
    if list_name.upper() not in {"L0", "L1"}:
        raise ValueError("--list must be L0 or L1")

    if list_name.upper() == "L0":
        u = frame.mv_l0_x.astype(float)
        v = frame.mv_l0_y.astype(float)
        valid = (frame.pred_flag & 1) != 0
    else:
        u = frame.mv_l1_x.astype(float)
        v = frame.mv_l1_y.astype(float)
        valid = (frame.pred_flag & 2) != 0

    sentinel = -32768
    valid &= (u != sentinel) & (v != sentinel)

    mag = np.sqrt(np.where(valid, u, 0.0) ** 2 + np.where(valid, v, 0.0) ** 2)

    if mode == "magnitude":
        masked = np.ma.masked_where(~valid, mag)
        plt.figure(figsize=(8, 6))
        im = plt.imshow(masked, interpolation="nearest", aspect="auto")
        plt.colorbar(im, label=f"{list_name} magnitude")
        plt.title(f"MV magnitude {list_name} | POC={frame.poc} | shape={frame.height}x{frame.width}")
        plt.xlabel("x")
        plt.ylabel("y")
    elif mode == "quiver":
        stride = max(1, stride)
        X, Y = _sample_grid(frame.height, frame.width, stride)
        uu = u[::stride, ::stride]
        vv = v[::stride, ::stride]
        vv_mask = valid[::stride, ::stride]
        uu = np.where(vv_mask, uu, 0.0)
        vv = np.where(vv_mask, vv, 0.0)

        bg = np.ma.masked_where(~valid, mag)
        plt.figure(figsize=(9, 7))
        plt.imshow(bg, interpolation="nearest", aspect="auto")
        plt.colorbar(label=f"{list_name} magnitude")
        plt.quiver(X, Y, uu, vv, angles="xy", scale_units="xy", scale=1)
        plt.title(f"MV quiver {list_name} | POC={frame.poc} | shape={frame.height}x{frame.width} | stride={stride}")
        plt.xlabel("x")
        plt.ylabel("y")
    elif mode == "pred-flag":
        plt.figure(figsize=(8, 6))
        im = plt.imshow(frame.pred_flag, interpolation="nearest", aspect="auto", vmin=0, vmax=3)
        plt.colorbar(im, label="pred_flag")
        plt.title(f"Prediction flag | POC={frame.poc} | shape={frame.height}x{frame.width}")
        plt.xlabel("x")
        plt.ylabel("y")
    else:
        raise ValueError("--mode must be one of: quiver, magnitude, pred-flag")

    if out:
        plt.savefig(out, dpi=160, bbox_inches="tight")
    else:
        plt.show()


def visualize_ctu_bits(frame: CTUBitsFrame, field: str = "total", out: str | None = None):
    field_map = {
        "total": frame.total_bits,
        "motion": frame.motion_bits,
        "coeff": frame.coeff_bits,
    }
    if field not in field_map:
        raise ValueError("--field must be one of: total, motion, coeff")

    mat = field_map[field]
    plt.figure(figsize=(8, 6))
    im = plt.imshow(mat, interpolation="nearest", aspect="auto")
    plt.colorbar(im, label=f"{field}_bits")
    plt.title(
        f"CTU bits ({field}) | POC={frame.poc} | grid={frame.ctb_height}x{frame.ctb_width} | ctu_size={frame.ctu_size}"
    )
    plt.xlabel("ctu_x")
    plt.ylabel("ctu_y")
    if out:
        plt.savefig(out, dpi=160, bbox_inches="tight")
    else:
        plt.show()


def visualize_cu_size(frame: CUSizeFrame, out: str | None = None):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(frame.matrix, interpolation="nearest", aspect="auto")
    plt.colorbar(im, label="CU size")
    plt.title(f"CU size | POC={frame.poc} | shape={frame.height}x{frame.width}")
    plt.xlabel("x")
    plt.ylabel("y")
    if out:
        plt.savefig(out, dpi=160, bbox_inches="tight")
    else:
        plt.show()


# ----------------------------
# CLI
# ----------------------------

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Visualize exported HEVC binary data")
    sub = p.add_subparsers(dest="kind", required=True)

    qp = sub.add_parser("qp", help="Visualize QP export")
    qp.add_argument("path")
    qp.add_argument("--frame", type=int, default=0)
    qp.add_argument("--list-frames", action="store_true")
    qp.add_argument("--save", type=str, default=None)

    mv = sub.add_parser("mv", help="Visualize motion-vector export")
    mv.add_argument("path")
    mv.add_argument("--frame", type=int, default=0)
    mv.add_argument("--list-frames", action="store_true")
    mv.add_argument("--list", choices=["L0", "L1"], default="L0")
    mv.add_argument("--mode", choices=["quiver", "magnitude", "pred-flag"], default="quiver")
    mv.add_argument("--stride", type=int, default=4)
    mv.add_argument("--save", type=str, default=None)

    ctu = sub.add_parser("ctu-bits", help="Visualize CTU bits export")
    ctu.add_argument("path")
    ctu.add_argument("--frame", type=int, default=0)
    ctu.add_argument("--list-frames", action="store_true")
    ctu.add_argument("--field", choices=["total", "motion", "coeff"], default="total")
    ctu.add_argument("--save", type=str, default=None)

    cu = sub.add_parser("cu-size", help="Visualize CU size export")
    cu.add_argument("path")
    cu.add_argument("--frame", type=int, default=0)
    cu.add_argument("--list-frames", action="store_true")
    cu.add_argument("--save", type=str, default=None)

    return p


def main():
    args = make_parser().parse_args()

    if args.kind == "qp":
        frames = load_qp_file(args.path)
        print(frames[args.frame].matrix.shape)
        print(frames[args.frame].matrix[:15,:15])
        print("min/max:", frames[args.frame].matrix.min(), frames[args.frame].matrix.max())
        print("mean:" , frames[args.frame].matrix.mean(), "std:", frames[args.frame].matrix.std())
        print("unique:", np.unique(frames[args.frame].matrix)[:20])

        if args.list_frames:
            list_frames(frames)
            return
        frame = choose_frame(frames, args.frame)
        visualize_qp(frame, out=args.save)

    elif args.kind == "mv":
        frames = load_mv_file(args.path)
        if args.list_frames:
            list_frames(frames)
            return
        frame = choose_frame(frames, args.frame)
        visualize_mv(frame, list_name=args.list, mode=args.mode, stride=args.stride, out=args.save)

    elif args.kind == "ctu-bits":
        frames = load_ctu_bits_file(args.path)
        if args.list_frames:
            list_frames(frames)
            return
        frame = choose_frame(frames, args.frame)
        visualize_ctu_bits(frame, field=args.field, out=args.save)

    elif args.kind == "cu-size":
        frames = load_cu_size_file(args.path)
        if args.list_frames:
            list_frames(frames)
            return
        frame = choose_frame(frames, args.frame)
        visualize_cu_size(frame, out=args.save)


if __name__ == "__main__":
    main()
