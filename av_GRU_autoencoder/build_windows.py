#!/usr/bin/env python3
"""
build_windows.py

Stage 2 utility:
Convert per-frame landmark CSV files from Stage 1 into fixed-length .npz
training windows for the AV-GRU autoencoder.

This version creates audio placeholders so you can test the Stage 2 motion
pipeline before integrating camera-audio features.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

MP_LANDMARKS = 21
XYZ = 3
HAND_VEC = MP_LANDMARKS * XYZ
LANDMARK_COL_RE = re.compile(r"^hand(\d+)_(\d+)$")
SIDES = ("right", "left")


def infer_feature_columns(df):
    cols = [c for c in df.columns if LANDMARK_COL_RE.match(c)]
    if not cols:
        raise ValueError("No landmark columns found. Expected hand0_*, hand1_* columns.")

    def sort_key(name):
        match = LANDMARK_COL_RE.match(name)
        return (int(match.group(1)), int(match.group(2)))

    return sorted(cols, key=sort_key)


def normalize_hand_vec(vec, scale_landmark=9):
    pts = vec.reshape(MP_LANDMARKS, XYZ).astype(np.float32)
    wrist = pts[0].copy()
    pts = pts - wrist
    scale = np.linalg.norm(pts[int(scale_landmark)])
    if scale > 1e-6:
        pts = pts / scale
    return pts.reshape(-1)


def clean_label(value):
    if pd.isna(value):
        return ""
    label = str(value).strip().lower()
    return label if label in SIDES else ""


def assign_side(label, hand_i, occupied, hand_order):
    if hand_order == "label" and label in SIDES and not occupied[label]:
        return label

    fallback = SIDES[hand_i] if hand_i < len(SIDES) else ""
    if fallback and not occupied[fallback]:
        return fallback

    for side in SIDES:
        if not occupied[side]:
            return side
    return ""


def row_to_base_feature(row, normalize_to_wrist=True, normalize_scale_landmark=9, hand_order="label"):
    hands = {side: np.zeros(HAND_VEC, dtype=np.float32) for side in SIDES}
    occupied = {side: False for side in SIDES}

    for hand_i in range(2):
        cols = [f"hand{hand_i}_{j}" for j in range(HAND_VEC)]
        missing = [c for c in cols if c not in row.index]
        if missing:
            raise ValueError(f"Missing landmark columns for hand{hand_i}; first missing column: {missing[0]}")

        vec = row[cols].to_numpy(dtype=np.float32)
        if not np.isfinite(vec).any():
            continue

        vec = np.nan_to_num(vec, nan=0.0).astype(np.float32)
        if normalize_to_wrist:
            vec = normalize_hand_vec(vec, normalize_scale_landmark)

        label = clean_label(row.get(f"hand{hand_i}_label", ""))
        side = assign_side(label, hand_i, occupied, hand_order)
        if side:
            hands[side] = vec
            occupied[side] = True

    return np.concatenate([hands["right"], hands["left"]]).astype(np.float32)


def build_motion_array(df, normalize_to_wrist=True, normalize_scale_landmark=9, include_velocity=False, hand_order="label"):
    base = np.stack(
        [
            row_to_base_feature(
                row,
                normalize_to_wrist=normalize_to_wrist,
                normalize_scale_landmark=normalize_scale_landmark,
                hand_order=hand_order,
            )
            for _, row in df.iterrows()
        ],
        axis=0,
    ).astype(np.float32)

    if not include_velocity:
        return base

    velocity = np.zeros_like(base)
    if base.shape[0] > 1:
        velocity[1:] = base[1:] - base[:-1]
    return np.concatenate([base, velocity], axis=1).astype(np.float32)


def clean_motion_array(arr, fill_mode):
    if fill_mode == "zero":
        return np.nan_to_num(arr, nan=0.0).astype(np.float32)
    if fill_mode == "forward_fill":
        return pd.DataFrame(arr).ffill().fillna(0.0).to_numpy(dtype=np.float32)
    raise ValueError(f"Unknown fill_mode: {fill_mode}")


def make_windows(arr, sequence_length, hop_length):
    n = arr.shape[0]
    for start in range(0, max(0, n - sequence_length + 1), hop_length):
        end = start + sequence_length
        yield start, end, arr[start:end]


def infer_id(pattern, name):
    match = re.search(pattern, name)
    return match.group(1) if match else ""


def main():
    parser = argparse.ArgumentParser(description="Build .npz training windows from Stage 1 landmark CSVs.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pattern", default="*_landmarks.csv")
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--hop-length", type=int, default=30)
    parser.add_argument("--fill-mode", choices=["zero", "forward_fill"], default="zero")
    parser.add_argument("--audio-dim", type=int, default=1)
    parser.add_argument("--audio-quality", type=float, default=0.0)
    parser.add_argument("--include-velocity", action="store_true", help="Append frame-to-frame velocity features.")
    parser.add_argument("--normalize-to-wrist", dest="normalize_to_wrist", action="store_true", default=True)
    parser.add_argument("--no-normalize-to-wrist", dest="normalize_to_wrist", action="store_false")
    parser.add_argument("--normalize-scale-landmark", type=int, default=9)
    parser.add_argument("--hand-order", choices=["label", "index"], default="label")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")

    csv_files = sorted(input_dir.glob(args.pattern))
    if args.max_files is not None:
        csv_files = csv_files[:args.max_files]
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {input_dir} matching {args.pattern}")

    manifest_rows = []
    total_windows = 0

    for csv_path in csv_files:
        print(f"Reading {csv_path}")
        df = pd.read_csv(csv_path)

        infer_feature_columns(df)
        if args.hand_order == "label" and not {"hand0_label", "hand1_label"}.issubset(df.columns):
            print(f"Warning: {csv_path.name} has no handedness labels; falling back to hand0=right, hand1=left.")

        motion = build_motion_array(
            df,
            normalize_to_wrist=args.normalize_to_wrist,
            normalize_scale_landmark=args.normalize_scale_landmark,
            include_velocity=args.include_velocity,
            hand_order=args.hand_order,
        )
        motion = clean_motion_array(motion, args.fill_mode)

        if motion.shape[0] < args.sequence_length:
            print(f"Skipping {csv_path.name}: only {motion.shape[0]} frames")
            continue

        timestamps = df["timestamp_ms"].to_numpy() if "timestamp_ms" in df.columns else None
        source_stem = csv_path.stem.replace("_landmarks", "")
        camera_id = infer_id(r"(cam\d+)", source_stem)
        session_id = infer_id(r"(ses\d+)", source_stem)

        for win_idx, (start, end, win_motion) in enumerate(
            make_windows(motion, args.sequence_length, args.hop_length)
        ):
            out_path = output_dir / f"{source_stem}_win{win_idx:06d}.npz"

            start_time_ms = float(timestamps[start]) if timestamps is not None else float(start)
            end_time_ms = float(timestamps[end - 1]) if timestamps is not None else float(end - 1)

            if not out_path.exists() or args.overwrite:
                feature_contract = {
                    "feature_order": "right_63_left_63" + ("_velocity_126" if args.include_velocity else ""),
                    "motion_dim": int(win_motion.shape[1]),
                    "include_velocity": bool(args.include_velocity),
                    "normalize_to_wrist": bool(args.normalize_to_wrist),
                    "normalize_scale_landmark": int(args.normalize_scale_landmark),
                    "hand_order": args.hand_order,
                    "missing_hand_value": 0.0,
                }
                np.savez_compressed(
                    out_path,
                    motion=win_motion.astype(np.float32),
                    audio=np.zeros((args.audio_dim,), dtype=np.float32),
                    audio_quality=np.float32(args.audio_quality),
                    source_csv=str(csv_path),
                    source_video=source_stem,
                    camera_id=camera_id,
                    session_id=session_id,
                    start_frame=np.int64(start),
                    end_frame=np.int64(end - 1),
                    start_time_ms=np.float32(start_time_ms),
                    end_time_ms=np.float32(end_time_ms),
                    feature_contract=json.dumps(feature_contract),
                )

            manifest_rows.append({
                "path": str(out_path),
                "window_file": str(out_path),
                "source_csv": str(csv_path),
                "source_video": source_stem,
                "camera_id": camera_id,
                "session_id": session_id,
                "start_frame": start,
                "end_frame": end - 1,
                "start_time_ms": start_time_ms,
                "end_time_ms": end_time_ms,
                "motion_shape": f"{win_motion.shape[0]}x{win_motion.shape[1]}",
                "audio_dim": args.audio_dim,
                "audio_quality": float(args.audio_quality),
                "include_velocity": bool(args.include_velocity),
                "normalize_to_wrist": bool(args.normalize_to_wrist),
                "normalize_scale_landmark": int(args.normalize_scale_landmark),
                "hand_order": args.hand_order,
            })
            total_windows += 1

    manifest_path = output_dir / "window_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_csv_files": len(csv_files),
        "total_windows": total_windows,
        "sequence_length": args.sequence_length,
        "hop_length": args.hop_length,
        "fill_mode": args.fill_mode,
        "audio_dim": args.audio_dim,
        "audio_quality": args.audio_quality,
        "motion_dim": int(manifest_rows[0]["motion_shape"].split("x")[1]) if manifest_rows else 0,
        "include_velocity": args.include_velocity,
        "normalize_to_wrist": args.normalize_to_wrist,
        "normalize_scale_landmark": args.normalize_scale_landmark,
        "hand_order": args.hand_order,
        "manifest": str(manifest_path),
        "manifest_has_required_path_column": True,
    }
    summary_path = output_dir / "window_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("")
    print("Done.")
    print(f"CSV files processed: {len(csv_files)}")
    print(f"Windows listed/written: {total_windows}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
