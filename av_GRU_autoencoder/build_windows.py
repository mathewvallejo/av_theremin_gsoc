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


def infer_feature_columns(df):
    cols = [c for c in df.columns if c.startswith("hand")]
    if not cols:
        raise ValueError("No landmark columns found. Expected hand0_*, hand1_* columns.")

    def sort_key(name):
        prefix, idx = name.split("_")
        return (int(prefix.replace("hand", "")), int(idx))

    return sorted(cols, key=sort_key)


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

        feature_cols = infer_feature_columns(df)
        motion = df[feature_cols].to_numpy(dtype=np.float32)
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
