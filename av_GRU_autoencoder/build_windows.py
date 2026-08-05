#!/usr/bin/env python3
"""
build_windows.py

Stage 2 utility:
Convert per-frame landmark CSV files from Stage 1 into fixed-length .npz
training windows for the AV-GRU autoencoder.

By default this creates audio placeholders so you can test the Stage 2 motion
pipeline quickly. For full AV training, use --audio-mode log_mel to extract a
compact log-mel audio target from the source video/audio file for each window.
"""

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

MP_LANDMARKS = 21
XYZ = 3
HAND_VEC = MP_LANDMARKS * XYZ
LANDMARK_COL_RE = re.compile(r"^hand(\d+)_(\d+)$")
SIDES = ("right", "left")
AUDIO_EXTENSIONS = (".wav", ".aif", ".aiff", ".flac", ".mp3", ".m4a", ".mp4", ".mov")


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


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + float(hz) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def make_mel_filter_bank(sample_rate, n_fft, n_mels, fmin=50.0, fmax=None):
    """Create a small triangular mel filter bank using only numpy."""
    nyquist = float(sample_rate) / 2.0
    fmax = nyquist if fmax is None else min(float(fmax), nyquist)
    fmin = max(0.0, min(float(fmin), fmax))

    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), int(n_mels) + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((int(n_fft) + 1) * hz_points / float(sample_rate)).astype(int)
    bins = np.clip(bins, 0, int(n_fft) // 2)

    filters = np.zeros((int(n_mels), int(n_fft) // 2 + 1), dtype=np.float32)
    for i in range(int(n_mels)):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center <= left:
            center = min(left + 1, filters.shape[1] - 1)
        if right <= center:
            right = min(center + 1, filters.shape[1])

        if center > left:
            filters[i, left:center] = (np.arange(left, center) - left) / float(center - left)
        if right > center:
            filters[i, center:right] = (right - np.arange(center, right)) / float(right - center)

        total = filters[i].sum()
        if total > 0:
            filters[i] /= total

    return filters


def decode_audio_ffmpeg(audio_path, sample_rate):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg is required for --audio-mode log_mel, but it was not found on PATH."
        )

    cmd = [
        ffmpeg,
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-f",
        "f32le",
        "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg could not decode audio from {audio_path}: {err}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def compute_log_mel_frames(audio, sample_rate, n_mels, n_fft, hop_length, fmin, fmax, top_db):
    audio = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0)
    if audio.size == 0:
        return np.zeros((0, int(n_mels)), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    n_fft = int(n_fft)
    hop_length = int(hop_length)
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("audio_n_fft and audio_hop_length must be positive.")

    if audio.size < n_fft:
        audio = np.pad(audio, (0, n_fft - audio.size))

    n_frames = 1 + int(np.ceil(max(0, audio.size - n_fft) / float(hop_length)))
    target_len = (n_frames - 1) * hop_length + n_fft
    if target_len > audio.size:
        audio = np.pad(audio, (0, target_len - audio.size))

    starts = np.arange(n_frames, dtype=np.int64) * hop_length
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(n_frames, n_fft),
        strides=(audio.strides[0] * hop_length, audio.strides[0]),
        writeable=False,
    )
    window = np.hanning(n_fft).astype(np.float32)
    spectrum = np.fft.rfft(frames * window[None, :], axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float32)

    filters = make_mel_filter_bank(sample_rate, n_fft, n_mels, fmin=fmin, fmax=fmax)
    mel_power = power @ filters.T
    max_power = float(np.max(mel_power)) if mel_power.size else 0.0
    if max_power <= 1e-12:
        log_mel = np.zeros_like(mel_power, dtype=np.float32)
    else:
        floor = max_power * (10.0 ** (-float(top_db) / 10.0))
        db = 10.0 * np.log10(np.maximum(mel_power, floor) / max_power)
        log_mel = np.clip((db + float(top_db)) / float(top_db), 0.0, 1.0).astype(np.float32)

    times_ms = ((starts.astype(np.float32) + n_fft / 2.0) / float(sample_rate)) * 1000.0
    return log_mel, times_ms.astype(np.float32)


def first_nonempty_video_value(df):
    if "video" not in df.columns:
        return ""
    values = df["video"].dropna().astype(str).str.strip()
    values = values[values != ""]
    return values.iloc[0] if not values.empty else ""


def resolve_existing_path(candidates):
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    return None


def resolve_audio_source(csv_path, df, source_stem, audio_source_dir=None):
    candidates = []
    video_value = first_nonempty_video_value(df)
    if video_value:
        video_path = Path(video_value)
        if video_path.is_absolute():
            candidates.append(video_path)
        else:
            candidates.extend([
                csv_path.parent / video_path,
                Path.cwd() / video_path,
                csv_path.parent.parent / video_path,
            ])

    if audio_source_dir:
        audio_dir = Path(audio_source_dir)
        candidates.extend(audio_dir / f"{source_stem}{ext}" for ext in AUDIO_EXTENSIONS)

    candidates.extend(
        csv_path.parent.parent / "Video_Data" / f"{source_stem}{ext}"
        for ext in AUDIO_EXTENSIONS
    )
    return resolve_existing_path(candidates)


class AudioFeatureExtractor:
    def __init__(
        self,
        mode="placeholder",
        audio_dim=1,
        placeholder_quality=0.0,
        sample_rate=16000,
        n_fft=512,
        hop_length=160,
        fmin=50.0,
        fmax=None,
        top_db=80.0,
        allow_missing=False,
    ):
        self.mode = mode
        self.audio_dim = int(audio_dim)
        self.placeholder_quality = float(placeholder_quality)
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.fmin = float(fmin)
        self.fmax = None if fmax is None else float(fmax)
        self.top_db = float(top_db)
        self.allow_missing = bool(allow_missing)
        self.cache = {}
        self.warned = set()

    @property
    def schema(self):
        if self.mode == "log_mel":
            return f"log_mel_{self.audio_dim}"
        return f"placeholder_{self.audio_dim}"

    def contract(self):
        return {
            "audio_mode": self.mode,
            "audio_schema": self.schema,
            "audio_dim": self.audio_dim,
            "audio_sample_rate": self.sample_rate if self.mode == "log_mel" else None,
            "audio_n_fft": self.n_fft if self.mode == "log_mel" else None,
            "audio_hop_length": self.hop_length if self.mode == "log_mel" else None,
            "audio_fmin": self.fmin if self.mode == "log_mel" else None,
            "audio_fmax": self.fmax if self.mode == "log_mel" else None,
            "audio_top_db": self.top_db if self.mode == "log_mel" else None,
        }

    def missing_feature(self, reason):
        if not self.allow_missing:
            raise RuntimeError(reason + " Pass --allow-missing-audio to write zero targets instead.")
        if reason not in self.warned:
            print(f"Warning: {reason} Writing zero audio targets with audio_quality=0.0.")
            self.warned.add(reason)
        return np.zeros((self.audio_dim,), dtype=np.float32), 0.0

    def load_log_mel(self, audio_path):
        audio_path = Path(audio_path)
        key = str(audio_path)
        if key not in self.cache:
            audio = decode_audio_ffmpeg(audio_path, self.sample_rate)
            log_mel, times_ms = compute_log_mel_frames(
                audio,
                sample_rate=self.sample_rate,
                n_mels=self.audio_dim,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                fmin=self.fmin,
                fmax=self.fmax,
                top_db=self.top_db,
            )
            self.cache[key] = (log_mel, times_ms)
        return self.cache[key]

    def feature_for_window(self, audio_path, start_time_ms, end_time_ms):
        if self.mode == "placeholder":
            return (
                np.zeros((self.audio_dim,), dtype=np.float32),
                self.placeholder_quality,
            )

        if audio_path is None:
            return self.missing_feature("No source audio/video path was found for this CSV.")

        try:
            log_mel, times_ms = self.load_log_mel(audio_path)
        except RuntimeError as exc:
            return self.missing_feature(str(exc))

        if log_mel.shape[0] == 0:
            return self.missing_feature(f"No decodable audio samples were found in {audio_path}.")

        start_time_ms = float(start_time_ms)
        end_time_ms = float(end_time_ms)
        mask = (times_ms >= start_time_ms) & (times_ms <= end_time_ms)
        if not mask.any():
            # Very short edge windows can miss frame centers; include a tiny margin.
            mask = (times_ms >= start_time_ms - 50.0) & (times_ms <= end_time_ms + 50.0)
        if not mask.any():
            return self.missing_feature(
                f"No audio feature frames overlapped {start_time_ms:.1f}-{end_time_ms:.1f} ms in {audio_path}."
            )

        return np.mean(log_mel[mask], axis=0).astype(np.float32), 1.0


def main():
    parser = argparse.ArgumentParser(description="Build .npz training windows from Stage 1 landmark CSVs.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pattern", default="*_landmarks.csv")
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--hop-length", type=int, default=30)
    parser.add_argument("--fill-mode", choices=["zero", "forward_fill"], default="zero")
    parser.add_argument("--audio-mode", choices=["placeholder", "log_mel"], default="placeholder")
    parser.add_argument("--audio-dim", type=int, default=None)
    parser.add_argument("--audio-quality", type=float, default=0.0, help="Quality value used for placeholder audio.")
    parser.add_argument("--audio-source-dir", default=None, help="Directory containing source audio/video files.")
    parser.add_argument("--audio-sample-rate", type=int, default=16000)
    parser.add_argument("--audio-n-fft", type=int, default=512)
    parser.add_argument("--audio-hop-length", type=int, default=160)
    parser.add_argument("--audio-fmin", type=float, default=50.0)
    parser.add_argument("--audio-fmax", type=float, default=None)
    parser.add_argument("--audio-top-db", type=float, default=80.0)
    parser.add_argument("--allow-missing-audio", action="store_true")
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
    audio_dim = int(args.audio_dim) if args.audio_dim is not None else (12 if args.audio_mode == "log_mel" else 1)
    audio_extractor = AudioFeatureExtractor(
        mode=args.audio_mode,
        audio_dim=audio_dim,
        placeholder_quality=args.audio_quality,
        sample_rate=args.audio_sample_rate,
        n_fft=args.audio_n_fft,
        hop_length=args.audio_hop_length,
        fmin=args.audio_fmin,
        fmax=args.audio_fmax,
        top_db=args.audio_top_db,
        allow_missing=args.allow_missing_audio,
    )

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Audio mode: {args.audio_mode} ({audio_extractor.schema})")

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
        if args.audio_mode == "log_mel" and timestamps is None:
            raise ValueError(f"{csv_path} has no timestamp_ms column; log-mel audio alignment requires timestamps.")
        source_stem = csv_path.stem.replace("_landmarks", "")
        camera_id = infer_id(r"(cam\d+)", source_stem)
        session_id = infer_id(r"(ses\d+)", source_stem)
        audio_source = resolve_audio_source(csv_path, df, source_stem, args.audio_source_dir)
        if args.audio_mode == "log_mel":
            print(f"Audio source: {audio_source if audio_source else 'not found'}")

        for win_idx, (start, end, win_motion) in enumerate(
            make_windows(motion, args.sequence_length, args.hop_length)
        ):
            out_path = output_dir / f"{source_stem}_win{win_idx:06d}.npz"

            start_time_ms = float(timestamps[start]) if timestamps is not None else float(start)
            end_time_ms = float(timestamps[end - 1]) if timestamps is not None else float(end - 1)
            audio_feature, audio_quality = audio_extractor.feature_for_window(
                audio_source,
                start_time_ms,
                end_time_ms,
            )

            if not out_path.exists() or args.overwrite:
                feature_contract = {
                    "feature_order": "right_63_left_63" + ("_velocity_126" if args.include_velocity else ""),
                    "motion_dim": int(win_motion.shape[1]),
                    "include_velocity": bool(args.include_velocity),
                    "normalize_to_wrist": bool(args.normalize_to_wrist),
                    "normalize_scale_landmark": int(args.normalize_scale_landmark),
                    "hand_order": args.hand_order,
                    "missing_hand_value": 0.0,
                    **audio_extractor.contract(),
                }
                np.savez_compressed(
                    out_path,
                    motion=win_motion.astype(np.float32),
                    audio=audio_feature.astype(np.float32),
                    audio_quality=np.float32(audio_quality),
                    audio_schema=audio_extractor.schema,
                    audio_mode=args.audio_mode,
                    audio_source=str(audio_source) if audio_source else "",
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
                "audio_dim": audio_dim,
                "audio_quality": float(audio_quality),
                "audio_mode": args.audio_mode,
                "audio_schema": audio_extractor.schema,
                "audio_source": str(audio_source) if audio_source else "",
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
        "audio_mode": args.audio_mode,
        "audio_schema": audio_extractor.schema,
        "audio_dim": audio_dim,
        "audio_quality": args.audio_quality if args.audio_mode == "placeholder" else "per_window",
        "audio_sample_rate": args.audio_sample_rate if args.audio_mode == "log_mel" else None,
        "audio_n_fft": args.audio_n_fft if args.audio_mode == "log_mel" else None,
        "audio_hop_length": args.audio_hop_length if args.audio_mode == "log_mel" else None,
        "audio_fmin": args.audio_fmin if args.audio_mode == "log_mel" else None,
        "audio_fmax": args.audio_fmax if args.audio_mode == "log_mel" else None,
        "audio_top_db": args.audio_top_db if args.audio_mode == "log_mel" else None,
        "allow_missing_audio": args.allow_missing_audio,
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
