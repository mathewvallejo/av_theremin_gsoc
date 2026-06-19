"""Export trained Stage 2 artifacts for the Stage 3 realtime runtime.

This script:
1. Copies the scaler and cluster model files.
2. Exports the encoder as a TorchScript module (encoder_scripted.pt).
   TorchScript captures the exact encode() graph including mean pooling
   and LayerNorm, so Stage 3 does not need to redefine the architecture.
3. Writes a runtime_model_config.json that Stage 3 reads for feature dims.

Usage:
    python export_for_runtime.py --config configs/default.yaml
"""

import argparse
import json
import shutil
from pathlib import Path

import torch

from models.av_gru_autoencoder import AVGRUAutoencoder
from src.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Export Stage 2 artifacts for Stage 3 runtime.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--checkpoint", default=None,
        help="Path to checkpoint .pt file. Defaults to outputs/checkpoints/best_model.pt."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["data"]["output_dir"])
    export_dir = Path(cfg["runtime_export"]["export_dir"])
    export_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else out_dir / "checkpoints" / "best_model.pt"

    # --- Copy scaler and cluster artifacts ---
    items_to_copy = [
        (out_dir / "scalers" / "feature_scaler.joblib", export_dir / "feature_scaler.joblib"),
        (out_dir / "clustering" / "embedding_scaler.joblib", export_dir / "embedding_scaler.joblib"),
        (out_dir / "clustering" / "cluster_model.joblib", export_dir / "cluster_model.joblib"),
    ]
    for src, dst in items_to_copy:
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied {src.name} -> {dst}")
        else:
            print(f"Optional item not found, skipping: {src}")

    # --- Stub cluster_names.json if not already present ---
    cluster_names_dst = export_dir / "cluster_names.json"
    if not cluster_names_dst.exists():
        cluster_names_dst.write_text(json.dumps({"-1": "noise_or_transition"}, indent=2))
        print(f"Created stub {cluster_names_dst.name}")

    # --- Export encoder as TorchScript ---
    # TorchScript exports the exact encode() graph, including the mean
    # pooling and LayerNorm, so Stage 3 does not have to redefine the
    # architecture and cannot silently diverge.
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = AVGRUAutoencoder(
        motion_dim=cfg["features"]["motion_dim"],
        audio_dim=cfg["features"]["audio_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        latent_dim=cfg["model"]["latent_dim"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
        bidirectional=cfg["model"]["bidirectional"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Trace encode() with a dummy input of the correct shape.
    seq_len = cfg["features"]["sequence_length"]
    motion_dim = cfg["features"]["motion_dim"]
    dummy = torch.zeros(1, seq_len, motion_dim)
    scripted_encoder = torch.jit.trace(model.encode, dummy)
    scripted_path = export_dir / "encoder_scripted.pt"
    scripted_encoder.save(str(scripted_path))
    print(f"Exported TorchScript encoder -> {scripted_path}")

    # --- Write runtime model config ---
    runtime_cfg = {
        "feature_dim": cfg["features"]["motion_dim"],
        "sequence_length": cfg["features"]["sequence_length"],
        "latent_dim": cfg["model"]["latent_dim"],
        "model": {
            "hidden_dim": cfg["model"]["hidden_dim"],
            "latent_dim": cfg["model"]["latent_dim"],
            "num_layers": cfg["model"]["num_layers"],
            "bidirectional": cfg["model"]["bidirectional"],
        },
    }
    config_path = export_dir / "runtime_model_config.json"
    config_path.write_text(json.dumps(runtime_cfg, indent=2))
    print(f"Wrote {config_path.name}")

    print(f"\nRuntime package ready at: {export_dir}")
    print("Copy the contents of this folder into av_Gesture_OSC_runtime/models/")


if __name__ == "__main__":
    main()
