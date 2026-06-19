import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from models.av_gru_autoencoder import AVGRUAutoencoder
from src.config import load_config
from src.dataset import discover_manifest


def embed_window(model, scaler, row, device):
    """Load one windowed .npz, scale it, and return the latent vector."""
    data = np.load(row["path"])
    x = data["motion"].astype(np.float32)

    T, D = x.shape
    x_scaled = scaler.transform(x.reshape(T, D)).reshape(1, T, D)
    x_tensor = torch.from_numpy(x_scaled).float().to(device)

    with torch.no_grad():
        z = model.encode(x_tensor).cpu().numpy()[0]

    return z


def main():
    parser = argparse.ArgumentParser(description="Embed all windows into the latent space.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pt")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg["data"]["output_dir"])
    emb_dir = out_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = out_dir / "scalers" / "feature_scaler.joblib"
    scaler = joblib.load(scaler_path)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
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

    device = torch.device("cpu")

    df = discover_manifest(cfg)
    z_list = []
    meta_rows = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Embedding"):
        z = embed_window(model, scaler, row, device)
        z_list.append(z)
        meta_rows.append(row.to_dict())

    z_arr = np.vstack(z_list)
    np.save(emb_dir / "embeddings.npy", z_arr)

    meta = pd.DataFrame(meta_rows)
    for j in range(z_arr.shape[1]):
        meta[f"z_{j:02d}"] = z_arr[:, j]
    meta.to_csv(emb_dir / "embeddings.csv", index=False)

    print(f"Wrote {emb_dir / 'embeddings.npy'} and {emb_dir / 'embeddings.csv'}")
    print(f"Embedded {len(z_list)} windows, latent dim {z_arr.shape[1]}")


if __name__ == "__main__":
    main()
