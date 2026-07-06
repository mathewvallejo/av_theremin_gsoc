import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config import load_config
from src.dataset import discover_manifest
from models.av_gru_autoencoder import AVGRUAutoencoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg['data']['output_dir'])

    emb_dir = out_dir / 'embeddings'
    emb_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = out_dir / 'scalers' / 'feature_scaler.joblib'
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else out_dir / 'checkpoints' / 'best_model.pt'
    )

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Feature scaler not found: {scaler_path}\n"
            "Run train.py first."
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run train.py first, or pass --checkpoint explicitly."
        )

    scaler = joblib.load(scaler_path)
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    model = AVGRUAutoencoder(
        motion_dim=cfg['features']['motion_dim'],
        audio_dim=cfg['features']['audio_dim'],
        hidden_dim=cfg['model']['hidden_dim'],
        latent_dim=cfg['model']['latent_dim'],
        num_layers=cfg['model']['num_layers'],
        dropout=cfg['model']['dropout'],
        bidirectional=cfg['model']['bidirectional'],
    )

    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    df = discover_manifest(cfg)

    if df.empty:
        raise RuntimeError("Manifest contains no rows. Rebuild windows or check manifest_csv.")

    rows = []
    z_list = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        data = np.load(row['path'])
        x = data['motion'].astype(np.float32)
        x = scaler.transform(x.reshape(-1, x.shape[-1])).reshape(1, x.shape[0], x.shape[1])

        with torch.no_grad():
            z = model.encode(torch.from_numpy(x)).numpy()[0]

        z_list.append(z)
        rows.append(row.to_dict())

    if not z_list:
        raise RuntimeError("No embeddings were generated.")

    z_arr = np.vstack(z_list)
    np.save(emb_dir / 'embeddings.npy', z_arr)

    meta = pd.DataFrame(rows)
    for j in range(z_arr.shape[1]):
        meta[f'z_{j:02d}'] = z_arr[:, j]

    embeddings_csv = emb_dir / 'embeddings.csv'
    meta.to_csv(embeddings_csv, index=False)

    print(f'Wrote {embeddings_csv}')
    print(f'Wrote {emb_dir / "embeddings.npy"}')


if __name__ == '__main__':
    main()
