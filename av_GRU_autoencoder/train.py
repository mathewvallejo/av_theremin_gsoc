import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import joblib
from sklearn.preprocessing import StandardScaler

from src.config import load_config
from src.dataset import discover_manifest, make_splits, AVWindowDataset
from src.losses import av_loss
from models.av_gru_autoencoder import AVGRUAutoencoder


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def choose_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def fit_motion_scaler(rows, sample_limit=5000):
    if rows.empty:
        raise RuntimeError(
            "Training split is empty. Check window_manifest.csv, path values, "
            "and val/test split fractions."
        )

    mats = []
    for p in rows['path'].head(sample_limit):
        arr = np.load(p)['motion'].astype(np.float32)
        mats.append(arr.reshape(-1, arr.shape[-1]))

    if not mats:
        raise RuntimeError(
            "No motion arrays loaded from training split. Check that manifest paths exist."
        )

    scaler = StandardScaler()
    scaler.fit(np.vstack(mats))
    return scaler


def apply_scaler_batch(batch, scaler):
    x = batch['motion'].numpy()
    b, t, d = x.shape
    x2 = scaler.transform(x.reshape(-1, d)).reshape(b, t, d)
    batch['motion'] = torch.from_numpy(x2.astype(np.float32))
    return batch


def run_epoch(model, loader, optimizer, cfg, device, scaler=None, train=True):
    model.train(train)
    totals = []
    for batch in tqdm(loader, leave=False):
        if scaler is not None:
            batch = apply_scaler_batch(batch, scaler)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.set_grad_enabled(train):
            outputs = model(batch['motion'])
            loss, parts = av_loss(outputs, batch, cfg)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['training']['grad_clip'])
                optimizer.step()
        totals.append(parts)
    return {k: float(np.mean([d[k] for d in totals])) for k in totals[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg['seed'])
    out_dir = Path(cfg['data']['output_dir'])
    ckpt_dir = out_dir / 'checkpoints'
    scaler_dir = out_dir / 'scalers'
    metrics_dir = out_dir / 'metrics'
    for d in [ckpt_dir, scaler_dir, metrics_dir]: d.mkdir(parents=True, exist_ok=True)

    df = discover_manifest(cfg)
    splits = make_splits(df, cfg['data']['val_fraction'], cfg['data']['test_fraction'], cfg['seed'])
    for name, split_df in splits.items():
        split_df.to_csv(out_dir / f'{name}_split.csv', index=False)

    scaler = fit_motion_scaler(splits['train'])
    joblib.dump(scaler, scaler_dir / 'feature_scaler.joblib')

    train_ds, val_ds = AVWindowDataset(splits['train']), AVWindowDataset(splits['val'])
    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['training']['batch_size'], shuffle=False)

    device = choose_device(cfg['training']['device'])
    model = AVGRUAutoencoder(
        motion_dim=cfg['features']['motion_dim'],
        audio_dim=cfg['features']['audio_dim'],
        hidden_dim=cfg['model']['hidden_dim'],
        latent_dim=cfg['model']['latent_dim'],
        num_layers=cfg['model']['num_layers'],
        dropout=cfg['model']['dropout'],
        bidirectional=cfg['model']['bidirectional'],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])

    history = []
    best_val = float('inf')
    patience = cfg['training']['patience']
    bad_epochs = 0
    for epoch in range(1, cfg['training']['epochs'] + 1):
        train_m = run_epoch(model, train_loader, optimizer, cfg, device, scaler, train=True)
        val_m = run_epoch(model, val_loader, optimizer, cfg, device, scaler, train=False)
        rec = {'epoch': epoch, **{f'train_{k}': v for k, v in train_m.items()}, **{f'val_{k}': v for k, v in val_m.items()}}
        history.append(rec)
        print(json.dumps(rec, indent=2))
        if val_m['total'] < best_val:
            best_val = val_m['total']; bad_epochs = 0
            torch.save({'model_state_dict': model.state_dict(), 'config': cfg}, ckpt_dir / 'best_model.pt')
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print('Early stopping.')
                break
    with (metrics_dir / 'training_history.json').open('w') as f:
        json.dump(history, f, indent=2)

if __name__ == '__main__':
    main()
