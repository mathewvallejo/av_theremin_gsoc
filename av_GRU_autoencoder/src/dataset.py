from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class AVWindowDataset(Dataset):
    """Dataset for pre-windowed motion/audio sequences.

    Supported storage:
    1. manifest CSV with columns: path, split(optional), source_video(optional), start_time(optional), end_time(optional)
       Each .npz file must contain motion, audio, audio_quality.
    2. dataset_dir containing .npz files with those keys.
    """

    def __init__(self, rows):
        self.rows = rows.reset_index(drop=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        data = np.load(row['path'])
        motion = data['motion'].astype(np.float32)
        audio = data['audio'].astype(np.float32)
        audio_quality = np.array(data.get('audio_quality', 1.0), dtype=np.float32).reshape(1)
        return {
            'motion': torch.from_numpy(motion),
            'audio': torch.from_numpy(audio),
            'audio_quality': torch.from_numpy(audio_quality),
            'index': idx,
        }


def discover_manifest(cfg):
    data_cfg = cfg['data']
    manifest = Path(data_cfg.get('manifest_csv', ''))
    dataset_dir = Path(data_cfg.get('dataset_dir', ''))

    if manifest.exists():
        df = pd.read_csv(manifest)
        if 'path' not in df.columns:
            raise ValueError('manifest_csv must include a path column')
        df['path'] = df['path'].apply(lambda p: str(Path(p)))
        return df

    files = sorted(dataset_dir.glob('*.npz'))
    if not files:
        raise FileNotFoundError(
            f'No .npz windows found in {dataset_dir}. Run Stage 1 preprocessing/windowing first.'
        )
    return pd.DataFrame({'path': [str(p) for p in files]})


def make_splits(df, val_fraction=0.15, test_fraction=0.15, seed=42):
    if 'split' in df.columns:
        return {
            'train': df[df['split'] == 'train'].copy(),
            'val': df[df['split'] == 'val'].copy(),
            'test': df[df['split'] == 'test'].copy(),
        }
    train_val, test = train_test_split(df, test_size=test_fraction, random_state=seed, shuffle=True)
    rel_val = val_fraction / max(1e-9, 1.0 - test_fraction)
    train, val = train_test_split(train_val, test_size=rel_val, random_state=seed, shuffle=True)
    return {'train': train, 'val': val, 'test': test}
