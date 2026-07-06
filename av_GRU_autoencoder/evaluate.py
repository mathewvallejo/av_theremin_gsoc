import argparse
from pathlib import Path
import json

import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score

from src.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--clusters_csv', default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg['data']['output_dir'])

    reports_dir = out_dir / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)

    clusters_csv = (
        Path(args.clusters_csv)
        if args.clusters_csv
        else out_dir / 'clustering' / 'cluster_assignments.csv'
    )

    if not clusters_csv.exists():
        raise FileNotFoundError(
            f"Cluster assignments not found: {clusters_csv}\n"
            "Run cluster.py first, or pass --clusters_csv explicitly."
        )

    df = pd.read_csv(clusters_csv)
    z_cols = [c for c in df.columns if c.startswith('z_')]

    if 'cluster' not in df.columns:
        raise ValueError("cluster_assignments.csv must contain a 'cluster' column.")

    if not z_cols:
        raise ValueError("cluster_assignments.csv must contain latent columns named z_XX.")

    labels = df['cluster'].values
    valid = labels >= 0

    report = {
        'clusters_csv': str(clusters_csv),
        'n_windows': int(len(df)),
        'n_clusters_excluding_noise': int(len(set(labels[valid]))),
        'noise_windows': int((labels < 0).sum()),
    }

    if valid.sum() > 2 and len(set(labels[valid])) > 1:
        z = df.loc[valid, z_cols].values
        y = labels[valid]
        report['silhouette'] = float(silhouette_score(z, y))
        report['davies_bouldin'] = float(davies_bouldin_score(z, y))
    else:
        report['silhouette'] = None
        report['davies_bouldin'] = None

    report_path = reports_dir / 'evaluation_report.json'
    with report_path.open('w') as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f'Wrote {report_path}')


if __name__ == '__main__':
    main()
