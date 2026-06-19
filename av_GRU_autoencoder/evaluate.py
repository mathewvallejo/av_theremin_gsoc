import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clusters_csv', default='outputs/clustering/cluster_assignments.csv')
    args = p.parse_args()
    out = Path('outputs/reports'); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.clusters_csv)
    z_cols = [c for c in df.columns if c.startswith('z_')]
    labels = df['cluster'].values
    valid = labels >= 0
    report = {'n_windows': int(len(df)), 'n_clusters_excluding_noise': int(len(set(labels[valid]))), 'noise_windows': int((labels < 0).sum())}
    if valid.sum() > 2 and len(set(labels[valid])) > 1:
        z = df.loc[valid, z_cols].values
        y = labels[valid]
        report['silhouette'] = float(silhouette_score(z, y))
        report['davies_bouldin'] = float(davies_bouldin_score(z, y))
    else:
        report['silhouette'] = None
        report['davies_bouldin'] = None
    with (out / 'evaluation_report.json').open('w') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
