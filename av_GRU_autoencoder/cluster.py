import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap

from src.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--embeddings_csv', default=None)
    parser.add_argument('--method', default=None, choices=['hdbscan', 'kmeans'])
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--min_cluster_size', type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(cfg['data']['output_dir'])

    clustering_cfg = cfg.get('clustering', {})

    method = args.method or clustering_cfg.get('method', 'hdbscan')
    k = args.k if args.k is not None else clustering_cfg.get('kmeans_clusters', 8)
    min_cluster_size = (
        args.min_cluster_size
        if args.min_cluster_size is not None
        else clustering_cfg.get('min_cluster_size', 20)
    )
    umap_neighbors = clustering_cfg.get('umap_neighbors', 25)
    umap_min_dist = clustering_cfg.get('umap_min_dist', 0.1)

    clustering_dir = out_dir / 'clustering'
    clustering_dir.mkdir(parents=True, exist_ok=True)

    embeddings_csv = (
        Path(args.embeddings_csv)
        if args.embeddings_csv
        else out_dir / 'embeddings' / 'embeddings.csv'
    )

    if not embeddings_csv.exists():
        raise FileNotFoundError(
            f"Embeddings CSV not found: {embeddings_csv}\n"
            "Run embed.py first, or pass --embeddings_csv explicitly."
        )

    df = pd.read_csv(embeddings_csv)
    z_cols = [c for c in df.columns if c.startswith('z_')]

    if not z_cols:
        raise ValueError("Embeddings CSV must contain latent columns named z_XX.")

    z = df[z_cols].values

    scaler = StandardScaler().fit(z)
    zs = scaler.transform(z)
    joblib.dump(scaler, clustering_dir / 'embedding_scaler.joblib')

    n_neighbors = min(int(umap_neighbors), max(2, len(df) - 1))
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=umap_min_dist,
        random_state=42,
    )
    xy = reducer.fit_transform(zs)

    df['umap_x'] = xy[:, 0]
    df['umap_y'] = xy[:, 1]

    joblib.dump(reducer, clustering_dir / 'umap_model.joblib')

    if method == 'hdbscan':
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            prediction_data=True,
        )
        labels = clusterer.fit_predict(zs)
    else:
        clusterer = KMeans(
            n_clusters=k,
            random_state=42,
            n_init='auto',
        )
        labels = clusterer.fit_predict(zs)

    df['cluster'] = labels

    joblib.dump(clusterer, clustering_dir / 'cluster_model.joblib')

    assignments_csv = clustering_dir / 'cluster_assignments.csv'
    df.to_csv(assignments_csv, index=False)

    print(df['cluster'].value_counts().sort_index())
    print(f'Wrote {assignments_csv}')


if __name__ == '__main__':
    main()
