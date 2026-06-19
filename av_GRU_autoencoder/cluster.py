import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--embeddings_csv', default='outputs/embeddings/embeddings.csv')
    p.add_argument('--method', default='hdbscan', choices=['hdbscan', 'kmeans'])
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--min_cluster_size', type=int, default=20)
    args = p.parse_args()

    out = Path('outputs/clustering'); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.embeddings_csv)
    z_cols = [c for c in df.columns if c.startswith('z_')]
    z = df[z_cols].values
    scaler = StandardScaler().fit(z)
    zs = scaler.transform(z)
    joblib.dump(scaler, out / 'embedding_scaler.joblib')

    reducer = umap.UMAP(n_neighbors=25, min_dist=0.1, random_state=42)
    xy = reducer.fit_transform(zs)
    df['umap_x'] = xy[:, 0]; df['umap_y'] = xy[:, 1]
    joblib.dump(reducer, out / 'umap_model.joblib')

    if args.method == 'hdbscan':
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size, prediction_data=True)
        labels = clusterer.fit_predict(zs)
    else:
        clusterer = KMeans(n_clusters=args.k, random_state=42, n_init='auto')
        labels = clusterer.fit_predict(zs)
    df['cluster'] = labels
    joblib.dump(clusterer, out / 'cluster_model.joblib')
    df.to_csv(out / 'cluster_assignments.csv', index=False)
    print(df['cluster'].value_counts().sort_index())
    print(f'Wrote {out / "cluster_assignments.csv"}')

if __name__ == '__main__': main()
