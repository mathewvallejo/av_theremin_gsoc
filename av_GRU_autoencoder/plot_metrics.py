from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt

out = Path('outputs/plots'); out.mkdir(parents=True, exist_ok=True)
hist_path = Path('outputs/metrics/training_history.json')
if hist_path.exists():
    hist = pd.DataFrame(json.loads(hist_path.read_text()))
    plt.figure()
    plt.plot(hist['epoch'], hist['train_total'], label='train')
    plt.plot(hist['epoch'], hist['val_total'], label='val')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend(); plt.tight_layout()
    plt.savefig(out / 'train_val_loss.png', dpi=160)

cl_path = Path('outputs/clustering/cluster_assignments.csv')
if cl_path.exists():
    df = pd.read_csv(cl_path)
    plt.figure()
    plt.scatter(df['umap_x'], df['umap_y'], c=df['cluster'], s=8)
    plt.xlabel('UMAP 1'); plt.ylabel('UMAP 2'); plt.tight_layout()
    plt.savefig(out / 'embedding_umap_clusters.png', dpi=160)
    plt.figure()
    df['cluster'].value_counts().sort_index().plot(kind='bar')
    plt.xlabel('cluster'); plt.ylabel('windows'); plt.tight_layout()
    plt.savefig(out / 'cluster_distribution.png', dpi=160)
print(f'Wrote plots to {out}')
