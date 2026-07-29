from pathlib import Path
import argparse
import json

import pandas as pd
import matplotlib.pyplot as plt

from src.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(cfg["data"]["output_dir"])

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    hist_path = run_dir / "metrics" / "training_history.json"
    if hist_path.exists():
        hist = pd.DataFrame(json.loads(hist_path.read_text()))

        plt.figure()
        plt.plot(hist["epoch"], hist["train_total"], label="train")
        plt.plot(hist["epoch"], hist["val_total"], label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "train_val_loss.png", dpi=160)
        plt.close()
    else:
        print(f"Missing training history: {hist_path}")

    cl_path = run_dir / "clustering" / "cluster_assignments.csv"
    if cl_path.exists():
        df = pd.read_csv(cl_path)

        if {"umap_x", "umap_y", "cluster"}.issubset(df.columns):
            plt.figure()
            plt.scatter(df["umap_x"], df["umap_y"], c=df["cluster"], s=8)
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
            plt.tight_layout()
            plt.savefig(plots_dir / "embedding_umap_clusters.png", dpi=160)
            plt.close()

        if "cluster" in df.columns:
            plt.figure()
            df["cluster"].value_counts().sort_index().plot(kind="bar")
            plt.xlabel("cluster")
            plt.ylabel("windows")
            plt.tight_layout()
            plt.savefig(plots_dir / "cluster_distribution.png", dpi=160)
            plt.close()
    else:
        print(f"Missing cluster assignments: {cl_path}")

    print(f"Wrote plots to {plots_dir}")


if __name__ == "__main__":
    main()