"""Gesture model runtime loader for Stage 3.

Loads the TorchScript encoder exported by Stage 2's export_for_runtime.py.
Using TorchScript guarantees the encode() graph — including mean pooling
and LayerNorm — is identical to what was trained, with no risk of
architectural drift between stages.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import torch


def load_json_optional(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r") as f:
        return json.load(f)


class GestureRuntimeModel:
    def __init__(
        self,
        encoder_path,
        feature_scaler_path,
        cluster_model_path,
        runtime_model_config_path,
        embedding_scaler_path=None,
        cluster_names_path=None,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.cfg = load_json_optional(runtime_model_config_path, default={})

        # Load TorchScript encoder exported by Stage 2 export_for_runtime.py.
        # This is the exact encode() graph including mean pooling and LayerNorm.
        self.encoder = torch.jit.load(encoder_path, map_location=self.device)
        self.encoder.eval()

        self.feature_scaler = joblib.load(feature_scaler_path)
        self.cluster_model = joblib.load(cluster_model_path)

        self.embedding_scaler = None
        if embedding_scaler_path and Path(embedding_scaler_path).exists():
            self.embedding_scaler = joblib.load(embedding_scaler_path)

        self.cluster_names = {}
        if cluster_names_path and Path(cluster_names_path).exists():
            self.cluster_names = load_json_optional(cluster_names_path, default={}) or {}

    def embed(self, window):
        """Encode a motion window to a latent vector.

        Args:
            window: np.ndarray of shape [T, D] — one rolling window of motion features.

        Returns:
            np.ndarray of shape [latent_dim].
        """
        window = np.asarray(window, dtype=np.float32)
        T, D = window.shape

        scaled = self.feature_scaler.transform(window.reshape(T, D))
        x = torch.from_numpy(scaled.reshape(1, T, D)).float().to(self.device)

        with torch.no_grad():
            z = self.encoder(x).cpu().numpy()[0]

        return z.astype(np.float32)

    def predict_cluster(self, z):
        """Assign a cluster label to a latent vector.

        Returns:
            tuple: (cluster_id, confidence, name)
        """
        z_in = z.reshape(1, -1)
        if self.embedding_scaler is not None:
            z_in = self.embedding_scaler.transform(z_in)

        if hasattr(self.cluster_model, "predict"):
            cluster = int(self.cluster_model.predict(z_in)[0])
        elif hasattr(self.cluster_model, "approximate_predict"):
            cluster = int(self.cluster_model.approximate_predict(z_in)[0])
        else:
            cluster = -1

        confidence = 1.0
        if hasattr(self.cluster_model, "predict_proba"):
            probs = self.cluster_model.predict_proba(z_in)[0]
            confidence = float(np.max(probs))

        name = self.cluster_names.get(str(cluster), self.cluster_names.get(cluster, f"cluster_{cluster}"))
        return cluster, confidence, name

    def infer(self, window):
        """Full inference: window → latent → cluster.

        Returns:
            dict with keys: latent, cluster, confidence, name.
        """
        z = self.embed(window)
        cluster, confidence, name = self.predict_cluster(z)
        return {
            "latent": z,
            "cluster": cluster,
            "confidence": confidence,
            "name": name,
        }
