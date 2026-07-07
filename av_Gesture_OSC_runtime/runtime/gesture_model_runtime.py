import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    """
    Runtime encoder matching Stage 2 AVGRUAutoencoder.encode().
    Stage 2 architecture:
      encoder GRU -> mean pool over time -> LayerNorm -> Linear to latent.
    """
    def __init__(self, input_dim, hidden_dim=64, latent_dim=16, num_layers=1, dropout=0.1, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        directions = 2 if bidirectional else 1
        enc_out_dim = hidden_dim * directions
        self.to_latent = nn.Sequential(
            nn.LayerNorm(enc_out_dim),
            nn.Linear(enc_out_dim, latent_dim),
        )

    def forward(self, x):
        enc, _ = self.encoder(x)
        pooled = enc.mean(dim=1)
        return self.to_latent(pooled)


def load_json_optional(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r") as f:
        return json.load(f)


def resolve_existing_path(path, alternates=None, label="file"):
    candidates = [Path(path)]
    for alt in alternates or []:
        candidates.append(Path(alt))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not find {label}. Tried:\n{tried}")


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
        self.cfg = load_json_optional(runtime_model_config_path, default=None)
        if self.cfg is None:
            raise FileNotFoundError(
                f"Runtime model config not found: {runtime_model_config_path}\n"
                "Run Stage 2 export_for_runtime.py again and copy runtime_model_config.json into this runtime models folder."
            )

        model_cfg = self.cfg.get("model", {})
        feature_cfg = self.cfg.get("features", {})
        input_dim = int(model_cfg.get("input_dim", model_cfg.get("motion_dim", feature_cfg.get("feature_dim", 126))))
        hidden_dim = int(model_cfg.get("hidden_dim", 64))
        latent_dim = int(model_cfg.get("latent_dim", 16))
        num_layers = int(model_cfg.get("num_layers", 1))
        dropout = float(model_cfg.get("dropout", 0.1))
        bidirectional = bool(model_cfg.get("bidirectional", True))

        encoder_path = resolve_existing_path(
            encoder_path,
            alternates=[Path(encoder_path).with_name("av_gru_encoder.pt")],
            label="encoder checkpoint",
        )
        self.encoder = GRUEncoder(input_dim, hidden_dim, latent_dim, num_layers, dropout, bidirectional)
        state = torch.load(encoder_path, map_location=self.device)

        # Accept Stage 2 best_model.pt dict or raw state dict.
        if isinstance(state, dict) and "model_state_dict" in state:
            full_state = state["model_state_dict"]
            state = {
                k: v for k, v in full_state.items()
                if k.startswith("encoder.") or k.startswith("to_latent.")
            }
        elif isinstance(state, dict) and "encoder_state_dict" in state:
            state = state["encoder_state_dict"]

        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        critical_missing = [k for k in missing if k.startswith("encoder") or k.startswith("to_latent")]
        if critical_missing:
            raise RuntimeError(
                "Encoder checkpoint does not match runtime architecture. Missing keys: "
                + ", ".join(critical_missing)
            )
        if unexpected:
            print("Ignoring non-runtime checkpoint keys:", ", ".join(unexpected[:8]))

        self.encoder.to(self.device)
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
        """
        window shape: [T, D]
        Applies feature scaler framewise, then encoder.
        """
        window = np.asarray(window, dtype=np.float32)
        T, D = window.shape
        expected_dim = int(self.cfg.get("features", {}).get("feature_dim", self.cfg.get("model", {}).get("input_dim", D)))
        if D != expected_dim:
            raise ValueError(f"Runtime feature dimension mismatch: got {D}, expected {expected_dim}")

        flat_scaled = self.feature_scaler.transform(window.reshape(T, D))
        x = torch.from_numpy(flat_scaled.reshape(1, T, D)).float().to(self.device)

        with torch.no_grad():
            z = self.encoder(x).cpu().numpy()[0]

        return z.astype(np.float32)

    def predict_cluster(self, z):
        z_in = z.reshape(1, -1)
        if self.embedding_scaler is not None:
            z_in = self.embedding_scaler.transform(z_in)

        if hasattr(self.cluster_model, "predict"):
            cluster = int(self.cluster_model.predict(z_in)[0])
        elif hasattr(self.cluster_model, "approximate_predict"):
            try:
                import hdbscan
                cluster, strengths = hdbscan.approximate_predict(self.cluster_model, z_in)
                cluster = int(cluster[0])
            except Exception:
                cluster = -1
        else:
            cluster = -1

        confidence = 1.0
        if hasattr(self.cluster_model, "predict_proba"):
            probs = self.cluster_model.predict_proba(z_in)[0]
            confidence = float(np.max(probs))

        name = self.cluster_names.get(str(cluster), self.cluster_names.get(cluster, f"cluster_{cluster}"))
        return cluster, confidence, name

    def infer(self, window):
        z = self.embed(window)
        cluster, confidence, name = self.predict_cluster(z)
        return {
            "latent": z,
            "cluster": cluster,
            "confidence": confidence,
            "name": name,
        }
