import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

DEFAULT_ARTIFACT_FILES = {
    "encoder_path": "encoder.pt",
    "feature_scaler_path": "feature_scaler.joblib",
    "cluster_model_path": "cluster_model.joblib",
    "runtime_model_config_path": "runtime_model_config.json",
    "embedding_scaler_path": "embedding_scaler.joblib",
    "cluster_names_path": "cluster_names.json",
}


def apply_artifact_dir_override(model_cfg, artifact_dir):
    """Use one Stage 2 export folder instead of any configured per-file overrides."""
    model_cfg["artifact_dir"] = artifact_dir
    for key in DEFAULT_ARTIFACT_FILES:
        model_cfg.pop(key, None)
    return model_cfg


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


def resolve_runtime_model_paths(model_cfg):
    """Resolve Stage 3 model artifacts from one export directory plus optional overrides."""
    artifact_dir = Path(model_cfg.get("artifact_dir", "models"))

    def resolve(key, required=True):
        explicit_path = model_cfg.get(key)
        if explicit_path:
            return explicit_path

        candidate = artifact_dir / DEFAULT_ARTIFACT_FILES[key]
        if required or candidate.exists():
            return candidate
        return None

    return {
        "encoder_path": resolve("encoder_path"),
        "feature_scaler_path": resolve("feature_scaler_path"),
        "cluster_model_path": resolve("cluster_model_path"),
        "runtime_model_config_path": resolve("runtime_model_config_path"),
        "embedding_scaler_path": resolve("embedding_scaler_path", required=False),
        "cluster_names_path": resolve("cluster_names_path", required=False),
    }


def infer_cluster_input_dtype(cluster_model, fallback=np.float32):
    """Match sklearn cluster prediction input to the fitted model arrays."""
    for attr in ("cluster_centers_", "_fit_X", "_raw_data", "components_", "means_"):
        value = getattr(cluster_model, attr, None)
        if isinstance(value, np.ndarray) and value.dtype.kind == "f":
            return value.dtype

    prediction_data = getattr(cluster_model, "prediction_data_", None)
    raw_data = getattr(prediction_data, "raw_data", None)
    if isinstance(raw_data, np.ndarray) and raw_data.dtype.kind == "f":
        return raw_data.dtype

    return np.dtype(fallback)


def nearest_center_cluster(cluster_model, z_in):
    centers = getattr(cluster_model, "cluster_centers_", None)
    if not isinstance(centers, np.ndarray):
        return None

    centers = np.asarray(centers, dtype=z_in.dtype)
    diff = centers - z_in[0]
    distances = np.einsum("ij,ij->i", diff, diff)
    return int(np.argmin(distances))


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
                "Run Stage 2 export_for_runtime.py and set runtime_model.artifact_dir to that export_for_runtime folder."
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
        self.cluster_input_dtype = infer_cluster_input_dtype(self.cluster_model)
        print(f"Cluster prediction dtype: {self.cluster_input_dtype}")

        self.embedding_scaler = None
        if embedding_scaler_path and Path(embedding_scaler_path).exists():
            self.embedding_scaler = joblib.load(embedding_scaler_path)

        self.cluster_names = {}
        if cluster_names_path and Path(cluster_names_path).exists():
            self.cluster_names = load_json_optional(cluster_names_path, default={}) or {}

    @classmethod
    def from_config(cls, model_cfg, device="cpu"):
        return cls(**resolve_runtime_model_paths(model_cfg), device=device)

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
        z_in = np.asarray(z_in, dtype=self.cluster_input_dtype, order="C")

        confidence = 1.0
        nearest_cluster = nearest_center_cluster(self.cluster_model, z_in)
        if nearest_cluster is not None:
            cluster = nearest_cluster
        elif hasattr(self.cluster_model, "predict"):
            cluster = int(self.cluster_model.predict(z_in)[0])
        elif hasattr(self.cluster_model, "prediction_data_"):
            try:
                import hdbscan
                labels, strengths = hdbscan.approximate_predict(self.cluster_model, z_in)
                cluster = int(labels[0])
                confidence = float(strengths[0])
            except Exception as exc:
                print(f"HDBSCAN approximate prediction failed: {exc}")
                cluster = -1
        else:
            cluster = -1

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
