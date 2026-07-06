import argparse
from pathlib import Path
import shutil
import json

from src.config import load_config


def build_runtime_model_config(cfg):
    """Create the compact model/feature metadata consumed by Stage 3."""
    features = cfg.get("features", {})
    model = cfg.get("model", {})
    return {
        "format_version": 1,
        "model_type": "av_gru_autoencoder",
        "features": {
            "motion_dim": int(features.get("motion_dim", 126)),
            "feature_dim": int(features.get("motion_dim", 126)),
            "sequence_length": int(features.get("sequence_length", 60)),
            "audio_dim": int(features.get("audio_dim", 1)),
            "include_velocity": bool(features.get("include_velocity", False)),
            "normalize_to_wrist": bool(features.get("normalize_to_wrist", True)),
            "normalize_scale_landmark": int(features.get("normalize_scale_landmark", 9)),
        },
        "model": {
            "motion_dim": int(features.get("motion_dim", 126)),
            "input_dim": int(features.get("motion_dim", 126)),
            "audio_dim": int(features.get("audio_dim", 1)),
            "hidden_dim": int(model.get("hidden_dim", 64)),
            "latent_dim": int(model.get("latent_dim", 16)),
            "num_layers": int(model.get("num_layers", 1)),
            "dropout": float(model.get("dropout", 0.1)),
            "bidirectional": bool(model.get("bidirectional", True)),
        },
    }


def copy_required(source_path, dest_path, copied, missing):
    if source_path.exists():
        shutil.copy2(source_path, dest_path)
        copied.append(str(dest_path))
        return True
    missing.append(str(source_path))
    print(f"Missing required export item: {source_path}")
    return False


def copy_optional(source_path, dest_path, copied):
    if source_path.exists():
        shutil.copy2(source_path, dest_path)
        copied.append(str(dest_path))
        return True
    print(f"Missing optional export item: {source_path}")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--export_dir', default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    src = Path(cfg['data']['output_dir'])

    dst = (
        Path(args.export_dir)
        if args.export_dir
        else Path(cfg.get('runtime_export', {}).get('export_dir', src / 'export_for_runtime'))
    )
    dst.mkdir(parents=True, exist_ok=True)

    copied = []
    missing = []

    # Stage 3 standard names. encoder.pt is the canonical runtime checkpoint name.
    copy_required(src / 'checkpoints' / 'best_model.pt', dst / 'encoder.pt', copied, missing)
    copy_required(src / 'scalers' / 'feature_scaler.joblib', dst / 'feature_scaler.joblib', copied, missing)
    copy_required(src / 'clustering' / 'cluster_model.joblib', dst / 'cluster_model.joblib', copied, missing)
    copy_optional(src / 'clustering' / 'embedding_scaler.joblib', dst / 'embedding_scaler.joblib', copied)

    # Backwards-compatible alias for older Stage 3 notes/scripts.
    if (dst / 'encoder.pt').exists():
        shutil.copy2(dst / 'encoder.pt', dst / 'av_gru_encoder.pt')
        copied.append(str(dst / 'av_gru_encoder.pt'))

    cluster_names = dst / 'cluster_names.json'
    if not cluster_names.exists():
        cluster_names.write_text(json.dumps({'-1': 'noise_or_transition'}, indent=2))
    copied.append(str(cluster_names))

    runtime_cfg = build_runtime_model_config(cfg)
    runtime_cfg_path = dst / 'runtime_model_config.json'
    runtime_cfg_path.write_text(json.dumps(runtime_cfg, indent=2))
    copied.append(str(runtime_cfg_path))

    manifest = {
        'format_version': 2,
        'source_output_dir': str(src),
        'export_dir': str(dst),
        'runtime_model_config': str(runtime_cfg_path),
        'canonical_files': {
            'encoder': str(dst / 'encoder.pt'),
            'feature_scaler': str(dst / 'feature_scaler.joblib'),
            'embedding_scaler': str(dst / 'embedding_scaler.joblib'),
            'cluster_model': str(dst / 'cluster_model.joblib'),
            'cluster_names': str(cluster_names),
            'runtime_model_config': str(runtime_cfg_path),
        },
        'copied_files': copied,
        'missing_files': missing,
    }

    manifest_path = dst / 'runtime_export_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if missing:
        raise FileNotFoundError(
            'Runtime export is incomplete. Missing required files:\n' + '\n'.join(missing)
        )

    print(f'Runtime package written to {dst}')
    print(f'Wrote {runtime_cfg_path}')
    print(f'Wrote {manifest_path}')


if __name__ == '__main__':
    main()
