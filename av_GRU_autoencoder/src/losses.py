import torch
import torch.nn.functional as F


def latent_smoothness(z):
    if z.ndim != 2 or z.shape[0] < 2:
        return torch.tensor(0.0, device=z.device)
    return torch.mean((z[1:] - z[:-1]) ** 2)


def av_loss(outputs, batch, cfg):
    weights = cfg['loss']
    motion_loss = F.mse_loss(outputs['motion_recon'], batch['motion'])

    audio_target = batch['audio']
    if audio_target.ndim == 3:
        audio_target = audio_target.mean(dim=1)
    audio_quality = batch['audio_quality'].view(-1, 1)
    audio_pred = outputs['audio_pred']
    audio_mse = (audio_pred - audio_target) ** 2
    audio_weight = float(weights.get('audio_prediction_weight', 0.0))
    use_audio = bool(cfg.get('features', {}).get('use_audio_guidance', audio_weight > 0.0))
    if use_audio and audio_weight > 0.0:
        threshold = float(cfg.get('features', {}).get('audio_quality_threshold', 0.0))
        quality_weight = torch.where(
            audio_quality >= threshold,
            audio_quality,
            torch.zeros_like(audio_quality),
        )
        gated_audio_loss = torch.mean(audio_mse * quality_weight)
    else:
        gated_audio_loss = torch.zeros((), dtype=motion_loss.dtype, device=motion_loss.device)

    smooth_loss = latent_smoothness(outputs['z'])
    total = (
        weights['motion_reconstruction_weight'] * motion_loss +
        audio_weight * gated_audio_loss +
        weights['latent_smoothness_weight'] * smooth_loss
    )
    return total, {
        'total': float(total.detach().cpu()),
        'motion': float(motion_loss.detach().cpu()),
        'audio': float(gated_audio_loss.detach().cpu()),
        'smooth': float(smooth_loss.detach().cpu()),
    }
