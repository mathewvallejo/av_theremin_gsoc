import torch
from torch import nn


class AVGRUAutoencoder(nn.Module):
    """Audio-guided GRU autoencoder for landmark motion sequences.

    Input to the encoder is motion only. Audio is used as an auxiliary
    self-supervised target during training, gated by audio quality.
    This allows the deployed runtime to use camera/MediaPipe input only.
    """

    def __init__(self, motion_dim, audio_dim, hidden_dim=128, latent_dim=24,
                 num_layers=1, dropout=0.1, bidirectional=True):
        super().__init__()
        self.motion_dim = motion_dim
        self.audio_dim = audio_dim
        self.latent_dim = latent_dim
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.encoder = nn.GRU(
            input_size=motion_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        enc_out_dim = hidden_dim * self.num_directions
        self.to_latent = nn.Sequential(
            nn.LayerNorm(enc_out_dim),
            nn.Linear(enc_out_dim, latent_dim),
        )
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
        )
        self.decoder = nn.GRU(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.motion_head = nn.Linear(hidden_dim, motion_dim)
        self.audio_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, audio_dim),
        )

    def encode(self, motion_seq):
        enc, _ = self.encoder(motion_seq)
        pooled = enc.mean(dim=1)
        z = self.to_latent(pooled)
        return z

    def forward(self, motion_seq):
        z = self.encode(motion_seq)
        b, t, _ = motion_seq.shape
        dec_in = z.unsqueeze(1).repeat(1, t, 1)
        dec_out, _ = self.decoder(dec_in)
        motion_recon = self.motion_head(dec_out)
        audio_pred = self.audio_head(z)
        return {
            "z": z,
            "motion_recon": motion_recon,
            "audio_pred": audio_pred,
        }
