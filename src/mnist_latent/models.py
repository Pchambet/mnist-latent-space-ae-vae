"""Convolutional autoencoder and VAE, in their original ("legacy") and corrected forms.

The legacy variants reproduce the original architecture exactly: every decoder block, including the
last one, is ConvTranspose -> BatchNorm -> LeakyReLU, so the output image is batch-normalised
and unbounded instead of a pixel intensity in [0, 1]. The fixed variants only change the
output head (a plain ConvTranspose producing logits, squashed by a sigmoid); the rest of the
architecture is identical, so the gap between the two isolates the effect of the head (and,
for the VAE, of the loss that goes with it). For the denoising AE that gap is small: the
16-dim code of this architecture, not its head, is what limits it.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn.functional import interpolate

Head = Literal["legacy", "sigmoid"]
DEFAULT_CHANNELS = (16, 32, 64, 32, 16)


def _block(conv: nn.Module, channels: int) -> nn.Sequential:
    return nn.Sequential(conv, nn.BatchNorm2d(channels), nn.LeakyReLU())


class Encoder(nn.Sequential):
    """Stack of stride-2 3x3 convolutions: 28 -> 14 -> 7 -> 4 -> 2 -> 1 for five layers."""

    def __init__(self, in_channels: int, channels: tuple[int, ...]) -> None:
        layers = []
        for c in channels:
            layers.append(_block(nn.Conv2d(in_channels, c, 3, stride=2, padding=1), c))
            in_channels = c
        super().__init__(*layers)


class Decoder(nn.Sequential):
    """Stack of stride-2 transposed convolutions: 1 -> 2 -> 4 -> 8 -> 16 -> 32.

    With ``head="sigmoid"`` the last block is a bare transposed convolution returning
    logits; with ``head="legacy"`` it keeps BatchNorm + LeakyReLU like the hidden blocks.
    """

    def __init__(self, in_channels: int, channels: tuple[int, ...], head: Head) -> None:
        layers: list[nn.Module] = []
        for i, c in enumerate(channels):
            conv = nn.ConvTranspose2d(in_channels, c, 3, stride=2, padding=1, output_padding=1)
            is_last = i == len(channels) - 1
            layers.append(conv if (is_last and head == "sigmoid") else _block(conv, c))
            in_channels = c
        super().__init__(*layers)


def _decoder_channels(encoder_channels: tuple[int, ...], out_channels: int) -> tuple[int, ...]:
    """Mirror the encoder, dropping the bottleneck width and ending on the image channels."""
    return tuple(encoder_channels[-2::-1]) + (out_channels,)


class AE(nn.Module):
    """Convolutional autoencoder; the latent code is the 1x1 bottleneck feature map."""

    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, ...] = DEFAULT_CHANNELS,
        head: Head = "sigmoid",
        image_size: int = 28,
    ) -> None:
        super().__init__()
        self.head = head
        self.image_size = image_size
        self.encoder = Encoder(in_channels, channels)
        self.decoder = Decoder(channels[-1], _decoder_channels(channels, in_channels), head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the reconstructed image (B, C, 28, 28); in [0, 1] for the sigmoid head."""
        out = self.decoder(self.encoder(x))
        out = interpolate(out, size=self.image_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(out) if self.head == "sigmoid" else out


class VAE(nn.Module):
    """Convolutional VAE with a diagonal-Gaussian posterior q(z|x) = N(mu(x), diag(sigma^2(x))).

    ``decode`` returns logits for the sigmoid head (Bernoulli likelihood) and raw pixel
    values for the legacy head; ``mean_image`` gives the image-space mean in both cases.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        in_channels: int = 1,
        channels: tuple[int, ...] = DEFAULT_CHANNELS,
        head: Head = "sigmoid",
        image_size: int = 28,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.head = head
        self.image_size = image_size
        self.encoder = Encoder(in_channels, channels)
        with torch.no_grad():
            feat = self.encoder.eval()(torch.zeros(2, in_channels, image_size, image_size))
        self.encoder.train()
        self.feature_shape = tuple(feat.shape[1:])
        n_feat = feat[0].numel()
        self.fc_mu = nn.Linear(n_feat, latent_dim)
        self.fc_logvar = nn.Linear(n_feat, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, n_feat)
        self.decoder = Decoder(channels[-1], _decoder_channels(channels, in_channels), head)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = mu + sigma * eps keeps the sample differentiable in (mu, logvar)."""
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z).view(-1, *self.feature_shape)
        out = self.decoder(h)
        return interpolate(out, size=self.image_size, mode="bilinear", align_corners=False)

    def mean_image(self, decoded: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(decoded) if self.head == "sigmoid" else decoded

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        return self.decode(self.reparameterize(mu, logvar)), mu, logvar
