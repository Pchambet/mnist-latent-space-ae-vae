# src/vae_mnist.py

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from .ae_mnist import Encoder, Decoder  # même encodeur/décodeur que l'AE


class VAE(nn.Module):
    """
    Variational Autoencoder convolutionnel pour MNIST.

    - Encoder CNN -> mu, logvar (espace latent de dimension latent_dim)
    - Reparametrization trick : z = mu + sigma * eps
    - Decoder CNN via une couche fully-connected intermédiaire
    """

    def __init__(self,
                 input_dim: int,
                 in_channels: int = 1,
                 encoder_channels_seq=None,
                 latent_dim: int = 16):
        """
        input_dim : taille des images d'entrée (28 pour MNIST)
        in_channels : nombre de canaux en entrée (1 pour MNIST)
        encoder_channels_seq : séquence des canaux pour l'encodeur
        latent_dim : dimension de l'espace latent (taille de z)
        """
        super(VAE, self).__init__()

        if encoder_channels_seq is None:
            encoder_channels_seq = (32, 64, 128, 256, 512)

        self.input_dim = input_dim
        self.in_channels = in_channels
        self.encoder_channels_seq = encoder_channels_seq
        self.latent_dim = latent_dim

        # --- Encodeur convolutionnel ---
        self.encodeur = Encoder(in_channels, channels_out_seq=encoder_channels_seq)

        # On calcule la taille de la feature map en sortie de l'encodeur
        # en faisant un passage à vide avec un tenseur factice.
        # On utilise un batch_size=2 et le mode eval pour ne pas casser BatchNorm.
        self.encodeur.eval()
        with torch.no_grad():
            dummy = torch.zeros(2, in_channels, input_dim, input_dim)
            feat = self.encodeur(dummy)
        self.encodeur.train()  # on repasse en mode train pour l'entraînement réel

        self.feature_channels = feat.shape[1]
        self.feature_h = feat.shape[2]
        self.feature_w = feat.shape[3]
        self.feature_size = self.feature_channels * self.feature_h * self.feature_w

        # --- Projections vers mu et logvar ---
        self.fc_mu = nn.Linear(self.feature_size, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_size, latent_dim)

        # --- Projection inverse : z -> feature map pour le décodeur ---
        self.fc_decode = nn.Linear(latent_dim, self.feature_size)

        # --- Décodeur convolutionnel ---
        decoder_channels_seq = encoder_channels_seq[-2::-1] + (in_channels,)
        self.decodeur = Decoder(self.feature_channels,
                                channels_out_seq=decoder_channels_seq)

    def encode(self, x):
        """
        Encode x -> mu, logvar
        """
        x_feat = self.encodeur(x)                # (B, C, H, W)
        x_flat = x_feat.flatten(1)               # (B, C*H*W)
        mu = self.fc_mu(x_flat)                  # (B, latent_dim)
        logvar = self.fc_logvar(x_flat)          # (B, latent_dim)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparametrization trick :
        z = mu + sigma * eps, eps ~ N(0, I)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        """
        Decode z -> reconstruction x_hat
        """
        h = self.fc_decode(z)                                                # (B, feature_size)
        h = h.view(-1, self.feature_channels, self.feature_h, self.feature_w)  # (B, C, H, W)
        x_recon = self.decodeur(h)                                           # (B, in_channels, H', W')
        x_interp = interpolate(x_recon,
                               size=self.input_dim,
                               mode='bilinear',
                               align_corners=False)
        return x_interp

    def forward(self, x):
        """
        Retourne :
        - x_recon : reconstruction
        - mu, logvar : paramètres de la gaussienne q(z|x)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    @staticmethod
    def reconstruction_loss(x, x_recon):
        """
        Coût de reconstruction (MSE).
        """
        return nn.MSELoss()(x_recon, x)

    @staticmethod
    def kl_divergence(mu, logvar):
        """
        KL divergence entre q(z|x) = N(mu, sigma^2)
        et p(z) = N(0, I).

        Formule classique :
        KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        """
        # mu, logvar : (B, latent_dim)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # (B,)
        return torch.mean(kld)

    def full_loss(self, x, x_recon, mu, logvar, beta=1.0):
        """
        Retourne :
        - loss totale = recon_loss + beta * KLD
        - recon_loss détaché
        - kld détaché
        """
        recon = self.reconstruction_loss(x, x_recon)
        kld = self.kl_divergence(mu, logvar)
        loss = recon + beta * kld
        return loss, recon.detach(), kld.detach()