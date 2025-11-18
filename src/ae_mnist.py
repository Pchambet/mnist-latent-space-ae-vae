# src/ae_mnist.py

import torch
import torch.nn as nn
from torch.nn.functional import interpolate


def conv_out_dim(w, k, s, p):
    """
    Compute output width of a Conv2d layer (square image).
    """
    return int(((w - k + 2 * p) / s) + 1)


def conv_out_dim_(nb_layers, w, k, s, p):
    """
    Apply conv_out_dim nb_layers times (for a stack of convs).
    """
    r = tuple()
    for _ in range(nb_layers):
        w = conv_out_dim(w, k, s, p)
        r += (w,)
    return r


def convTransp_out_dim(w, k, s, p, d):
    """
    Compute output width of a ConvTranspose2d layer (square image).
    Formula aligned with PyTorch docs.
    """
    # Hout = (Hin − 1) * stride − 2 * padding + dilation * (kernel_size − 1) + 1
    return int((w - 1) * s - 2 * p + d * (k - 1) + 1)


def convTransp_out_dim_(nb_layers, w, k, s, p, d):
    r = tuple()
    for _ in range(nb_layers):
        w = convTransp_out_dim(w, k, s, p, d)
        r += (w,)
    return r


class Decoder(nn.Module):
    """
    Décoder convolutionnel.
    Supposition : les deux dimensions spatiales des images sont identiques.
    """
    def __init__(self,
                 in_channels,
                 channels_out_seq=(256, 128, 64, 32, 1),
                 k=3, s=2, p=1, d=1, op=1,
                 k_p=None, s_p=None, p_p=None):

        if k_p is None:
            k_p = k
        if s_p is None:
            s_p = s
        if p_p is None:
            p_p = p

        super(Decoder, self).__init__()

        # fonctions globales éventuellement utiles
        self.layers_input_dim = lambda w: convTransp_out_dim_(len(channels_out_seq), w, k, s, p, d)
        self.channels_seq = channels_out_seq

        # construction du décodeur
        modules = []
        for c_outputs in channels_out_seq:
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels,
                                       out_channels=c_outputs,
                                       kernel_size=k,
                                       stride=s,
                                       padding=p,
                                       output_padding=op,
                                       dilation=d),
                    nn.BatchNorm2d(c_outputs),
                    nn.LeakyReLU())
            )
            in_channels = c_outputs

        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    """
    Encodeur convolutionnel.
    Supposition : les deux dimensions spatiales des images sont identiques.
    """
    def __init__(self,
                 in_channels,
                 channels_out_seq=(32, 64, 128, 256, 512),
                 k=3, s=2, p=1,
                 k_p=None, s_p=None, p_p=None):

        if k_p is None:
            k_p = k
        if s_p is None:
            s_p = s
        if p_p is None:
            p_p = p

        super(Encoder, self).__init__()

        self.layers_input_dim = lambda w: conv_out_dim_(len(channels_out_seq), w, k, s, p)
        self.channels_seq = channels_out_seq

        modules = []
        for c_outputs in channels_out_seq:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels,
                              out_channels=c_outputs,
                              kernel_size=k,
                              stride=s,
                              padding=p),
                    nn.BatchNorm2d(c_outputs),
                    nn.LeakyReLU())
            )
            in_channels = c_outputs

        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


class AE(nn.Module):
    """
    Auto-Encodeur convolutionnel simple pour MNIST (ou images 1 canal).
    """
    def __init__(self, in_channels=1, encoder_channels_seq=None):
        super(AE, self).__init__()

        if encoder_channels_seq is None:
            encoder_channels_seq = (32, 64, 128, 256, 512)

        self.input_channels = in_channels
        self.encoder_channels_seq = encoder_channels_seq

        # séquence de canaux pour le décodeur :
        # on supprime le dernier canal (bottleneck), on inverse, puis on rajoute in_channels
        decoder_channels_seq = encoder_channels_seq[-2::-1] + (in_channels,)

        # encodeur / décodeur
        self.encodeur = Encoder(in_channels, channels_out_seq=encoder_channels_seq)
        self.decodeur = Decoder(self.encodeur.channels_seq[-1],
                                channels_out_seq=decoder_channels_seq)

    def forward(self, x):
        """
        x : (B, C, H, W)
        retour : reconstruction (B, C, H, W)
        """
        input_dim = x.shape[2]   # H = W supposé
        x_latent = self.encodeur(x)
        x_reconstruit = self.decodeur(x_latent)
        x_interp = interpolate(x_reconstruit, size=input_dim, mode='bilinear', align_corners=False)
        return x_interp