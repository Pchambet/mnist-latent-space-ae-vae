# vae_mnist.py

# -*- coding: utf-8 -*-
"""
TP VAE sur MNIST
- Entraînement d'un VAE convolutionnel
- Visualisation des reconstructions

Auteur : Pierre Chambet (d'après support de TP d'Aymeric Chazottes)
"""

import matplotlib
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import ssl
import certifi

def _https_context(*args, **kwargs):
    return ssl.create_default_context(cafile=certifi.where())

ssl._create_default_https_context = _https_context

from src.vae_mnist import VAE


def build_dataloaders_vae(batch_size=512, root='./', train_size=10000, val_size=10000):
    """
    Version simplifiée de build_dataloaders pour le VAE.

    Ici, on ne met pas de bruit artificiel en entrée :
    les images sont supposées "propres", et le bruit est dans l'espace latent.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.,), (1.,)),
    ])

    dataset = datasets.MNIST(root=root, download=True, train=True, transform=transform)
    train_set, val_set, _ = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, len(dataset) - train_size - val_size]
    )
    test_set = datasets.MNIST(root=root, download=True, train=False, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_vae(model, train_loader, val_loader, device, num_epochs=20, beta=1.0):
    """
    Entraîne un VAE sur MNIST.

    Loss = reconstruction + beta * KLD.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = {
        "train_loss": [],
        "train_recon": [],
        "train_kld": [],
        "val_loss": [],
        "val_recon": [],
        "val_kld": [],
    }

    best_state_dict = None
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):

        # === TRAIN ===
        model.train()
        running_loss = 0.0
        running_recon = 0.0
        running_kld = 0.0

        for inputs, _ in train_loader:
            inputs = inputs.to(device)

            optimizer.zero_grad()
            x_recon, mu, logvar = model(inputs)
            loss, recon, kld = model.full_loss(inputs, x_recon, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_recon += recon.item() * batch_size
            running_kld += kld.item() * batch_size

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_recon = running_recon / len(train_loader.dataset)
        epoch_train_kld = running_kld / len(train_loader.dataset)

        history["train_loss"].append(epoch_train_loss)
        history["train_recon"].append(epoch_train_recon)
        history["train_kld"].append(epoch_train_kld)

        # === VALIDATION ===
        model.eval()
        running_loss_val = 0.0
        running_recon_val = 0.0
        running_kld_val = 0.0

        with torch.no_grad():
            for inputs_val, _ in val_loader:
                inputs_val = inputs_val.to(device)
                x_recon_val, mu_val, logvar_val = model(inputs_val)
                loss_val, recon_val, kld_val = model.full_loss(inputs_val, x_recon_val, mu_val, logvar_val, beta=beta)

                batch_size_val = inputs_val.size(0)
                running_loss_val += loss_val.item() * batch_size_val
                running_recon_val += recon_val.item() * batch_size_val
                running_kld_val += kld_val.item() * batch_size_val

        epoch_val_loss = running_loss_val / len(val_loader.dataset)
        epoch_val_recon = running_recon_val / len(val_loader.dataset)
        epoch_val_kld = running_kld_val / len(val_loader.dataset)

        history["val_loss"].append(epoch_val_loss)
        history["val_recon"].append(epoch_val_recon)
        history["val_kld"].append(epoch_val_kld)

        # sauvegarde des meilleurs poids (en fonction de la loss totale val)
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state_dict = model.state_dict()

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"[VAE] Epoch {epoch:3d} | "
                f"Train: loss={epoch_train_loss:.4f}, recon={epoch_train_recon:.4f}, kld={epoch_train_kld:.4f} | "
                f"Val: loss={epoch_val_loss:.4f}, recon={epoch_val_recon:.4f}, kld={epoch_val_kld:.4f}"
            )

    return history, best_state_dict


def show_vae_reconstructions(model, data_loader, device, batch_size=512, title="VAE Reconstructions"):
    """
    Affiche quelques reconstructions du VAE.
    """
    model.eval()
    inputs, _ = next(iter(data_loader))
    inputs = inputs.to(device)

    with torch.no_grad():
        x_recon, mu, logvar = model(inputs)

    inputs = inputs.cpu()
    x_recon = x_recon.cpu()

    plt.figure(figsize=(10, 4))
    plt.suptitle(title)

    for i in range(5):
        idx = 1 + i * (batch_size // 5)

        # originales
        plt.subplot(2, 5, 1 + i)
        plt.imshow(inputs[idx].detach().numpy().squeeze(), cmap='gray')
        plt.axis('off')

        # reconstruites
        plt.subplot(2, 5, 6 + i)
        plt.imshow(x_recon[idx].detach().numpy().squeeze(), cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def generate_samples(model, device, n_samples=16):
    """
    Génère des échantillons en tirant z ~ N(0, I) dans l'espace latent.
    """
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, model.latent_dim).to(device)
        samples = model.decode(z).cpu()

    # affichage
    n_cols = 4
    n_rows = n_samples // n_cols
    plt.figure(figsize=(n_cols * 2, n_rows * 2))
    plt.suptitle("Samples générés par le VAE")

    for i in range(n_samples):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(samples[i].detach().numpy().squeeze(), cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # support de calcul
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Les calculs se font sur', device)

    batch_size = 512
    input_dim = 28
    latent_dim = 16
    beta = 1.0  # poids du terme KL

    # dataloaders
    train_loader, val_loader, test_loader = build_dataloaders_vae(
        batch_size=batch_size,
        root='./'
    )

    # récupération du nombre de canaux
    sample, _ = next(iter(train_loader))
    in_channels = sample.shape[1]

    # architecture CNN similaire à l'AE
    encoder_channels_seq = (16, 32, 64, 32, 16)
    model = VAE(
        input_dim=input_dim,
        in_channels=in_channels,
        encoder_channels_seq=encoder_channels_seq,
        latent_dim=latent_dim
    ).to(device)

    # entraînement
    history, best_state_dict = train_vae(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=20,
        beta=beta
    )

    # chargement du meilleur réseau
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # visualisation des reconstructions
    show_vae_reconstructions(model, val_loader, device, batch_size=batch_size,
                             title="Reconstructions VAE - Validation")

    # génération d'échantillons
    generate_samples(model, device, n_samples=16)