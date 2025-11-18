# autoencodeur_mnist.py

# -*- coding: utf-8 -*-
"""
TP Auto-Encodeur sur MNIST
- Chargement des données
- Entraînement d'un AE convolutionnel (avec ou sans bruit)
- Visualisation des reconstructions

Auteur : Pierre Chambet (d'après support de TP d'Aymeric Chazottes)
"""

import matplotlib
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import ssl
import certifi

def _https_context(*args, **kwargs):
    return ssl.create_default_context(cafile=certifi.where())

ssl._create_default_https_context = _https_context

from src.ae_mnist import AE          # notre modèle
from src.datasets import NoisyMNISTDataset   # dataset bruité


def build_dataloaders(batch_size=512, root='./', denoising=False, train_size=10000, val_size=10000):
    """
    Crée les DataLoaders train / val / test pour MNIST.

    - denoising = False : on renvoie (image_propre, label)
    - denoising = True  : on renvoie (image_bruitee, image_propre)
    """

    if not denoising:
        # Mode "classique" : images propres directement depuis transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.,), (1.,)),
        ])

        dataset = datasets.MNIST(root=root, download=True, train=True, transform=transform)
        train_set, val_set, _ = random_split(dataset, [train_size, val_size, len(dataset) - train_size - val_size])
        test_set = datasets.MNIST(root=root, download=True, train=False, transform=transform)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    else:
        # Mode débruitage : on travaille sur data brutes (0–255) et on Ajoute le bruit nous-mêmes
        base_train = datasets.MNIST(root=root, download=True, train=True, transform=None)
        base_test = datasets.MNIST(root=root, download=True, train=False, transform=None)

        train_subset, val_subset, _ = random_split(
            base_train,
            [train_size, val_size, len(base_train) - train_size - val_size]
        )

        train_set = NoisyMNISTDataset(train_subset, noise_amplitude=0.5)
        val_set = NoisyMNISTDataset(val_subset, noise_amplitude=0.5)
        # Pour le test, on peut aussi ajouter du bruit (même amplitude)
        test_set = NoisyMNISTDataset(base_test, noise_amplitude=0.5)

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_autoencoder(model, train_loader, val_loader, device, num_epochs=20, denoising=False):
    """
    Entraîne un auto-encodeur sur MNIST.

    - Si denoising = False : on apprend à reconstruire l'entrée (x -> x).
    - Si denoising = True  : on apprend à débruiter (x_bruite -> x_propre).
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []
    val_losses = []
    best_state_dict = None
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):

        # === TRAIN ===
        model.train()
        running_train_loss = 0.0

        for batch in train_loader:
            if denoising:
                inputs, targets = batch           # (x_bruite, x_propre)
            else:
                inputs, _ = batch                 # (x, label)
                targets = inputs                  # reconstruction directe

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # === VALIDATION ===
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if denoising:
                    inputs_val, targets_val = batch
                else:
                    inputs_val, _ = batch
                    targets_val = inputs_val

                inputs_val = inputs_val.to(device)
                targets_val = targets_val.to(device)

                outputs_val = model(inputs_val)
                loss_val = criterion(outputs_val, targets_val)
                running_val_loss += loss_val.item() * inputs_val.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        # sauvegarde du meilleur modèle
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_state_dict = model.state_dict()

        if epoch == 1 or epoch % 5 == 0:
            mode_str = "denoising" if denoising else "reconstruction"
            print(f"[{mode_str}] Epoch {epoch:3d} | Train: {epoch_train_loss:.4f} | Val: {epoch_val_loss:.4f}")

    return (train_losses, val_losses), best_state_dict


def show_reconstructions(model, data_loader, device, batch_size=512, title="Reconstructions", denoising=False):
    """
    Affiche quelques images originales / bruitées et reconstruites.

    - Si denoising = False : ligne 1 = entrée, ligne 2 = reconstruction.
    - Si denoising = True  : ligne 1 = entrée bruitée, ligne 2 = reconstruction (censée être propre).
    """
    model.eval()
    batch = next(iter(data_loader))
    if denoising:
        inputs, targets = batch
    else:
        inputs, _ = batch
        targets = inputs

    inputs = inputs.to(device)

    with torch.no_grad():
        outputs = model(inputs)

    inputs = inputs.cpu()
    targets = targets.cpu()
    outputs = outputs.cpu()

    plt.figure(figsize=(10, 6))
    plt.suptitle(title)

    for i in range(5):
        idx = 1 + i * (batch_size // 5)

        # entrée (propre ou bruitée selon le mode)
        plt.subplot(3, 5, 1 + i)
        plt.imshow(inputs[idx].detach().numpy().squeeze(), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title("Input")

        # cible "idéale" (image propre)
        plt.subplot(3, 5, 6 + i)
        plt.imshow(targets[idx].detach().numpy().squeeze(), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title("Target")

        # reconstruction
        plt.subplot(3, 5, 11 + i)
        plt.imshow(outputs[idx].detach().numpy().squeeze(), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title("Reconstruction")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # support de calcul
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Les calculs se font sur', device)

    batch_size = 512
    denoising_mode = True   # 🔴 IMPORTANT : ici on active le mode "débruitage"

    train_loader, val_loader, test_loader = build_dataloaders(
        batch_size=batch_size,
        root='./',
        denoising=denoising_mode
    )

    # récupération du nombre de canaux
    if denoising_mode:
        sample, _ = next(iter(train_loader))   # (x_bruite, x_propre)
    else:
        sample, _ = next(iter(train_loader))   # (x, label)

    input_channels = sample.shape[1]

    # architecture un peu plus légère pour aller vite
    encoder_channels_seq = (16, 32, 64, 32, 16)
    model = AE(in_channels=input_channels, encoder_channels_seq=encoder_channels_seq)
    model.to(device)

    # entraînement
    (train_losses, val_losses), best_state_dict = train_autoencoder(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=20,
        denoising=denoising_mode
    )

    # chargement du meilleur réseau
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # visualisation sur validation
    show_reconstructions(
        model,
        val_loader,
        device,
        batch_size=batch_size,
        title="Denoising AE - Validation",
        denoising=denoising_mode
    )

    # visualisation sur test
    show_reconstructions(
        model,
        test_loader,
        device,
        batch_size=batch_size,
        title="Denoising AE - Test",
        denoising=denoising_mode
    )