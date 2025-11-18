# src/datasets.py

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets


class NoisyMNISTDataset(Dataset):
    """
    Dataset pour MNIST bruité :
    - input : image bruitée
    - target : image propre

    Le dataset d'origine doit être un MNIST ou un Subset[MNIST].
    """

    def __init__(self, dataset, noise_amplitude=0.5):
        super().__init__()

        # On récupère les images brutes (0–255) et les labels.
        if isinstance(dataset, datasets.MNIST):
            targets = dataset.data          # (N, 28, 28)
            labels = dataset.targets
        elif isinstance(dataset, Subset) and isinstance(dataset.dataset, datasets.MNIST):
            # Subset sur un MNIST : on restreint data/targets aux indices du subset
            base_data = dataset.dataset.data
            base_targets = dataset.dataset.targets
            indices = dataset.indices
            targets = base_data[indices]
            labels = base_targets[indices]
        else:
            raise TypeError(
                f"Type inattendu pour dataset : {type(dataset)}. "
                "Attendu : torchvision.datasets.MNIST ou torch.utils.data.Subset[MNIST]."
            )

        # Normalisation en [0, 1]
        self.targets = targets.float() / 255.0   # images propres
        self.labels = labels

        # Génération du bruit
        self.noise_amplitude = noise_amplitude
        noise = torch.rand_like(self.targets) - 0.5  # U(-0.5, 0.5)
        self.inputs = self.targets + noise_amplitude * noise

        # On s'assure de rester dans [0, 1]
        self.inputs.clamp_(0.0, 1.0)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        """
        Retourne (input_bruite, target_propre) en format (1, H, W)
        """
        x_noisy = self.inputs[idx].unsqueeze(0)    # (1, 28, 28)
        x_clean = self.targets[idx].unsqueeze(0)   # (1, 28, 28)
        return x_noisy, x_clean