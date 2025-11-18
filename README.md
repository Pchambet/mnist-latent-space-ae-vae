<p align="center">
  <h1>Denoising Autoencoder & Variational Autoencoder</h1>
  <p><em>MNIST • Latent Spaces • Generative Modeling</em></p>

  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Deep%20Learning-000000?style=for-the-badge&logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/MNIST-Digit%20Dataset-orange?style=for-the-badge" />
</p>

---

This project explores two fundamental deep learning architectures used to model and understand latent spaces:

- A **Denoising Convolutional Autoencoder (AE)**, trained to reconstruct clean MNIST digits from heavily corrupted inputs.
- A **Variational Autoencoder (VAE)**, trained to learn a smooth probabilistic latent space and generate new digit images.

The repository includes:
- modular and well-structured PyTorch code,
- full training pipelines for both AE and VAE,
- high-quality visual results (reconstructions + generated samples),
- and a detailed LaTeX report (in French) covering the theory and interpretation.

This project is designed to be both **educational** and **professional**:  
readers can understand how noise shapes latent representations, how VAEs handle uncertainty, and how both models behave on a real dataset.

Below is an example of the denoising autoencoder in action.  
The model receives a heavily corrupted MNIST digit (top row),  
is trained to recover the clean target (middle row),  
and produces a clean reconstruction (bottom row).

<p align="center">
  <img src="figures/ae_denoising_val.jpg" width="80%" />
</p>

## Project Structure

The repository follows a clear, modular, and professional organization:

```.
./
├── README.md                    ← Project overview and instructions
├── requirements.txt             ← Python dependencies  
├── autoencodeur_mnist.py        ← Full training script for the denoising AE
├── vae_mnist.py                 ← Full training + sampling script for the VAE
├── src/
│   ├── ae_mnist.py              ← Encoder, Decoder, and Autoencoder classes
│   ├── vae_mnist.py             ← VAE class, KL divergence, reparameterization
│   ├── datasets.py              ← Custom MNIST dataset with injected noise
│   └── utils.py                 ← Training loops, plotting helpers, metrics
├── figures/                     ← All generated images (AE/VAE outputs)
│   ├── ae_denoising_val.png
│   ├── ae_denoising_test.png
│   ├── vae_reconstructions.png
│   └── vae_generated_samples.png
└── report/
    └── TP_AE_VAE_Chambet.pdf    ← Complete theoretical + experimental report
```

Each component is isolated to make the code easy to read, modify, and reuse.
Training scripts call the models located in /src/, and store visual outputs in /figures/.

## Installation

This project uses Python 3 and PyTorch.  
You can run everything inside a virtual environment.

### 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate    # (On Windows: .venv\Scripts\activate)

### 2. Install dependencies
pip install -r requirements.txt

## Usage

### Train the Denoising Autoencoder
python3 autoencodeur_mnist.py

### Train the Variational Autoencoder + generate samples
python3 vae_mnist.py

All generated figures (AE reconstructions, VAE reconstructions, VAE samples)  
are automatically saved inside the /figures/ directory.

## Denoising Autoencoder (AE)

The denoising autoencoder is trained to recover a clean MNIST digit from a heavily corrupted version of the same image.
Instead of learning to copy pixels, the model must learn the *structural essence* of each digit (strokes, contours, shapes).

### Objective
noisy_input  →  clean_output

### What the AE learns
- A robust latent representation that ignores high-frequency noise.
- A nonlinear denoising function that preserves digit identity.
- A smooth reconstruction space where noise is naturally filtered out.

### Results
The model removes most of the injected noise and outputs clean, stable digits.
Validation and test losses remain low (≈ 0.02), confirming good generalization.

See:
- figures/ae_denoising_val.png
- figures/ae_denoising_test.png

## Variational Autoencoder (VAE)

The VAE does not encode an image into a single latent vector.
Instead, it learns a *distribution* for each input:

    q(z | x) = N( μ(x), σ²(x) )

A latent sample is obtained through the reparameterization trick:

    z = μ + σ · ε       with ε ~ N(0, I)

This design forces the latent space to be smooth, continuous, and generative.

### What this implies
- Reconstructions are naturally **blurrier** than in a classic AE.
- The model learns plausible *average digits*, not pixel-perfect copies.
- The KL divergence regularizes the latent space and prevents memorization.

### Results
- Reconstructions: recognizable but visibly smoothed.
- Generated samples: similar “mean-digit” shapes (expected from a basic VAE with strong KL + small latent dimension).

See:
- figures/vae_reconstructions.png
- figures/vae_generated_samples.png

## Report Overview

The full report (written in French) provides a complete theoretical and experimental analysis of the project.

It covers:
- the structure and role of latent spaces,
- the difference between denoising AEs and VAEs,
- the mathematical foundations of the VAE (KL divergence, latent sampling),
- the reparameterization trick and why it is necessary,
- the effect of noise in the input vs. noise in the latent space,
- reconstruction behavior across training sets,
- iterative reconstructions (successive applications of the model),
- and detailed answers to all assignment questions.

You can find the report here:
report/TP_AE_VAE_Chambet.pdf

## Educational & Professional Purpose

This project was designed to be both a learning tool and a professional showcase.

### Educational value
- Understand how neural networks learn internal representations.
- Explore denoising, latent spaces, KL regularization, and generative modeling.
- Visualize how different architectures behave on real data.

### Professional value
- Clean, modular, and well-structured PyTorch code.
- Clear training pipelines and reproducible experiments.
- High-quality figures and a rigorous written report.
- Demonstrates practical skills in:
  - convolutional networks,
  - autoencoders,
  - VAEs and probabilistic modeling,
  - PyTorch engineering and dataset handling,
  - interpretation of deep learning models.

This repository is meant to be readable, reusable, and valuable for both students and professionals.

## Author

**Pierre Chambet**  
Télécom SudParis — Institut Polytechnique de Paris  
Master’s student in Data Science & Applied Mathematics

Feel free to reach out for collaboration, discussion, or feedback.
