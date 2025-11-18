# Denoising Autoencoder & Variational Autoencoder on MNIST
Understanding latent spaces, denoising, and generative modeling

This repository contains a complete study of Convolutional Autoencoders (AE) and Variational Autoencoders (VAE) applied to the MNIST handwritten digits dataset.
The goal is to explore:

1. How an autoencoder can remove noise injected directly into the input images.
2. How a VAE injects noise inside the latent space using the reparameterization trick.

The project includes modular Python code, training scripts, generated figures, and a full LaTeX report answering all theoretical questions.

---------------------------------------------------------------------

PROJECT OVERVIEW

Denoising Autoencoder (AE)
- Takes a noisy MNIST digit and reconstructs a clean version.
- Learns robust latent features and acts as a nonlinear denoising filter.

Variational Autoencoder (VAE)
- Encodes images as Gaussian distributions q(z|x) = N(mu(x), sigma^2(x)).
- Samples latent vectors using z = mu + sigma * epsilon where epsilon ~ N(0,I).
- Produces blurry reconstructions and average-like generated digits, as expected from a basic VAE.

Everything is implemented in PyTorch with a clean modular structure.

---------------------------------------------------------------------

PROJECT STRUCTURE

.
├── autoencodeur_mnist.py        (AE training script)
├── vae_mnist.py                 (VAE training and sampling script)
├── src/
│   ├── ae_mnist.py              (Encoder, Decoder, AE classes)
│   ├── vae_mnist.py             (VAE class, KL, reparameterization)
│   ├── datasets.py              (Noisy MNIST dataset)
│   └── utils.py                 (Training loops and plotting tools)
├── figures/
│   ├── ae_denoising_val.png
│   ├── ae_denoising_test.png
│   ├── vae_reconstructions.png
│   └── vae_generated_samples.png
└── report/
    └── TP_AE_VAE_Chambet.pdf    (Full LaTeX report)

---------------------------------------------------------------------

1. DENOISING AUTOENCODER (AE)

Objective:
Learn: noisy_input → clean_target
The AE must learn digit structure instead of copying pixels.

Results:
- Strong noise removal
- Smooth and clean reconstructions
- Low train/validation MSE (~0.02)
- Latent space becomes noise-invariant

See the figures:
- ae_denoising_val.png
- ae_denoising_test.png

---------------------------------------------------------------------

2. VARIATIONAL AUTOENCODER (VAE)

Objective:
Learn a probabilistic latent space.

Why reconstructions are blurry:
z = mu + sigma * epsilon includes random noise.
The model reconstructs average plausible digits, not exact images.

Why generated samples look similar:
- Small latent dimension (16)
- Strong KL regularization
- Only 20 epochs of training
- High compression

→ The model collapses to a typical MNIST mode (common VAE behavior).

See:
- vae_reconstructions.png
- vae_generated_samples.png

---------------------------------------------------------------------

3. REPORT (LATEX)

The PDF report includes:
- AE vs VAE explanation
- Effect of noise in inputs vs latent space
- Reparameterization trick
- KL divergence
- Iterative reconstructions
- Answers to all theoretical questions

---------------------------------------------------------------------

4. INSTALLATION

Create virtual environment:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Run the AE:
python3 autoencodeur_mnist.py

Run the VAE:
python3 vae_mnist.py

---------------------------------------------------------------------

EDUCATIONAL PURPOSE

This project is designed to be:
- Academic (clear theory)
- Pedagogical (figures and explanations)
- Professional (clean modular PyTorch code)

It demonstrates skills in:
- CNNs
- Image denoising
- Generative modeling
- Latent space structure
- PyTorch engineering

---------------------------------------------------------------------

AUTHOR

Pierre Chambet
Télécom SudParis / Institut Polytechnique de Paris
Master’s student in Data Science and Applied Mathematics

---------------------------------------------------------------------
