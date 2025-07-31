# Effective VAEs

This repository provides a minimal yet powerful implementation of **Variational Autoencoders (VAEs)** using PyTorch, applied to the MNIST dataset. The focus is on clarity, modularity, and experimentation with core VAE techniques, including amortized inference and latent optimization.

## Overview

Variational Autoencoders (VAEs) are generative models that learn to encode data into a continuous latent space and decode from this space back to the data distribution. This repository explores and implements two main paradigms for variational inference:

- **Amortized Inference**: using a neural encoder to predict variational parameters for each input
- **Latent Optimization**: directly optimizing a variational distribution per datapoint

Both approaches are evaluated and visualized through reconstruction quality, sample generation, and log-likelihood estimation.

## Features

- Modular VAE architecture built in PyTorch
- Latent space sampling and reparameterization with numerical stability techniques
- Support for both amortized inference and latent optimization
- Stable ELBO training with MSE + KL divergence loss
- Visualization of training dynamics, including:
  - Reconstructions over training epochs
  - Prior sampling from the decoder
  - Evolution of generated samples
- Log-likelihood estimation via importance sampling (Monte Carlo)
- Clean training loops with separation between encoder, decoder, and optimization logic

## Structure

```
.
├── models/                 # Encoder and decoder architectures
├── train_amortized.py     # Training loop for amortized VAE
├── train_latent_opt.py    # Training loop for latent optimization VAE
├── utils/                 # Utilities: sampling, loss, evaluation, plotting
├── evaluate_loglik.py     # Script for computing log-likelihood estimates
├── config.py              # Configuration and hyperparameters
└── README.md
```

## Experiments Implemented

- Amortized VAE training with latent dimension 200, σₚ = 0.4  
  Visualizes reconstructions at epochs 1, 5, 10, 20, 30
- Latent Optimization VAE, trained by directly learning latent means/variances  
  Includes comparison with amortized method on reconstruction and sample quality
- Log-likelihood estimation using log-sum-exp trick with 1000 samples from q(z|x)

## Requirements

- Python ≥ 3.8  
- PyTorch ≥ 1.10  
- torchvision  
- matplotlib  
- numpy  
- tqdm

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Getting Started

1. **Amortized VAE**

```bash
python train_amortized.py
```

2. **Latent Optimization VAE**

```bash
python train_latent_opt.py
```

3. **Log-Likelihood Evaluation**

```bash
python evaluate_loglik.py
```

All output plots are saved automatically under the `results/` directory.

## Notes

- The MNIST training subset includes 20,000 images (2,000 per digit).
- Encoder outputs log-variance (`log_sigma²`) for stability.
- KL-divergence is computed analytically assuming diagonal Gaussians.

## Citation

The implementation and methodology follow:

- Kingma & Welling (2013), *Auto-Encoding Variational Bayes*  
- Hoffman et al. (2013), *Stochastic Variational Inference*

