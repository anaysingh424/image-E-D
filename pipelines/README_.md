# CNN-Based Image Compression and Encryption

## Overview
This project compresses, encrypts, and decrypts grayscale images using a CNN Autoencoder, Henon and Ikeda chaotic maps, permutation-diffusion, and AES-256-GCM.

## Main Features
- Converts input images to grayscale and resizes them to 256 × 256.
- Compresses images into a 128 × 128 latent representation using a CNN Autoencoder.
- Generates chaotic sequences using Henon and Ikeda maps.
- Applies permutation and XOR-based diffusion.
- Uses AES-256-GCM for authenticated encryption.
- Reconstructs images after decryption and calculates evaluation metrics.

## Requirements
- Python 3.11 or 3.12
- Packages listed in `requirements.txt`

## Installation
```bash
pip install -r requirements.txt
```

## Run
Place your images in the `test_images` folder, then run:

```bash
python cae_chaos_crypto.py
```

You can also specify an image folder and training settings:

```bash
python cae_chaos_crypto.py test_images --epochs=3000 --batch-size=16
```

## Outputs
The script creates:
- `models_original/` – saved autoencoder model.
- `results_original/` – reconstructed images, error maps, loss curve, and evaluation metrics.

## Evaluation
The project reports PSNR, SSIM, MSE, MAE, NPCR, UACI, entropy, pixel correlation, and execution time.

## Note
This is a research and educational implementation. Its evaluation metrics do not, by themselves, prove cryptographic security.
