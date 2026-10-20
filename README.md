# Int-WINN-Chaos: Privacy-Preserving Image Transmission Pipeline

A high-security, bit-exact invertible image encryption pipeline combining:
- **Integer Wavelet Transform (IWT)** — lossless subband decomposition
- **3D Sine-Tent-Cosine Hyperchaotic Map** — high-entropy keystream generation
- **Invertible Neural Network Coupling Blocks** — integer-domain neural diffusion
- **RSA-OAEP Asymmetric Key Envelopes** — secure key transmission

Achieves **PSNR = ∞ dB**, **SSIM = 1.0000**, and **OCR CER = 0.0%** under clean transmission.

---

## Project Structure

```
Image E&D/
├── core/
│   ├── base.py          # Abstract BaseImageCipher interface
│   ├── chaos.py         # 3D-STC Hyperchaotic Generator
│   ├── envelope.py      # RSA-OAEP key envelope manager
│   ├── inn_layers.py    # Integer Coupling Blocks (INN)
│   └── iwt.py           # Integer Lifting Wavelet Transform
├── pipelines/
│   ├── int_winn_chaos.py  # Proposed SOTA pipeline
│   └── baseline_aes.py    # AES-256-CBC baseline
├── evaluation/
│   ├── metrics.py         # Entropy, NPCR, UACI, PSNR, SSIM
│   ├── ocr_eval.py        # OCR CER evaluation (Tesseract)
│   ├── robustness.py      # Salt & Pepper and Occlusion attacks
│   └── report_generator.py  # Markdown report generation
├── datasets/              # Local datasets (NOT committed to git)
│   ├── usc_sipi/          # USC-SIPI natural image benchmark set
│   └── sroie/             # SROIE document receipt dataset
├── demo.py                # Visual 2×3 demo with metrics
├── run_benchmarks.py      # Full evaluation suite
├── requirements.txt
└── IMPLEMENTATION_PLAN.md
```

---

## Setup

### Prerequisites
- Python 3.10+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed and on PATH

### Install dependencies

```bash
pip install -r requirements.txt
```

### Dataset Setup
Place your benchmark images in the following directories:
- `datasets/usc_sipi/` — `.png` natural images (USC-SIPI Miscellaneous dataset)
- `datasets/sroie/` — `.jpg` receipt images + `.txt` ground-truth files (SROIE task 1)

---

## Running

### Visual Demo (generates `demo_results.png`)
```bash
python demo.py
```

### Full Benchmark Suite (generates `BENCHMARK_REPORT.md`)
```bash
python run_benchmarks.py
```

---

## Benchmark Results Summary

| Metric | Baseline AES-256 | Proposed Int-WINN-Chaos |
|:---|:---:|:---:|
| Shannon Entropy | 7.9972 | **7.9998** |
| Pixel Correlation (H) | 0.0018 | **−0.0004** |
| NPCR (%) | 99.61% | **99.64%** |
| UACI (%) | 33.45% | **33.48%** |
| Clean PSNR | ∞ dB | **∞ dB (Bit-Exact)** |
| Clean SSIM | 1.0000 | **1.0000** |
| OCR CER (Encrypted) | 100.0% | **100.0%** |
| OCR CER (Decrypted) | 0.0% | **0.0%** |
| PSNR @ 5% S&P Noise | ~11 dB | **~30 dB** |
| Key Space | 2^256 | **> 2^318** |

---

## Collaborating

```bash
# Clone the repo
git clone https://github.com/anaysingh424/image-ed.git
cd image-ed

# Create and work on your feature branch
git checkout -b feature/your-feature-name

# Push your branch
git push origin feature/your-feature-name

# Open a Pull Request on GitHub
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
