# Quantum Image Encryption Lab

An educational Streamlit project demonstrating a **hybrid quantum-circuit-simulation + classical image transformation** workflow.

## Features
- Upload PNG/JPG/BMP images
- Grayscale or RGB processing
- Parameterized 8-qubit circuit simulated with Qiskit Aer (when installed)
- Deterministic demo keystream for repeatable decryption
- Pixel permutation and XOR transformation
- Exact-recovery check and SHA-256 comparison
- NumPy fallback if Qiskit Aer is unavailable

## Important scientific/security limitation
This is a teaching prototype, **not production cryptography**. The permutation and XOR are classical operations; a quantum circuit is simulated to produce a demo bitstream. The design has not been cryptographically analyzed, uses a password-derived seed, and does not provide authenticated encryption. Do not use it to protect private images. For real applications use a vetted authenticated encryption scheme such as AES-GCM with sound key management.

## Run locally
Requires Python 3.10+.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501).

## Suggested experiment
1. Try a small grayscale image first.
2. Encrypt and decrypt with the same passphrase; confirm exact recovery.
3. Change the passphrase and confirm recovery no longer matches.
4. Compare original and encrypted pixel histograms and neighboring-pixel correlations.
5. Record runtime at several image sizes. Do not interpret these tests alone as proof of security.

## Project title
**Quantum-Circuit-Assisted Image Encryption and Decryption: A Hybrid Simulation Prototype**
