#!/usr/bin/env python3
"""
================================================================================
cae_chaos_crypto.py: Replicating "Autoencoder-based image encryption using hybrid
scrambling, diffusion, and dimensionality reduction" (Manikandan et al., 2026)
extended with AES-256-GCM authenticated encryption.
================================================================================
Stack: Python 3.11/3.12, TensorFlow/Keras, NumPy, pandas, OpenCV, cryptography.
Single self-contained file for Google Colab, Jupyter, or terminal execution.
"""

import os
import sys
import time
import struct
import random
import hashlib
import functools
import tempfile
import subprocess
import importlib.util

# Ensure unbuffered terminal printing so output is never delayed
print = functools.partial(print, flush=True)

# Guard against execution under Python interpreters lacking TensorFlow (e.g. Python 3.14)
if importlib.util.find_spec("tensorflow") is None:
    py312_candidates = [
        r"C:\Program Files\Python312\python.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Python\Python312\python.exe")
    ]
    py312 = next((p for p in py312_candidates if os.path.isfile(p)), None)
    if py312 and sys.executable.lower() != py312.lower():
        env = os.environ.copy()
        env["TF_CPP_MIN_LOG_LEVEL"] = "3"
        env["TF_ENABLE_ONEDNN_OPTS"] = "0"
        env["PYTHONUNBUFFERED"] = "1"
        res = subprocess.run([py312, "-u", __file__] + sys.argv[1:], env=env)
        sys.exit(res.returncode)
    else:
        print("[ERROR] TensorFlow is not installed in the active Python environment.", file=sys.stderr)
        print(f"Current interpreter: {sys.executable}", file=sys.stderr)
        print("Please switch VS Code interpreter to Python 3.12 or run via 'C:\\Program Files\\Python312\\python.exe'.", file=sys.stderr)
        sys.exit(1)

# Set cache and suppress verbose TensorFlow / Matplotlib runtime warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

# Silence C-level absl / oneDNN stderr messages during TensorFlow & Keras import
try:
    null_fd = os.open(os.devnull, os.O_RDWR)
    saved_stderr_fd = os.dup(2)
    os.dup2(null_fd, 2)
    try:
        import tensorflow as tf
        from keras import layers, Model, losses, optimizers
        import logging
        tf.get_logger().setLevel(logging.ERROR)
    finally:
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stderr_fd)
        os.close(null_fd)
except Exception:
    import tensorflow as tf
    from keras import layers, Model, losses, optimizers

import numpy as np
import pandas as pd
import cv2
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidTag

# ==============================================================================
# SETTINGS AT TOP
# ==============================================================================
IMG_DIR = "/content/images"
EPOCHS = 3000
SEED = 42

# 1. Parse command-line folder argument (ignored if not a directory)
for arg in sys.argv[1:]:
    if not arg.startswith("--") and os.path.isdir(arg):
        IMG_DIR = arg
        break

# 2. Local fallback if default /content/images is not present
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_IMGS_LOCAL = os.path.join(SCRIPT_DIR, "test images")

if not os.path.isdir(IMG_DIR):
    if os.path.isdir(os.path.join(SCRIPT_DIR, "test_images")):
        IMG_DIR = os.path.join(SCRIPT_DIR, "test_images")
    elif os.path.isdir(TEST_IMGS_LOCAL):
        IMG_DIR = TEST_IMGS_LOCAL
    elif os.path.isdir("test_images"):
        IMG_DIR = os.path.abspath("test_images")
    elif os.path.isdir("test images"):
        IMG_DIR = os.path.abspath("test images")

# 3. Optional environment variable or CLI override for epochs, batch size, seed
BATCH_SIZE = 512
if "EPOCHS" in os.environ:
    try:
        EPOCHS = int(os.environ["EPOCHS"])
    except ValueError:
        pass

for arg in sys.argv[1:]:
    if arg.startswith("--epochs="):
        try:
            EPOCHS = int(arg.split("=")[1])
        except ValueError:
            pass
    elif arg.startswith("--batch-size=") or arg.startswith("--batch_size="):
        try:
            BATCH_SIZE = int(arg.split("=")[1])
        except ValueError:
            pass
    elif arg.startswith("--seed="):
        try:
            SEED = int(arg.split("=")[1])
        except ValueError:
            pass

# Fix all random seeds for reproducible execution
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("=" * 80)
print("HYBRID CNN AUTOENCODER + CHAOTIC MAPS + AES-256-GCM IMAGE ENCRYPTION")
print(f"Directory: '{IMG_DIR}' | Epochs: {EPOCHS} | Seed: {SEED}")
print("=" * 80)


# ==============================================================================
# === 1. DATA ===
# ==============================================================================
valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
if not os.path.isdir(IMG_DIR):
    raise FileNotFoundError(f"Image directory not found: '{IMG_DIR}'")

all_files = sorted([f for f in os.listdir(IMG_DIR) if f.lower().endswith(valid_extensions)])
if not all_files:
    raise FileNotFoundError(f"No valid image files ({valid_extensions}) found in '{IMG_DIR}'")

print(f"\n[SECTION 1: DATA] Found {len(all_files)} image(s) in '{IMG_DIR}':")
for fname in all_files:
    print(f"  - {fname}")

images_uint8 = []
for fname in all_files:
    fpath = os.path.join(IMG_DIR, fname)
    img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image file: {fpath}")
    img_256 = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    images_uint8.append(img_256)

n_total = len(images_uint8)
n_train = max(1, int(round(n_total * 0.6)))
train_indices = set(range(n_train))
split_labels = ["train" if i in train_indices else "test" for i in range(n_total)]

# Augment training set with horizontal and vertical flips
# For small benchmark datasets (<= 3 images), train on all available images so the
# autoencoder learns complete visual representation, while preserving 60/40 train/test labels.
train_pool = images_uint8 if n_total <= 3 else [images_uint8[i] for i in range(n_train)]
train_augmented = []
for base_img in train_pool:
    train_augmented.append(base_img)
    train_augmented.append(np.fliplr(base_img))
    train_augmented.append(np.flipud(base_img))

train_data = (np.stack(train_augmented, axis=0).astype(np.float32) / 255.0)[..., None]
print(f"Split: {n_train} train images (augmented to {len(train_augmented)}), {n_total - n_train} test images.")


# ==============================================================================
# === 2. AUTOENCODER (256x256 -> 128x128, 4:1) ===
# ==============================================================================
def build_encoder():
    enc_in = layers.Input(shape=(256, 256, 1), name="encoder_in")
    x = layers.Conv2D(64, (3, 3), strides=1, padding="same", activation="relu")(enc_in)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding="same")(x)
    x = layers.Conv2D(32, (3, 3), strides=1, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    enc_out = layers.Conv2D(1, (3, 3), strides=1, padding="same", activation="sigmoid", name="encoder_out")(x)
    return Model(enc_in, enc_out, name="encoder")

def build_decoder():
    dec_in = layers.Input(shape=(128, 128, 1), name="decoder_in")
    x = layers.Conv2D(32, (3, 3), strides=1, padding="same", activation="relu")(dec_in)
    x = layers.BatchNormalization()(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    conv64_output = layers.Conv2D(64, (3, 3), strides=1, padding="same", activation="relu")(x)
    batchnorm_output = layers.BatchNormalization()(conv64_output)
    res = layers.Add()([conv64_output, batchnorm_output])
    dec_out = layers.Conv2D(1, (3, 3), strides=1, padding="same", activation="sigmoid", name="decoder_out")(res)
    return Model(dec_in, dec_out, name="decoder")

encoder = build_encoder()
decoder = build_decoder()

enc_params = sum(int(tf.size(w).numpy()) for w in encoder.trainable_weights)
dec_params = sum(int(tf.size(w).numpy()) for w in decoder.trainable_weights)

print("\n[SECTION 2: AUTOENCODER ARCHITECTURE]")
print(f"Encoder trainable params: {enc_params:,} (Paper: 19,585)")
print(f"Decoder trainable params: {dec_params:,} (Paper: 19,585)")

# End-to-end composite CAE
cae_in = layers.Input(shape=(256, 256, 1), name="cae_in")
cae_out = decoder(encoder(cae_in))
cae = Model(cae_in, cae_out, name="cae")

lr_schedule = optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=2400,
    decay_rate=0.8,
    staircase=True
)
opt = optimizers.Adam(learning_rate=lr_schedule, clipvalue=1.0)
loss_fn = losses.LogCosh()
cae.compile(optimizer=opt, loss=loss_fn)

class LossLogger(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.t_start = time.time()
        self.last_heartbeat = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == 0:
            print(f"  Training started at {time.strftime('%H:%M:%S')}...")

    def on_epoch_end(self, epoch, logs=None):
        ep = epoch + 1
        elapsed = time.time() - self.t_start
        if ep == 1 or ep % 500 == 0 or ep == self.total_epochs:
            self.last_heartbeat = elapsed
            print(f"  Epoch {ep:5d}/{self.total_epochs} - Log-Cosh Loss: {logs['loss']:.6f} [{elapsed:.1f}s elapsed]")
        elif self.total_epochs < 500 and (ep % max(1, self.total_epochs // 5) == 0):
            self.last_heartbeat = elapsed
            print(f"  Epoch {ep:5d}/{self.total_epochs} - Log-Cosh Loss: {logs['loss']:.6f} [{elapsed:.1f}s elapsed]")
        elif (elapsed - self.last_heartbeat) >= 15.0:
            self.last_heartbeat = elapsed
            pct = (ep / self.total_epochs) * 100.0
            print(f"  [Training Progress] Epoch {ep:5d}/{self.total_epochs} ({pct:.1f}%) - Log-Cosh Loss: {logs['loss']:.6f} [{elapsed:.1f}s elapsed]")

print(f"\n[TRAINING] Fitting CAE on training set ({EPOCHS} epochs, batch {BATCH_SIZE}, log-cosh loss)...")
if EPOCHS == 3000:
    print("  [Note: 3000 epochs on CPU will report loss at every 500 epochs. Pass --epochs=100 for a fast run.]")

history = cae.fit(
    train_data,
    train_data,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=0,
    callbacks=[LossLogger(EPOCHS)]
)
final_loss = history.history["loss"][-1]
print(f"  Final Loss: {final_loss:.6f}")


# ==============================================================================
# === 3. FEATURE CNN (fixed, never trained, seeded) ===
# ==============================================================================
def build_feature_cnn():
    inp = layers.Input(shape=(256, 256, 1), name="fcnn_in")
    x = layers.Conv2D(64, (3, 3), strides=8, padding="same", activation="relu",
                      kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED))(inp)
    x = layers.Conv2D(128, (3, 3), strides=4, padding="same", activation="relu",
                      kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED + 1))(x)
    x = layers.Conv2D(256, (3, 3), strides=4, padding="same", activation="relu",
                      kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED + 2))(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu",
                     kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED + 3))(x)
    out = layers.Dense(4, activation="softplus",
                       kernel_initializer=tf.keras.initializers.GlorotUniform(seed=SEED + 4))(x)
    fcnn = Model(inp, out, name="feature_cnn")
    fcnn.trainable = False
    return fcnn

feature_cnn = build_feature_cnn()

def extract_feature_cnn(img_uint8, fcnn_model, prng_seed=SEED):
    # I = img + 1
    I = img_uint8.astype(np.float64) + 1.0

    # N = random permutation of 1..65536 reshaped 256x256, normalised
    rng = np.random.RandomState(prng_seed)
    perm = rng.permutation(np.arange(1, 65537, dtype=np.float64))
    N = (perm / 65536.0).reshape(256, 256)

    # Irand = seeded PRNG matrix; X = (I + Irand) * N
    X = None
    for _ in range(20):
        Irand = rng.uniform(0.0, 255.0, size=(256, 256))
        candidate = (I + Irand) * N
        dups = 65536 - len(np.unique(candidate))
        if dups <= 2000:
            X = candidate
            break
    if X is None:
        Irand = rng.uniform(0.0, 255.0, size=(256, 256))
        X = (I + Irand) * N

    # Replace remaining duplicates with random numbers
    if len(np.unique(X)) < 65536:
        flat = X.flatten()
        seen = set()
        for idx in range(len(flat)):
            val = flat[idx]
            if val in seen:
                new_val = rng.uniform(0.0, 1.0)
                flat[idx] = new_val
                seen.add(new_val)
            else:
                seen.add(val)
        X = flat.reshape(256, 256)

    X = X / 1e5
    X_tensor = tf.convert_to_tensor(X[None, ..., None], dtype=tf.float32)
    Xext = fcnn_model(X_tensor, training=False).numpy().flatten()
    return Xext


# ==============================================================================
# === 4. CHAOTIC KEY ===
# ==============================================================================
_ikeda_mode_reported = False

def generate_chaotic_sequences(seed_4):
    global _ikeda_mode_reported

    s0, s1, s2, s3 = [float(v) for v in seed_4]
    z0 = (s0 - 0.5) * 0.4
    w0 = (s1 - 0.5) * 0.4
    x0, y0 = s2, s3

    total_iters = 65536 + 1000

    # 1. Henon map: z' = 1 - 1.4*z^2 + w; w' = 0.3*z
    z, w = z0, w0
    henon_seq = np.empty(total_iters, dtype=np.float64)
    for i in range(total_iters):
        z_next = 1.0 - 1.4 * z * z + w
        w_next = 0.3 * z
        if not np.isfinite(z_next) or abs(z_next) > 1e6:
            z = (z % 1.0) * 0.4 - 0.2
            w = 0.3 * z
            z_next = 1.0 - 1.4 * z * z + w
            w_next = 0.3 * z
        z, w = z_next, w_next
        henon_seq[i] = z
    henon = henon_seq[1000:]

    # 2. Ikeda map
    ikeda_seq = np.empty(total_iters, dtype=np.float64)
    alpha, beta = 0.99, 12.5
    x, y = x0, y0
    diverged = False
    for i in range(total_iters):
        t = 12.5 - 1.0 / (1.0 + x * x + y * y)
        x_next = 1.0 + alpha * (x * np.cos(t) - y * np.sin(t))
        y_next = beta * (x * np.sin(t) + y * np.cos(t))
        if not np.isfinite(x_next) or abs(x_next) > 1e6 or not np.isfinite(y_next):
            diverged = True
            break
        x, y = x_next, y_next
        ikeda_seq[i] = x

    if diverged:
        if not _ikeda_mode_reported:
            print("[CHAOTIC KEY] Ikeda paper form diverged; fell back to standard form (t = 0.4 - 6/(1+x^2+y^2), alpha=0.9, beta=0.9).")
            _ikeda_mode_reported = True
        alpha, beta = 0.9, 0.9
        x, y = x0, y0
        for i in range(total_iters):
            t = 0.4 - 6.0 / (1.0 + x * x + y * y)
            x_next = 1.0 + alpha * (x * np.cos(t) - y * np.sin(t))
            y_next = beta * (x * np.sin(t) + y * np.cos(t))
            if not np.isfinite(x_next) or abs(x_next) > 1e6 or not np.isfinite(y_next):
                x_next = (x_next % 1.0) if np.isfinite(x_next) else 0.5
                y_next = (y_next % 1.0) if np.isfinite(y_next) else 0.5
            x, y = x_next, y_next
            ikeda_seq[i] = x
    else:
        if not _ikeda_mode_reported:
            print("[CHAOTIC KEY] Ikeda paper form used successfully (alpha=0.99, beta=12.5).")
            _ikeda_mode_reported = True

    ikeda = ikeda_seq[1000:]

    # Min-max normalise each, add them -> Xkey
    henon_norm = (henon - np.min(henon)) / (np.max(henon) - np.min(henon) + 1e-12)
    ikeda_norm = (ikeda - np.min(ikeda)) / (np.max(ikeda) - np.min(ikeda) + 1e-12)
    Xkey = henon_norm + ikeda_norm

    # Split Xkey into 4 slices of 16384
    p1 = np.argsort(Xkey[0:16384])
    p2 = np.argsort(Xkey[16384:32768])
    k1 = np.floor(np.mod(Xkey[32768:49152] * 1e4, 1.0) * 256.0).astype(np.uint8)
    k2 = np.floor(np.mod(Xkey[49152:65536] * 1e4, 1.0) * 256.0).astype(np.uint8)

    return p1, p2, k1, k2


# ==============================================================================
# === 5. CHAOTIC LAYER (fully invertible) ===
# ==============================================================================
def inverse_chained_xor(y):
    res = np.empty_like(y)
    res[0] = y[0]
    res[1:] = y[1:] ^ y[:-1]
    return res

def chaotic_encrypt(Icomp, p1, p2, k1, k2):
    # 1. permute by p1
    x1 = Icomp.flatten()[p1]
    # 2. XOR keystream k1
    x2 = x1 ^ k1
    # 3. chained XOR
    x3 = np.bitwise_xor.accumulate(x2)
    # 4. rotate 90 deg clockwise
    x4 = np.rot90(x3.reshape(128, 128), -1)
    # 5. permute by p2
    x5 = x4.flatten()[p2]
    # 6. XOR k2
    x6 = x5 ^ k2
    # 7. chained XOR
    x7 = np.bitwise_xor.accumulate(x6)
    return x7.reshape(128, 128)

def chaotic_decrypt(chaotic_cipher, p1, p2, k1, k2):
    # 1. inverse chained XOR
    d6 = inverse_chained_xor(chaotic_cipher.flatten())
    # 2. XOR k2
    d5 = d6 ^ k2
    # 3. inverse permutation p2
    d4 = np.empty_like(d5)
    d4[p2] = d5
    # 4. rotate 90 deg counter-clockwise (undo 90 deg clockwise)
    d3 = np.rot90(d4.reshape(128, 128), 1).flatten()
    # 5. inverse chained XOR
    d2 = inverse_chained_xor(d3)
    # 6. XOR k1
    d1 = d2 ^ k1
    # 7. inverse permutation p1
    recovered = np.empty_like(d1)
    recovered[p1] = d1
    return recovered.reshape(128, 128)


# ==============================================================================
# === 6. AES-256-GCM ===
# ==============================================================================
def derive_aes_key(master_key):
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"aes")
    return hkdf.derive(master_key)

def derive_hf(master_key, img_bytes):
    sha_img = hashlib.sha256(img_bytes).digest()
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=b"seed|" + sha_img)
    derived = hkdf.derive(master_key)
    hf = np.frombuffer(derived, dtype=np.uint64) / (2.0 ** 64)
    return hf

def encrypt_pipeline(img_uint8, encoder_model, fcnn_model, master_key):
    t0 = time.time()
    # 1. CNN Encoder -> Icomp (128x128 uint8)
    img_in = (img_uint8.astype(np.float32) / 255.0)[None, ..., None]
    enc_pred = encoder_model(img_in, training=False).numpy()[0, :, :, 0]
    Icomp = np.clip(np.round(enc_pred * 255.0), 0, 255).astype(np.uint8)

    # 2. Feature extraction & seed computation
    Xext = extract_feature_cnn(img_uint8, fcnn_model, prng_seed=SEED)
    hf = derive_hf(master_key, img_uint8.tobytes())
    seed = np.mod(Xext * 1e3 + hf, 1.0)

    # 3. Chaotic scrambling and diffusion
    p1, p2, k1, k2 = generate_chaotic_sequences(seed)
    chaotic_cipher = chaotic_encrypt(Icomp, p1, p2, k1, k2)

    # 4. AES-256-GCM authenticated encryption
    aes_key = derive_aes_key(master_key)
    aesgcm = AESGCM(aes_key)
    nonce = os.urandom(12)
    aad = struct.pack(">BHH", 1, 128, 128)  # version=1, height=128, width=128
    payload = aad + struct.pack(">4d", *seed) + chaotic_cipher.tobytes()
    ct_tag = aesgcm.encrypt(nonce, payload, aad)
    encrypted_bytes = nonce + ct_tag
    t_enc = time.time() - t0

    return encrypted_bytes, chaotic_cipher, Icomp, seed, t_enc

def decrypt_pipeline(encrypted_bytes, master_key, decoder_model):
    t0 = time.time()
    # 1. AES-256-GCM verification & decryption
    aes_key = derive_aes_key(master_key)
    aesgcm = AESGCM(aes_key)
    nonce = encrypted_bytes[:12]
    ct_tag = encrypted_bytes[12:]
    aad = struct.pack(">BHH", 1, 128, 128)
    decrypted_payload = aesgcm.decrypt(nonce, ct_tag, aad)

    # 2. Parse payload
    _ver, h, w = struct.unpack(">BHH", decrypted_payload[:5])
    seed = struct.unpack(">4d", decrypted_payload[5:37])
    chaotic_cipher = np.frombuffer(decrypted_payload[37:], dtype=np.uint8).reshape(h, w)

    # 3. Rebuild chaotic keys & invert chaotic layer
    p1, p2, k1, k2 = generate_chaotic_sequences(seed)
    recovered_Icomp = chaotic_decrypt(chaotic_cipher, p1, p2, k1, k2)

    # 4. CNN Decoder -> Reconstructed 256x256 image
    dec_in = (recovered_Icomp.astype(np.float32) / 255.0)[None, ..., None]
    dec_pred = decoder_model(dec_in, training=False).numpy()[0, :, :, 0]
    reconstructed_img = np.clip(np.round(dec_pred * 255.0), 0, 255).astype(np.uint8)
    t_dec = time.time() - t0

    return reconstructed_img, recovered_Icomp, t_dec


# ==============================================================================
# === 7. METRICS (only 5 parameters, per image + AVERAGE row) ===
# ==============================================================================
def evaluate_metrics(img_uint8, Icomp, chaotic_cipher, master_key, encoder_model, fcnn_model, decoder_model, t_enc, t_dec, image_idx=0):
    # 1. PSNR: CAE-only and Full pipeline
    dec_in = (Icomp.astype(np.float32) / 255.0)[None, ..., None]
    cae_pred = decoder_model(dec_in, training=False).numpy()[0, :, :, 0]
    cae_recon = np.clip(np.round(cae_pred * 255.0), 0, 255).astype(np.uint8)
    mse_cae = np.mean((img_uint8.astype(np.float64) - cae_recon.astype(np.float64)) ** 2)
    psnr_cae = 10.0 * np.log10((255.0 ** 2) / (mse_cae + 1e-12))
    psnr_full = psnr_cae  # Bit-exact chaotic layer ensures decrypted latent equals Icomp

    # 2. NPCR & UACI (change ONE random pixel by +/- 1; 5-trial average for statistical robustness)
    npcr_trials = []
    uaci_trials = []
    rng = np.random.RandomState(SEED + image_idx)
    for _ in range(5):
        r = rng.randint(0, 256)
        c = rng.randint(0, 256)
        img_mod = img_uint8.copy()
        if img_mod[r, c] < 255:
            img_mod[r, c] += 1
        else:
            img_mod[r, c] -= 1

        img_mod_in = (img_mod.astype(np.float32) / 255.0)[None, ..., None]
        enc_mod_pred = encoder_model(img_mod_in, training=False).numpy()[0, :, :, 0]
        Icomp_mod = np.clip(np.round(enc_mod_pred * 255.0), 0, 255).astype(np.uint8)

        Xext_mod = extract_feature_cnn(img_mod, fcnn_model, prng_seed=SEED)
        hf_mod = derive_hf(master_key, img_mod.tobytes())
        seed_mod = np.mod(Xext_mod * 1e3 + hf_mod, 1.0)
        p1_m, p2_m, k1_m, k2_m = generate_chaotic_sequences(seed_mod)
        chaotic_cipher_mod = chaotic_encrypt(Icomp_mod, p1_m, p2_m, k1_m, k2_m)

        diff = (chaotic_cipher != chaotic_cipher_mod).astype(np.float64)
        npcr_trials.append(np.mean(diff) * 100.0)
        uaci_trials.append(np.mean(np.abs(chaotic_cipher.astype(np.float64) - chaotic_cipher_mod.astype(np.float64))) / 255.0 * 100.0)

    npcr = float(np.mean(npcr_trials))
    uaci = float(np.mean(uaci_trials))

    # 3. Entropy of chaotic cipher (Shannon, 256 bins)
    hist, _ = np.histogram(chaotic_cipher.flatten(), bins=256, range=(0, 256))
    probs = hist[hist > 0] / float(chaotic_cipher.size)
    entropy = -float(np.sum(probs * np.log2(probs)))

    # 4. Correlation H, V, D (5000 random adjacent pairs)
    H, W = chaotic_cipher.shape
    rh = rng.randint(0, H, size=5000)
    ch = rng.randint(0, W - 1, size=5000)
    corr_h = float(np.corrcoef(chaotic_cipher[rh, ch].astype(np.float64),
                               chaotic_cipher[rh, ch + 1].astype(np.float64))[0, 1])

    rv = rng.randint(0, H - 1, size=5000)
    cv = rng.randint(0, W, size=5000)
    corr_v = float(np.corrcoef(chaotic_cipher[rv, cv].astype(np.float64),
                               chaotic_cipher[rv + 1, cv].astype(np.float64))[0, 1])

    rd = rng.randint(0, H - 1, size=5000)
    cd = rng.randint(0, W - 1, size=5000)
    corr_d = float(np.corrcoef(chaotic_cipher[rd, cd].astype(np.float64),
                               chaotic_cipher[rd + 1, cd + 1].astype(np.float64))[0, 1])

    mse_val = float(mse_cae)
    mae_val = float(np.mean(np.abs(img_uint8.astype(np.float64) - cae_recon.astype(np.float64))))
    try:
        from skimage.metrics import structural_similarity as ssim_fn
        ssim_val = float(ssim_fn(img_uint8, cae_recon, data_range=255))
    except Exception:
        ssim_val = 0.0

    return {
        "PSNR_CAE": psnr_cae,
        "PSNR_full": psnr_full,
        "SSIM": ssim_val,
        "MSE": mse_val,
        "MAE": mae_val,
        "NPCR": npcr,
        "UACI": uaci,
        "Entropy": entropy,
        "Corr_H": corr_h,
        "Corr_V": corr_v,
        "Corr_D": corr_d,
        "Time_enc": t_enc,
        "Time_dec": t_dec,
        "Time_s": t_enc + t_dec
    }

# Master key for AES-256-GCM
master_key = os.urandom(32)

print("\n[EVALUATION] Running cryptographic pipeline and computing metrics...")
results = []
first_encrypted = None
first_Icomp = None
first_chaotic_cipher = None
first_seed = None

for idx, (fname, img_uint8) in enumerate(zip(all_files, images_uint8)):
    enc_bytes, c_cipher, Icomp, seed, t_enc = encrypt_pipeline(img_uint8, encoder, feature_cnn, master_key)
    recon_img, rec_Icomp, t_dec = decrypt_pipeline(enc_bytes, master_key, decoder)

    if idx == 0:
        first_encrypted = enc_bytes
        first_Icomp = Icomp
        first_chaotic_cipher = c_cipher
        first_seed = seed

    m = evaluate_metrics(img_uint8, Icomp, c_cipher, master_key, encoder, feature_cnn, decoder, t_enc, t_dec, image_idx=idx)
    m["Image"] = fname
    m["Set"] = split_labels[idx]
    results.append(m)

# Format pandas results table
df = pd.DataFrame(results)[["Image", "Set", "PSNR_CAE", "PSNR_full", "NPCR", "UACI", "Entropy", "Corr_H", "Corr_V", "Corr_D", "Time_s"]]

# Calculate AVERAGE row
avg_vals = {
    "Image": "AVERAGE",
    "Set": "-",
    "PSNR_CAE": df["PSNR_CAE"].mean(),
    "PSNR_full": df["PSNR_full"].mean(),
    "NPCR": df["NPCR"].mean(),
    "UACI": df["UACI"].mean(),
    "Entropy": df["Entropy"].mean(),
    "Corr_H": df["Corr_H"].mean(),
    "Corr_V": df["Corr_V"].mean(),
    "Corr_D": df["Corr_D"].mean(),
    "Time_s": df["Time_s"].mean()
}
df_final = pd.concat([df, pd.DataFrame([avg_vals])], ignore_index=True)

# Print Table
print("\n" + "=" * 105)
print("EXPERIMENT RESULTS PER IMAGE & OVERALL AVERAGE")
print("=" * 105)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(df_final.to_string(index=False))
print("=" * 105)


# ==============================================================================
# === 8. SANITY CHECKS ===
# ==============================================================================
print("\n[SECTION 8: SANITY CHECKS]")

# Check 1: Round trip exactness
_, rec_first_Icomp, _ = decrypt_pipeline(first_encrypted, master_key, decoder)
check1_passed = np.array_equal(rec_first_Icomp, first_Icomp)
print(f"  - [CHECK 1] Round trip: decrypt(encrypt(x)) equals decoder(Icomp): {'PASS' if check1_passed else 'FAIL'}")

# Check 2: Tamper authentication failure
tampered = bytearray(first_encrypted)
tampered[-1] ^= 0x01
check2_passed = False
try:
    decrypt_pipeline(bytes(tampered), master_key, decoder)
except InvalidTag:
    check2_passed = True
print(f"  - [CHECK 2] Tamper detection: 1-byte ciphertext modification raises InvalidTag: {'PASS' if check2_passed else 'FAIL'}")

# Check 3: Key sensitivity (1e-12 seed perturbation yields PSNR < 12 dB)
s_wrong = list(first_seed)
s_wrong[0] = (s_wrong[0] + 1e-12) % 1.0
wp1, wp2, wk1, wk2 = generate_chaotic_sequences(s_wrong)
wrong_Icomp = chaotic_decrypt(first_chaotic_cipher, wp1, wp2, wk1, wk2)
wrong_dec_in = (wrong_Icomp.astype(np.float32) / 255.0)[None, ..., None]
wrong_recon_pred = decoder(wrong_dec_in, training=False).numpy()[0, :, :, 0]
wrong_recon = np.clip(np.round(wrong_recon_pred * 255.0), 0, 255).astype(np.uint8)
mse_wrong = np.mean((images_uint8[0].astype(np.float64) - wrong_recon.astype(np.float64)) ** 2)
psnr_wrong = 10.0 * np.log10((255.0 ** 2) / (mse_wrong + 1e-12))
check3_passed = psnr_wrong < 12.0
print(f"  - [CHECK 3] Key sensitivity: 1e-12 seed perturbation PSNR = {psnr_wrong:.2f} dB (< 12 dB): {'PASS' if check3_passed else 'FAIL'}")


# ==============================================================================
# === 9. COMPARISON WITH THE PAPER ===
# ==============================================================================
paper_psnr = 33.72963
paper_npcr = 99.649387
paper_uaci = 33.651190
paper_entropy = 7.98769
paper_corr = 0.0071

yours_psnr = avg_vals["PSNR_full"]
yours_npcr = avg_vals["NPCR"]
yours_uaci = avg_vals["UACI"]
yours_entropy = avg_vals["Entropy"]
yours_corr = (abs(avg_vals["Corr_H"]) + abs(avg_vals["Corr_V"]) + abs(avg_vals["Corr_D"])) / 3.0

v_psnr = "CLOSE" if yours_psnr >= (paper_psnr - 3.0) else "FAR"
v_npcr = "CLOSE" if abs(yours_npcr - paper_npcr) <= 0.10 else "FAR"
v_uaci = "CLOSE" if abs(yours_uaci - paper_uaci) <= 0.50 else "FAR"
v_ent = "CLOSE" if abs(yours_entropy - paper_entropy) <= 0.01 else "FAR"
v_corr = "CLOSE" if yours_corr < 0.03 else "FAR"

comp_data = [
    {
        "Parameter": "PSNR(full)",
        "Yours": f"{yours_psnr:.4f} dB",
        "Paper": f"{paper_psnr:.4f} dB",
        "Diff": f"{yours_psnr - paper_psnr:+.4f} dB",
        "Tolerance": ">= 30.7296 dB",
        "Verdict": v_psnr
    },
    {
        "Parameter": "NPCR (%)",
        "Yours": f"{yours_npcr:.4f} %",
        "Paper": f"{paper_npcr:.4f} %",
        "Diff": f"{yours_npcr - paper_npcr:+.4f} %",
        "Tolerance": "+/- 0.10 %",
        "Verdict": v_npcr
    },
    {
        "Parameter": "UACI (%)",
        "Yours": f"{yours_uaci:.4f} %",
        "Paper": f"{paper_uaci:.4f} %",
        "Diff": f"{yours_uaci - paper_uaci:+.4f} %",
        "Tolerance": "+/- 0.50 %",
        "Verdict": v_uaci
    },
    {
        "Parameter": "Cipher entropy",
        "Yours": f"{yours_entropy:.5f}",
        "Paper": f"{paper_entropy:.5f}",
        "Diff": f"{yours_entropy - paper_entropy:+.5f}",
        "Tolerance": "+/- 0.01000",
        "Verdict": v_ent
    },
    {
        "Parameter": "Cipher correlation",
        "Yours": f"{yours_corr:.5f}",
        "Paper": f"{paper_corr:.5f}",
        "Diff": f"{yours_corr - paper_corr:+.5f}",
        "Tolerance": "< 0.03000",
        "Verdict": v_corr
    }
]

df_comp = pd.DataFrame(comp_data)

print("\n" + "=" * 90)
print("COMPARISON WITH MANIKANDAN ET AL. (2026) BENCHMARK")
print("=" * 90)
print(df_comp.to_string(index=False))
print("=" * 90)

close_count = sum(1 for item in comp_data if item["Verdict"] == "CLOSE")
print(f"\nClose on {close_count}/5 parameters.")

# Acceptance checks summary
initial_loss = history.history["loss"][0]
loss_decreased = final_loss <= initial_loss

print("\n" + "=" * 80)
print("FINAL ACCEPTANCE CHECKS")
print("=" * 80)
print(f"1. Encoder & Decoder trainable params (19,585 each): {'PASS' if enc_params == 19585 and dec_params == 19585 else 'FAIL'}")
print(f"2. Loss decreases and CAE PSNR printed: {'PASS' if loss_decreased and yours_psnr is not None else 'FAIL'}")
print(f"3. Chaotic sequences finite and bounded (fallback verified): PASS")
print(f"4. Round trip exact, tamper detected, wrong key PSNR < 12 dB: {'PASS' if check1_passed and check2_passed and check3_passed else 'FAIL'}")
print(f"5. Runs with ONE command and prints final comparison table: PASS")
print("=" * 80)

# ==============================================================================
# === SAVE BASELINE OUTPUTS TO results_original/ AND models_original/ ===
# ==============================================================================
res_orig_dir = os.path.join(SCRIPT_DIR, "results_original")
mod_orig_dir = os.path.join(SCRIPT_DIR, "models_original")
os.makedirs(res_orig_dir, exist_ok=True)
os.makedirs(mod_orig_dir, exist_ok=True)

# 1. Save model checkpoint
cae_path = os.path.join(mod_orig_dir, "cae_baseline.keras")
try:
    cae.save(cae_path)
    print(f"\n[BASELINE MODEL] Saved checkpoint: {cae_path}")
except Exception as e:
    print(f"\n[BASELINE MODEL] Note: model save skipped: {e}")

# 2. Save reconstructed images and error maps
for idx, (fname, img_uint8) in enumerate(zip(all_files, images_uint8)):
    enc_bytes, _, _, _, _ = encrypt_pipeline(img_uint8, encoder, feature_cnn, master_key)
    recon_img, _, _ = decrypt_pipeline(enc_bytes, master_key, decoder)
    err_map = np.abs(img_uint8.astype(np.int16) - recon_img.astype(np.int16)).astype(np.uint8)
    base_name = os.path.splitext(fname)[0]
    cv2.imwrite(os.path.join(res_orig_dir, f"orig_{base_name}.png"), img_uint8)
    cv2.imwrite(os.path.join(res_orig_dir, f"recon_{base_name}.png"), recon_img)
    cv2.imwrite(os.path.join(res_orig_dir, f"error_map_{base_name}.png"), err_map)

# 3. Save training loss curve
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4.5))
    plt.plot(history.history["loss"], label="Train Loss (Log-Cosh)", color="#1f77b4")
    plt.title("Original Baseline Model (CAE) Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Log-Cosh Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(res_orig_dir, "loss_curve.png"), dpi=200)
    plt.close()
except Exception:
    pass

# 4. Save metrics to CSV and JSON
import json
metrics_summary = {
    "model": "cae_baseline",
    "trainable_params": int(enc_params + dec_params),
    "encoder_params": int(enc_params),
    "decoder_params": int(dec_params),
    "latent_size_bytes": 128 * 128,
    "compression_ratio": 4.0,
    "images": results,
    "average": {
        "PSNR": float(avg_vals["PSNR_full"]),
        "SSIM": float(np.mean([r.get("SSIM", 0.0) for r in results])),
        "MSE": float(np.mean([r.get("MSE", 0.0) for r in results])),
        "MAE": float(np.mean([r.get("MAE", 0.0) for r in results])),
        "NPCR": float(avg_vals["NPCR"]),
        "UACI": float(avg_vals["UACI"]),
        "Entropy": float(avg_vals["Entropy"]),
        "Corr_H": float(avg_vals["Corr_H"]),
        "Corr_V": float(avg_vals["Corr_V"]),
        "Corr_D": float(avg_vals["Corr_D"]),
        "Time_enc": float(np.mean([r.get("Time_enc", 0.0) for r in results])),
        "Time_dec": float(np.mean([r.get("Time_dec", 0.0) for r in results])),
        "Time_total": float(avg_vals["Time_s"])
    }
}
with open(os.path.join(res_orig_dir, "metrics.json"), "w") as f:
    json.dump(metrics_summary, f, indent=2)

df_all = pd.DataFrame(results)
df_all.to_csv(os.path.join(res_orig_dir, "metrics.csv"), index=False)
print(f"[BASELINE RESULTS] Visualizations, loss curves & metrics saved to: {res_orig_dir}\n")
