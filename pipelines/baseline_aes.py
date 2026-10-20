import os
import hashlib
import numpy as np
from typing import Tuple, Dict, Any
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend

from core.base import BaseImageCipher

class BaselineAESCipher(BaseImageCipher):
    """
    Standard AES-256-CBC Baseline Image Cipher.
    Used for comparative cryptographic and transmission benchmarks.
    """
    def __init__(self, key: bytes = None):
        super().__init__()
        if key is None:
            self.key = hashlib.sha256(b"AES256_BASELINE_MASTER_KEY_2026").digest()
        else:
            self.key = hashlib.sha256(key).digest()

    def encrypt(self, image: np.ndarray, public_key: Any = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        H, W, C = image.shape
        raw_bytes = image.tobytes()
        
        # PKCS7 padding
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(raw_bytes) + padder.finalize()
        
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted_bytes = encryptor.update(padded_data) + encryptor.finalize()
        
        # Cut or pad encrypted bytes to fit H * W * C exact shape for visual display
        expected_len = H * W * C
        if len(encrypted_bytes) > expected_len:
            cipher_bytes = encrypted_bytes[:expected_len]
            overflow = encrypted_bytes[expected_len:]
        else:
            overflow = b""
            cipher_bytes = encrypted_bytes + b"\x00" * (expected_len - len(encrypted_bytes))
            
        cipher_image = np.frombuffer(cipher_bytes, dtype=np.uint8).reshape((H, W, C)).copy()
        
        envelope_header = {
            "iv": iv,
            "overflow": overflow,
            "orig_shape": (H, W, C),
            "pad_len": len(padded_data) - len(raw_bytes)
        }
        
        return cipher_image, envelope_header

    def decrypt(self, cipher_image: np.ndarray, envelope_header: Dict[str, Any], private_key: Any = None) -> np.ndarray:
        iv = envelope_header["iv"]
        overflow = envelope_header["overflow"]
        H, W, C = envelope_header["orig_shape"]
        expected_len = H * W * C
        
        cipher_bytes = cipher_image.tobytes()
        full_encrypted_bytes = cipher_bytes[:expected_len] + overflow
        
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(iv), backend=default_backend())
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(full_encrypted_bytes) + decryptor.finalize()
        
        unpadder = padding.PKCS7(128).unpadder()
        raw_bytes = unpadder.update(padded_data) + unpadder.finalize()
        
        restored_image = np.frombuffer(raw_bytes[:H*W*C], dtype=np.uint8).reshape((H, W, C)).copy()
        return restored_image
