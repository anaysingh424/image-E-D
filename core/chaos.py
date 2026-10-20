import numpy as np
import hashlib
from typing import Tuple

class ChaosGenerator:
    """
    3D-Sine-Tent-Cosine (3D-STC) Discrete Hyperchaotic Generator.
    Generates high-entropy, pseudo-random, non-repeating keystreams.
    """
    def __init__(self, x0: float = 0.345678912, y0: float = 0.654321987, z0: float = 0.123456789):
        self.x = (float(x0) % 0.98) + 0.01
        self.y = (float(y0) % 0.98) + 0.01
        self.z = (float(z0) % 0.98) + 0.01
        self.step_count = 0

    @classmethod
    def from_hash(cls, hash_bytes: bytes):
        """
        Derive initial chaotic seeds (x0, y0, z0) deterministically from SHA-256 digest bytes.
        """
        if len(hash_bytes) < 32:
            hash_bytes = hashlib.sha256(hash_bytes).digest()
            
        # Parse 8-byte chunks with 64-bit float precision
        v1 = (int.from_bytes(hash_bytes[0:10], 'big') % (2**52)) / (2**52)
        v2 = (int.from_bytes(hash_bytes[10:20], 'big') % (2**52)) / (2**52)
        v3 = (int.from_bytes(hash_bytes[20:30], 'big') % (2**52)) / (2**52)
        
        return cls(x0=v1, y0=v2, z0=v3)

    def _step(self) -> float:
        """Execute one iteration of 3D-STC hyperchaotic map."""
        self.step_count += 1
        
        # x_next Sine-Cosine non-linear map with dynamic perturbation
        x_raw = np.sin(3.98765 * np.pi * (self.y + 0.123456)) * np.cos(4.12345 * np.pi * (self.z + 0.654321))
        x_next = abs(x_raw + np.sin(self.step_count * 0.001)) % 1.0
        if x_next < 1e-6:
            x_next = 0.345678
            
        # y_next Tent Map with perturbation
        if x_next < 0.499:
            y_next = x_next / 0.499
        else:
            y_next = (1.0 - x_next) / 0.501
        y_next = abs(y_next) % 1.0
        if y_next < 1e-6:
            y_next = 0.654321
            
        # z_next Cosine-Sine coupled state
        z_raw = np.cos(3.87654 * np.pi * (x_next + y_next)) * np.sin(4.09876 * np.pi * (self.x + 0.4321))
        z_next = abs(z_raw) % 1.0
        if z_next < 1e-6:
            z_next = 0.123456
            
        self.x, self.y, self.z = x_next, y_next, z_next
        return (x_next + y_next * 1e2 + z_next * 1e4)

    def generate_bytes(self, length: int) -> np.ndarray:
        """
        Generate `length` pseudo-random uint8 byte keystream with uniform distribution.
        """
        # Warm-up transient iterations
        for _ in range(100):
            self._step()
            
        keystream = np.empty(length, dtype=np.uint8)
        for i in range(length):
            v = self._step()
            # Extract 8-bit integer via high-frequency hashing of floating point state
            val = int(abs(v) * 1e8) % 256
            keystream[i] = val
        return keystream

    def generate_permutation_indices(self, length: int) -> np.ndarray:
        """
        Generate a chaotic spatial permutation (shuffle) of range(length).
        """
        # Warm-up
        for _ in range(100):
            self._step()
            
        float_seq = np.empty(length, dtype=np.float64)
        for i in range(length):
            v = self._step()
            # High precision float mixing for argsort shuffling
            float_seq[i] = (v * 1e6 + i * 1e-7) % 1.00000000000
            
        perm_indices = np.argsort(float_seq)
        return perm_indices

def invert_permutation(perm_indices: np.ndarray) -> np.ndarray:
    """
    Compute inverse permutation array such that arr[perm][inv_perm] == arr.
    """
    inv_perm = np.empty_like(perm_indices)
    inv_perm[perm_indices] = np.arange(len(perm_indices))
    return inv_perm
