import hashlib
import numpy as np

def _seed_from_password(password: str) -> int:
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def quantum_keystream(password: str, nbytes: int) -> bytes:
    """
    Fast educational demo:
    Simulate the quantum circuit once, then expand its result
    into a reproducible classical byte stream.
    Not cryptographically secure.
    """
    seed = _seed_from_password(password)

    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator

        n = 8
        qc = QuantumCircuit(n, n)

        for q in range(n):
            bit = (seed >> q) & 1
            if bit:
                qc.x(q)
            qc.h(q)
            qc.rz(
                ((seed >> (q + 8)) & 255) * np.pi / 255.0,
                q
            )

        for q in range(n - 1):
            qc.cx(q, q + 1)

        qc.measure(range(n), range(n))

        sim = AerSimulator(seed_simulator=seed)
        tqc = transpile(qc, sim)

        # Run the quantum circuit only once.
        result = sim.run(
            tqc, shots=1, seed_simulator=seed
        ).result()

        measured = next(iter(result.get_counts())).replace(" ", "")
        quantum_value = int(measured, 2)

        # Expand the result using a reproducible classical generator.
        stream_seed = seed ^ quantum_value

    except ImportError:
        # Fallback when Qiskit Aer is unavailable.
        stream_seed = seed

    rng = np.random.default_rng(stream_seed)
    return rng.integers(
        0, 256, size=nbytes, dtype=np.uint8
    ).tobytes()

def _permutation(password: str, n: int) -> np.ndarray:
    seed = _seed_from_password(password) ^ 0xA5A5A5A5
    return np.random.default_rng(seed).permutation(n)

def encrypt_array(arr: np.ndarray, password: str):
    if not password:
        raise ValueError("Passphrase must not be empty.")
    flat = np.asarray(arr, dtype=np.uint8).reshape(-1)
    stream = np.frombuffer(quantum_keystream(password, flat.size), dtype=np.uint8)
    perm = _permutation(password, flat.size)
    # First permute, then XOR. Inverse operation is implemented in decrypt_array.
    scrambled = flat[perm] ^ stream
    cipher = scrambled.reshape(arr.shape)
    metadata = {"shape": list(arr.shape), "dtype": "uint8"}
    return cipher, metadata

def decrypt_array(cipher: np.ndarray, password: str, metadata: dict):
    flat = np.asarray(cipher, dtype=np.uint8).reshape(-1)
    stream = np.frombuffer(quantum_keystream(password, flat.size), dtype=np.uint8)
    perm = _permutation(password, flat.size)
    scrambled = flat ^ stream
    recovered = np.empty_like(scrambled)
    recovered[perm] = scrambled
    return recovered.reshape(tuple(metadata["shape"]))

def metrics(original: np.ndarray, cipher: np.ndarray) -> dict:
    a = np.asarray(original, dtype=np.uint8).reshape(-1)
    b = np.asarray(cipher, dtype=np.uint8).reshape(-1)
    return {"pixel_count": int(a.size), "different_pixels": int(np.count_nonzero(a != b))}
