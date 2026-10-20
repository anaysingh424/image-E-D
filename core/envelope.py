import hashlib
from typing import Tuple
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes

class EnvelopeManager:
    """
    RSA-2048 / RSA-OAEP Ephemeral Asymmetric Key Header Manager.
    Encrypts 256-bit SHA-256 image digest into an asymmetric envelope.
    """
    @staticmethod
    def generate_keypair(key_size: int = 2048) -> Tuple[rsa.RSAPrivateKey, rsa.RSAPublicKey]:
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
        )
        return private_key, private_key.public_key()

    @staticmethod
    def pack_envelope(seed_bytes: bytes, public_key: rsa.RSAPublicKey) -> bytes:
        """
        Encrypt 256-bit seed payload with recipient RSA public key using OAEP padding.
        """
        if len(seed_bytes) != 32:
            seed_bytes = hashlib.sha256(seed_bytes).digest()
            
        encrypted_envelope = public_key.encrypt(
            seed_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return encrypted_envelope

    @staticmethod
    def unpack_envelope(encrypted_envelope: bytes, private_key: rsa.RSAPrivateKey) -> bytes:
        """
        Decrypt RSA-OAEP envelope using recipient private key to recover exact 256-bit seed payload.
        """
        decrypted_seed = private_key.decrypt(
            encrypted_envelope,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return decrypted_seed
