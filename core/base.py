from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any
import numpy as np

class BaseImageCipher(ABC):
    """
    Standard interface for all image encryption pipelines.
    Subclass this class to create and benchmark new pipelines.
    """
    
    @abstractmethod
    def encrypt(self, image: np.ndarray, public_key: Any = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Encrypt an input image.
        
        Args:
            image: (H, W, C) uint8 numpy array
            public_key: Optional recipient public key or shared master seed
        Returns:
            cipher_image: (H, W, C) uint8 numpy array
            envelope_header: Dictionary of metadata/key tokens required for decryption
        """
        pass

    @abstractmethod
    def decrypt(self, cipher_image: np.ndarray, envelope_header: Dict[str, Any], private_key: Any = None) -> np.ndarray:
        """
        Decrypt a cipher image.
        
        Args:
            cipher_image: (H, W, C) uint8 numpy array
            envelope_header: Dictionary generated during encryption
            private_key: Optional recipient private key
        Returns:
            restored_image: (H, W, C) uint8 numpy array
        """
        pass
