import numpy as np

def apply_salt_and_pepper_noise(image: np.ndarray, amount: float = 0.05, salt_vs_pepper: float = 0.5) -> np.ndarray:
    """
    Apply Salt-and-Pepper channel noise to cipher image array (H, W, C).
    `amount`: Ratio of pixels to corrupt (e.g. 0.01 = 1%, 0.05 = 5%, 0.10 = 10%).
    """
    corrupted = image.copy()
    num_noise = int(amount * image.size)
    
    # Salt noise (val = 255)
    num_salt = int(num_noise * salt_vs_pepper)
    coords_salt = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    corrupted[tuple(coords_salt)] = 255
    
    # Pepper noise (val = 0)
    num_pepper = int(num_noise * (1.0 - salt_vs_pepper))
    coords_pepper = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    corrupted[tuple(coords_pepper)] = 0
    
    return corrupted

def apply_occlusion_attack(image: np.ndarray, crop_size: int = 64) -> np.ndarray:
    """
    Simulate Data Loss / Occlusion (Cropping Attack).
    Zeros out a central (crop_size x crop_size) block of the cipher image.
    """
    corrupted = image.copy()
    H, W = image.shape[:2]
    
    center_y, center_x = H // 2, W // 2
    half_crop = crop_size // 2
    
    y1, y2 = max(0, center_y - half_crop), min(H, center_y + half_crop)
    x1, x2 = max(0, center_x - half_crop), min(W, center_x + half_crop)
    
    corrupted[y1:y2, x1:x2] = 0
    return corrupted
