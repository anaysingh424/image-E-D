import numpy as np
import scipy.stats
from typing import Tuple, Dict

def calculate_entropy(image: np.ndarray) -> float:
    """
    Compute Shannon Information Entropy H(m) across pixel intensity levels [0, 255].
    Target for ideal random cipher: H > 7.998 bits.
    """
    flat = image.flatten()
    hist, _ = np.histogram(flat, bins=256, range=(0, 256))
    prob = hist / float(flat.size)
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log2(prob))
    return float(entropy)

def calculate_pixel_correlation(image: np.ndarray, num_samples: int = 10000) -> Dict[str, float]:
    """
    Calculate Adjacent Pixel Correlation Coefficient r_xy in Horizontal, Vertical, and Diagonal directions.
    Target for cipher image: |r| < 0.002.
    """
    if image.ndim == 3:
        # Convert to grayscale for correlation or average across channels
        gray = np.mean(image, axis=2).astype(np.float64)
    else:
        gray = image.astype(np.float64)
        
    H, W = gray.shape
    np.random.seed(42)
    
    # Horizontal pairs
    h_x_idx = np.random.randint(0, H, num_samples)
    h_y_idx = np.random.randint(0, W - 1, num_samples)
    x_h = gray[h_x_idx, h_y_idx]
    y_h = gray[h_x_idx, h_y_idx + 1]
    r_h = float(np.corrcoef(x_h, y_h)[0, 1]) if np.std(x_h) > 0 and np.std(y_h) > 0 else 0.0
    
    # Vertical pairs
    v_x_idx = np.random.randint(0, H - 1, num_samples)
    v_y_idx = np.random.randint(0, W, num_samples)
    x_v = gray[v_x_idx, v_y_idx]
    y_v = gray[v_x_idx + 1, v_y_idx]
    r_v = float(np.corrcoef(x_v, y_v)[0, 1]) if np.std(x_v) > 0 and np.std(y_v) > 0 else 0.0
    
    # Diagonal pairs
    d_x_idx = np.random.randint(0, H - 1, num_samples)
    d_y_idx = np.random.randint(0, W - 1, num_samples)
    x_d = gray[d_x_idx, d_y_idx]
    y_d = gray[d_x_idx + 1, d_y_idx + 1]
    r_d = float(np.corrcoef(x_d, y_d)[0, 1]) if np.std(x_d) > 0 and np.std(y_d) > 0 else 0.0
    
    return {"horizontal": r_h, "vertical": r_v, "diagonal": r_d}

def calculate_chi_square(image: np.ndarray) -> float:
    """
    Compute Histogram Chi-Square Statistic chi^2 against uniform distribution.
    Critical value threshold chi^2_0.05(255) = 293.247.
    """
    flat = image.flatten()
    hist, _ = np.histogram(flat, bins=256, range=(0, 256))
    expected = float(flat.size) / 256.0
    chi2 = np.sum(((hist.astype(np.float64) - expected) ** 2) / expected)
    return float(chi2)

def calculate_npcr_uaci(c1: np.ndarray, c2: np.ndarray) -> Tuple[float, float]:
    """
    Compute NPCR (%) and UACI (%) between two encrypted cipher images c1 and c2.
    Ideal theoretical expectations: NPCR = 99.6094%, UACI = 33.4635%.
    """
    c1 = c1.astype(np.int64)
    c2 = c2.astype(np.int64)
    
    diff = c1 != c2
    npcr = (np.sum(diff) / float(c1.size)) * 100.0
    
    abs_diff = np.abs(c1 - c2)
    uaci = (np.sum(abs_diff / 255.0) / float(c1.size)) * 100.0
    
    return float(npcr), float(uaci)

def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) in dB between two images.
    Returns float('inf') for identical images.
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr = 10.0 * np.log10((255.0 ** 2) / mse)
    return float(psnr)

def calculate_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM) between img1 and img2.
    Returns 1.0000 for bit-identical images.
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim)
