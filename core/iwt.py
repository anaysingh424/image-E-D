import numpy as np
from typing import Tuple

def _lifting_iwt_1d(x: np.ndarray, axis: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """
    1D Integer Lifting Haar Wavelet Transform along specified axis.
    Input dtype: int32 / int64
    Returns (s, d): low-pass update s and high-pass detail d
    """
    x = x.astype(np.int32)
    # Slice along axis
    slc_even = [slice(None)] * x.ndim
    slc_odd = [slice(None)] * x.ndim
    
    slc_even[axis] = slice(0, None, 2)
    slc_odd[axis] = slice(1, None, 2)
    
    x_even = x[tuple(slc_even)]
    x_odd = x[tuple(slc_odd)]
    
    # Dual Lifting (Predict detail): d = x_odd - x_even
    d = x_odd - x_even
    
    # Primal Lifting (Update smooth): s = x_even + floor(d // 2)
    s = x_even + (d // 2)
    
    return s, d

def _lifting_iiwt_1d(s: np.ndarray, d: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    1D Inverse Lifting Haar Wavelet Transform along specified axis.
    Reconstructs exact original x from (s, d).
    """
    # Undo Primal Lifting: x_even = s - floor(d // 2)
    x_even = s - (d // 2)
    
    # Undo Dual Lifting: x_odd = d + x_even
    x_odd = d + x_even
    
    # Interleave x_even and x_odd along axis
    out_shape = list(s.shape)
    out_shape[axis] = s.shape[axis] + d.shape[axis]
    x_rec = np.empty(out_shape, dtype=np.int32)
    
    slc_even = [slice(None)] * x_rec.ndim
    slc_odd = [slice(None)] * x_rec.ndim
    
    slc_even[axis] = slice(0, None, 2)
    slc_odd[axis] = slice(1, None, 2)
    
    x_rec[tuple(slc_even)] = x_even
    x_rec[tuple(slc_odd)] = x_odd
    
    return x_rec

def iwt2d(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    2D Integer Wavelet Transform on 2D or 3D image arrays.
    Args:
        img: numpy array of shape (H, W) or (H, W, C), dtype uint8 or int32.
             H and W should ideally be even.
    Returns:
        (LL, LH, HL, HH) subbands, all integer numpy arrays.
    """
    img_int = img.astype(np.int32)
    
    # Step 1: Transform along rows (axis 0)
    L, H = _lifting_iwt_1d(img_int, axis=0)
    
    # Step 2: Transform along columns (axis 1)
    LL, LH = _lifting_iwt_1d(L, axis=1)
    HL, HH = _lifting_iwt_1d(H, axis=1)
    
    return LL, LH, HL, HH

def iiwt2d(LL: np.ndarray, LH: np.ndarray, HL: np.ndarray, HH: np.ndarray, original_dtype=np.uint8) -> np.ndarray:
    """
    2D Inverse Integer Wavelet Transform.
    Reconstructs exact bit-identical original image from (LL, LH, HL, HH).
    """
    # Step 1: Inverse transform along columns (axis 1)
    L = _lifting_iiwt_1d(LL, LH, axis=1)
    H = _lifting_iiwt_1d(HL, HH, axis=1)
    
    # Step 2: Inverse transform along rows (axis 0)
    img_rec = _lifting_iiwt_1d(L, H, axis=0)
    
    if original_dtype == np.uint8:
        img_rec = np.clip(img_rec, 0, 255).astype(np.uint8)
    return img_rec
