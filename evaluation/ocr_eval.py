import pytesseract
import cv2
import numpy as np

def _levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings s1 and s2."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def extract_ocr_text(img: np.ndarray) -> str:
    """Extract OCR text from image using Tesseract."""
    try:
        # Check if tesseract is accessible
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        # If pytesseract binary is not found, return empty string or raw visual noise marker
        return ""

def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Compute Character Error Rate (CER) = (S + D + I) / N * 100%.
    Target on Encrypted Image: CER = 100.0%.
    Target on Clean Decrypted Image: CER = 0.0%.
    """
    ref_clean = "".join(reference.split())
    hyp_clean = "".join(hypothesis.split())
    
    if len(ref_clean) == 0:
        return 0.0 if len(hyp_clean) == 0 else 100.0
        
    distance = _levenshtein_distance(ref_clean, hyp_clean)
    cer = (distance / float(len(ref_clean))) * 100.0
    return float(min(100.0, cer))

def calculate_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate (WER) = (Sw + Dw + Iw) / Nw * 100%.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 100.0
        
    distance = _levenshtein_distance(ref_words, hyp_words)
    wer = (distance / float(len(ref_words))) * 100.0
    return float(min(100.0, wer))
