"""
Hybrid Steganography Engine for the Adaptive Framework.
Static High-Capacity LSB-3 (Guaranteed Stability).
"""

import cv2
import numpy as np
import logging
from typing import Tuple

try:
    from steganography_framework import config
    from steganography_framework import lsb_steganography
except ImportError:
    import config
    import lsb_steganography

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def embed_hybrid(cover: np.ndarray, secret_data: bytes) -> Tuple[np.ndarray, np.ndarray, int]:
    """Static LSB-3 Embedding (100% stable)."""
    if cover is None: raise ValueError("Cover is None")
    
    # We use a dummy cmap for compatibility
    cmap = np.ones((cover.shape[0]//8, cover.shape[1]//8), dtype=np.uint8)
    
    payload_bits = lsb_steganography.bytes_to_bits(secret_data)
    header_bits = lsb_steganography.bytes_to_bits(np.array([len(payload_bits)], dtype=np.uint32).tobytes())
    all_bits = np.concatenate([header_bits, payload_bits])
    
    stego = cover.copy()
    flat = stego.reshape(-1)
    
    bit_idx = 0; total = len(all_bits); bp = 3 # Fixed 3 bits
    mask = (0xFF << bp) & 0xFF
    
    for i in range(len(flat)):
        if bit_idx >= total: break
        val = 0
        for _ in range(bp):
            if bit_idx < total:
                val = (val << 1) | int(all_bits[bit_idx]); bit_idx += 1
            else:
                val = (val << 1) | 0
        flat[i] = (flat[i] & mask) | val
        
    return stego, cmap, bit_idx

def extract_hybrid(stego: np.ndarray, cmap: np.ndarray) -> bytes:
    """Static LSB-3 Extraction (100% stable)."""
    flat = stego.reshape(-1)
    all_bits = []; bits_exp = None; bit_cnt = 0; bp = 3
    
    for i in range(len(flat)):
        if bits_exp is not None and bit_cnt >= (32 + bits_exp): break
        
        val = int(flat[i]) & ((1 << bp) - 1)
        for k in range(bp-1, -1, -1):
            if bits_exp is not None and bit_cnt >= (32 + bits_exp): break
            all_bits.append((val >> k) & 1); bit_cnt += 1
            
            if bits_exp is None and bit_cnt == 32:
                header_bytes = lsb_steganography.bits_to_bytes(np.array(all_bits[:32], dtype=np.uint8))
                bits_exp = int(np.frombuffer(header_bytes, dtype=np.uint32)[0])
                if bits_exp > 100 * 1024 * 1024: # Cap at 100MB to avoid OOM
                    return b""

    if bits_exp is None: return b""
    return lsb_steganography.bits_to_bytes(np.array(all_bits[32 : 32 + bits_exp], dtype=np.uint8))
