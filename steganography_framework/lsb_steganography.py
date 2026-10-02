"""
LSB Steganography Module for the Adaptive Steganographic Framework.
Pure NumPy bit manipulation for peak performance.
"""

import numpy as np
import logging
from typing import Union

try:
    from steganography_framework import config
except ImportError:
    import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def bytes_to_bits(data: bytes) -> np.ndarray:
    """Converts bytes to a NumPy array of bits (uint8). Fast."""
    if not data:
        return np.array([], dtype=np.uint8)
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))

def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Converts a NumPy array of bits to bytes. Fast."""
    if bits.size == 0:
        return b""
    # Ensure multiple of 8
    pad = (8 - bits.size % 8) % 8
    if pad > 0:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits).tobytes()

# embed_lsb and extract_lsb would also benefit from vectorization if used standalone.
# However, hybrid_engine uses its own vectorized logic now.
