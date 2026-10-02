"""
Payload Control Module for the Adaptive Framework.
Deterministic, Bit-Perfect Complexity Analysis (MSB-based).
"""

import cv2
import numpy as np
import logging

try:
    from steganography_framework import config
except ImportError:
    import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def generate_complexity_map(image: np.ndarray) -> np.ndarray:
    """
    100% Stable Complexity Analysis.
    Uses block-wise standard deviation on MSBs (Morphed Image).
    """
    if image is None: return None
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # MSB invariance: Kill last 4 bits
    gray_msb = gray & 0xF0

    h, w = gray_msb.shape
    bs = config.BLOCK_SIZE
    nh, nw = h // bs, w // bs
    
    # Reshape into blocks: (nh, bs, nw, bs)
    blocks = gray_msb[:nh*bs, :nw*bs].reshape(nh, bs, nw, bs).transpose(0, 2, 1, 3).reshape(nh, nw, bs*bs)
    
    # Calculate Standard Deviation per block
    # This is 100% stable since it only uses the first 4 bits
    block_std = np.std(blocks, axis=2)
    
    # Calculate Edge Density (Sobel on MSB)
    sobel_x = cv2.Sobel(gray_msb, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_msb, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # Block-wise Edge Magnitude (average)
    edge_blocks = edge_mag[:nh*bs, :nw*bs].reshape(nh, bs, nw, bs).transpose(0, 2, 1, 3).mean(axis=(2, 3))

    # Adaptive Classification
    # HIGH if std > threshold OR edge_mag > threshold
    # Note: Thresholds are tuned for MSB-only images
    complexity_map = ((block_std > 10.0) | (edge_blocks > 15.0)).astype(np.uint8)
                      
    return complexity_map
