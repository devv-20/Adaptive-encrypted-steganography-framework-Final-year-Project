"""
DCT Steganography Module for the Adaptive Steganographic Framework.
Implements transform domain embedding using Discrete Cosine Transform.
Robust against moderate JPEG compression.
"""

import cv2
import numpy as np
import logging
from typing import Tuple

try:
    from steganography_framework import config
except ImportError:
    import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Standard JPEG Luma Quantization Table (can be used for scaling)
Q_TABLE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

def bytes_to_bits(data: bytes) -> list:
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits

def bits_to_bytes(bits: list) -> bytes:
    byte_list = []
    for i in range(0, len(bits), 8):
        byte = 0
        for bit in bits[i:i+8]:
            byte = (byte << 1) | bit
        byte_list.append(byte)
    return bytes(byte_list)

def embed_dct(cover: np.ndarray, secret_data: bytes) -> np.ndarray:
    """
    Embeds secret data into the cover image using DCT coefficients.
    We embed 1 bit per 8x8 block in a mid-frequency coefficient.
    
    Args:
        cover (np.ndarray): Cover image (BGR).
        secret_data (bytes): Encrypted byte stream.
        
    Returns:
        np.ndarray: Stego image.
    """
    # Convert to YCrCb as DCT is typically applied to Luminance (Y)
    ycrcb = cv2.cvtColor(cover, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y, cr, cb = cv2.split(ycrcb)
    
    bits = bytes_to_bits(secret_data)
    # Prefix with length
    header_bits = [int((len(bits) >> i) & 1) for i in range(31, -1, -1)]
    all_bits = header_bits + bits
    bit_idx = 0
    total_bits = len(all_bits)

    h, w = y.shape
    bs = config.BLOCK_SIZE
    
    # Coordinates of a mid-frequency coefficient (e.g., Row 4, Col 4)
    # This avoids DC (low freq) and very high freq components
    target_coeff = (4, 4)
    
    # Step size for quantization-based embedding (higher = more robust, lower = better quality)
    quant_step = 25.0 

    for i in range(0, h - bs + 1, bs):
        for j in range(0, w - bs + 1, bs):
            if bit_idx >= total_bits:
                break
                
            block = y[i:i+bs, j:j+bs]
            # 1. Apply DCT
            dct_block = cv2.dct(block)
            
            # 2. Embed bit using Quantization Index Modulation (QIM)
            # We modify dct_block[target_coeff]
            val = dct_block[target_coeff]
            bit = all_bits[bit_idx]
            
            # Quantize it: find nearest multiple of quant_step
            # If bit=0, we want it to be an even multiple of step/2?
            # Simpler QIM: q = round(val / step). If q % 2 != bit, adjust.
            q = round(val / quant_step)
            if q % 2 != bit:
                if val > q * quant_step:
                    q += 1
                else:
                    q -= 1
            
            dct_block[target_coeff] = q * quant_step
            
            # 3. Inverse DCT
            y[i:i+bs, j:j+bs] = cv2.idct(dct_block)
            bit_idx += 1
            
    if bit_idx < total_bits:
        logger.warning(f"DCT Capacity exceeded! Embedded {bit_idx}/{total_bits} bits.")
    else:
        logger.info(f"DCT Embedding complete. Embedded {bit_idx} bits.")

    # Reconstruct image
    y_clamped = np.clip(y, 0, 255)
    stego_ycrcb = cv2.merge([y_clamped, cr, cb])
    stego_bgr = cv2.cvtColor(stego_ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    
    return stego_bgr

def extract_dct(stego: np.ndarray) -> bytes:
    """
    Extracts secret data from the stego image in the DCT domain.
    """
    ycrcb = cv2.cvtColor(stego, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y, _, _ = cv2.split(ycrcb)
    
    h, w = y.shape
    bs = config.BLOCK_SIZE
    target_coeff = (4, 4)
    quant_step = 25.0
    
    extracted_bits = []
    total_bits_expected = None
    
    for i in range(0, h - bs + 1, bs):
        for j in range(0, w - bs + 1, bs):
            block = y[i:i+bs, j:j+bs]
            dct_block = cv2.dct(block)
            
            val = dct_block[target_coeff]
            # Decode bit: q = round(val / step). Bit is q % 2.
            q = round(val / quant_step)
            extracted_bits.append(int(abs(q) % 2))
            
            if total_bits_expected is None and len(extracted_bits) >= 32:
                total_bits_expected = 0
                for b in extracted_bits[:32]:
                    total_bits_expected = (total_bits_expected << 1) | b
                logger.info(f"DCT Extracted header: expecting {total_bits_expected} bits.")
            
            if total_bits_expected is not None and len(extracted_bits) >= (32 + total_bits_expected):
                return bits_to_bytes(extracted_bits[32:32 + total_bits_expected])
                
    return b""

if __name__ == "__main__":
    print("--- DCT Steganography Demo ---")
    
    # 1. Load/Create sample cover
    cover_img = np.random.randint(200, 255, (256, 256, 3), dtype=np.uint8) # Bright image better for DCT demo
    
    # 2. Secret data
    secret = b"DCT Stego Robust!"
    print(f"Secret: {secret}")
    
    # 3. Embed
    stego_img = embed_dct(cover_img, secret)
    
    # 4. Extract
    extracted = extract_dct(stego_img)
    print(f"Extracted: {extracted}")
    
    # 5. Verify
    assert secret == extracted
    print("Verification (Lossless): SUCCESS")
    
    # 6. Test basic JPEG robustness (Simulated)
    # Encode and decode as JPEG
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, encimg = cv2.imencode('.jpg', stego_img, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    
    extracted_robust = extract_dct(decimg)
    print(f"Extracted after JPEG (Q=90): {extracted_robust}")
    if secret == extracted_robust:
        print("Robustness Test: SUCCESS")
    else:
        print("Robustness Test: FAILED (Bit correlation needed or higher quant_step)")
