"""
Unified Extraction Module for the Steganographic Framework.
Acts as the reverse pipeline for the Hybrid Engine and Video Module.
Includes ChaCha20 decryption.
"""

import cv2
import numpy as np
import logging
import os

try:
    from steganography_framework import encryption
    from steganography_framework import hybrid_engine
    from steganography_framework import video_steganography
    from steganography_framework import payload_control
except ImportError:
    import encryption
    import hybrid_engine
    import video_steganography
    import payload_control

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def extract_from_image(stego_path: str, key: bytes, nonce: bytes) -> bytes:
    """
    Extracts and decrypts a secret from a stego image.
    """
    stego = cv2.imread(stego_path)
    if stego is None:
        raise FileNotFoundError(f"Could not read image at {stego_path}")
        
    # 1. Generate map (dummy or real depending on engine version)
    cmap = payload_control.generate_complexity_map(stego)
    
    # 2. Extract using Hybrid Engine
    encrypted_payload = hybrid_engine.extract_hybrid(stego, cmap)
    if not encrypted_payload:
        logger.warning("No payload found in image.")
        return b""
        
    # 3. Decrypt
    decrypted = encryption.decrypt_payload(encrypted_payload, key, nonce)
    return decrypted

def extract_from_video(stego_video_path: str, key: bytes, nonce: bytes) -> bytes:
    """
    Extracts and decrypts a secret from a stego video.
    """
    # 1. Video extraction loop
    try:
        encrypted_payload = video_steganography.extract_video(stego_video_path)
    except Exception as e:
        logger.error(f"Video payload recovery failed: {e}")
        return b""
    if not encrypted_payload:
        logger.warning("No payload found in video.")
        return b""
        
    # 2. Decrypt
    # Note: If the payload was split across frames, we need the whole blob.
    # Our video_steganography.extract_video already aggregates them.
    try:
        decrypted = encryption.decrypt_payload(encrypted_payload, key, nonce)
        return video_steganography.unpack_video_secret(decrypted)
    except Exception as e:
        logger.error(f"Decryption failed (wrong key/nonce or corrupted payload): {e}")
        return b""

if __name__ == "__main__":
    print("--- Unified Extraction Hook Initialized ---")
    # This module will be primarily used by main.py
