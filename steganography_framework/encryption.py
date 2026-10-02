"""
Encryption module for the Steganographic Framework.
Uses ChaCha20 stream cipher for high performance and security.
"""

import logging
from typing import Tuple
from Crypto.Cipher import ChaCha20  # type: ignore
from Crypto.Random import get_random_bytes  # type: ignore

try:
    from steganography_framework import config
except ImportError:
    import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def generate_key_and_nonce() -> Tuple[bytes, bytes]:
    """
    Generates a random 256-bit key and 96-bit nonce.
    
    Returns:
        Tuple[bytes, bytes]: (key, nonce)
    """
    key = get_random_bytes(config.CHACHA20_KEY_SIZE)
    nonce = get_random_bytes(config.CHACHA20_NONCE_SIZE)
    return key, nonce

def encrypt_payload(payload: bytes, key: bytes, nonce: bytes) -> bytes:
    """
    Encrypts the given payload using ChaCha20 cipher.
    
    Args:
        payload (bytes): The secret data to encrypt.
        key (bytes): 256-bit encryption key.
        nonce (bytes): 96-bit nonce.
        
    Returns:
        bytes: The encrypted byte stream.
        
    Raises:
        ValueError: If key or nonce size is incorrect.
    """
    if len(key) != config.CHACHA20_KEY_SIZE:
        raise ValueError(f"Key must be {config.CHACHA20_KEY_SIZE} bytes.")
    if len(nonce) != config.CHACHA20_NONCE_SIZE:
        raise ValueError(f"Nonce must be {config.CHACHA20_NONCE_SIZE} bytes.")
    
    try:
        cipher = ChaCha20.new(key=key, nonce=nonce)
        ciphertext = cipher.encrypt(payload)
        logger.info("Payload successfully encrypted.")
        return ciphertext
    except Exception as e:
        logger.error(f"Encryption failed: {str(e)}")
        raise

def decrypt_payload(ciphertext: bytes, key: bytes, nonce: bytes) -> bytes:
    """
    Decrypts the given ciphertext using ChaCha20 cipher.
    
    Args:
        ciphertext (bytes): The encrypted data.
        key (bytes): 256-bit encryption key.
        nonce (bytes): 96-bit nonce.
        
    Returns:
        bytes: The decrypted original payload.
        
    Raises:
        ValueError: If key or nonce size is incorrect.
    """
    if len(key) != config.CHACHA20_KEY_SIZE:
        raise ValueError(f"Key must be {config.CHACHA20_KEY_SIZE} bytes.")
    if len(nonce) != config.CHACHA20_NONCE_SIZE:
        raise ValueError(f"Nonce must be {config.CHACHA20_NONCE_SIZE} bytes.")
    
    try:
        cipher = ChaCha20.new(key=key, nonce=nonce)
        decrypted_payload = cipher.decrypt(ciphertext)
        logger.info("Payload successfully decrypted.")
        return decrypted_payload
    except Exception as e:
        logger.error(f"Decryption failed: {str(e)}")
        raise

if __name__ == "__main__":
    # --- Quick Demo ---
    print("--- ChaCha20 Encryption Demo ---")
    
    # 1. Prepare sample data
    secret_message = b"Top secret data: Steganography is fun!"
    print(f"Original: {secret_message.decode()}")
    
    # 2. Generate credentials
    test_key, test_nonce = generate_key_and_nonce()
    print(f"Key (hex): {test_key.hex()}")
    print(f"Nonce (hex): {test_nonce.hex()}")
    
    # 3. Encrypt
    encrypted = encrypt_payload(secret_message, test_key, test_nonce)
    print(f"Encrypted (hex): {encrypted.hex()}")
    
    # 4. Decrypt
    decrypted = decrypt_payload(encrypted, test_key, test_nonce)
    print(f"Decrypted: {decrypted.decode()}")
    
    # 5. Verify
    assert secret_message == decrypted
    print("Verification: SUCCESS")
