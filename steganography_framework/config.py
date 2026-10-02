"""
Configuration constants for the Adaptive Encrypted Steganographic Framework.
This module stores all hardcoded parameters to ensure consistency across the framework.
"""

import os

# Encryption Constants (ChaCha20)
CHACHA20_KEY_SIZE = 32   # 256-bit key
CHACHA20_NONCE_SIZE = 12  # 96-bit nonce (required for standard ChaCha20)

# Steganography Constants
BLOCK_SIZE = 8           # Standard block size for DCT and complexity analysis (8x8)
LOW_COMPLEXITY_BITS = 1   # Bits per channel for LSB in smooth regions
HIGH_COMPLEXITY_BITS = 3  # Bits per channel for LSB in textured/edge regions

# Payload Control (Texture/Edge Analysis)
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150
LAPLACIAN_VAR_THRESHOLD = 100.0  # Threshold for classifying HIGH vs LOW complexity

# DCT Constants
DCT_COEFF_START = 1      # Skip the DC component (index 0)
DCT_COEFF_END = 28       # Practical mid-frequency range for 8x8 blocks

# Video Constants
FRAME_CONSISTENCY_CHECK = True

# Common Paths (Optional - can be overridden via CLI)
DEFAULT_LOG_DIR = os.path.join(os.getcwd(), "logs")

# FFmpeg Path (Portable setup)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFMPEG_PATH = os.path.join(BASE_DIR, "vendor", "ffmpeg_extracted", "ffmpeg-8.1-essentials_build", "bin", "ffmpeg.exe")
