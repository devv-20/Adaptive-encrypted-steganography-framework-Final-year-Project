"""
Video Steganography Module for the Adaptive Framework.
Processes frames using the Hybrid Engine and reconstructs video.
Features bit-splitting across frames and fallback reconstruction.
"""

import cv2
import numpy as np
import logging
import os
import subprocess
import shutil
import struct
import hashlib
import zlib
import math
from typing import List, Tuple

try:
    from steganography_framework import config
    from steganography_framework import hybrid_engine
    from steganography_framework import lsb_steganography
except ImportError:
    import config
    import hybrid_engine
    import lsb_steganography

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

VIDEO_MAGIC = b"VSTG"
VIDEO_VERSION = 1
VIDEO_HEADER_FORMAT = ">4sBII"
VIDEO_HEADER_SIZE = struct.calcsize(VIDEO_HEADER_FORMAT)
VIDEO_SECRET_MAGIC = b"VSEC"
VIDEO_SECRET_VERSION = 1
VIDEO_SECRET_HEADER_FORMAT = ">4sBI32s"
VIDEO_SECRET_HEADER_SIZE = struct.calcsize(VIDEO_SECRET_HEADER_FORMAT)
VIDEO_CHUNK_HEADER_BITS = 32
VIDEO_LSB_BITS_PER_CHANNEL = 1
VIDEO_FRAME_CAPACITY_UTILIZATION = 0.35

def extract_frames(video_path: str, temp_dir: str) -> List[str]:
    """Extracts frames from a video file into a temporary directory."""
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(temp_dir, f"frame_{count:06d}.png")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        count += 1
    cap.release()
    return frame_paths

def _build_video_payload(encrypted_payload: bytes, frame_step: int) -> bytes:
    """Prepends a compact video-level envelope for robust extraction."""
    header = struct.pack(VIDEO_HEADER_FORMAT, VIDEO_MAGIC, VIDEO_VERSION, len(encrypted_payload), frame_step)
    return header + encrypted_payload

def pack_video_secret(secret_bytes: bytes) -> bytes:
    """Wraps compressed video plaintext with marker, length and integrity hash."""
    compressed_secret = zlib.compress(secret_bytes, level=6)
    digest = hashlib.sha256(compressed_secret).digest()
    header = struct.pack(
        VIDEO_SECRET_HEADER_FORMAT,
        VIDEO_SECRET_MAGIC,
        VIDEO_SECRET_VERSION,
        len(compressed_secret),
        digest
    )
    return header + compressed_secret

def unpack_video_secret(payload: bytes) -> bytes:
    """Unwraps, validates and decompresses video plaintext envelope."""
    if len(payload) < VIDEO_SECRET_HEADER_SIZE:
        raise ValueError("Decrypted video payload is too small.")
    magic, version, data_len, expected_digest = struct.unpack(
        VIDEO_SECRET_HEADER_FORMAT, payload[:VIDEO_SECRET_HEADER_SIZE]
    )
    if magic != VIDEO_SECRET_MAGIC:
        raise ValueError("Invalid video payload marker after decryption.")
    if version != VIDEO_SECRET_VERSION:
        raise ValueError(f"Unsupported decrypted video payload version: {version}")
    end_idx = VIDEO_SECRET_HEADER_SIZE + data_len
    if end_idx > len(payload):
        raise ValueError("Decrypted video payload is truncated.")
    compressed_data = payload[VIDEO_SECRET_HEADER_SIZE:end_idx]
    actual_digest = hashlib.sha256(compressed_data).digest()
    if actual_digest != expected_digest:
        raise ValueError("Decrypted video payload integrity check failed.")
    try:
        return zlib.decompress(compressed_data)
    except zlib.error as exc:
        raise ValueError("Compressed video payload could not be decompressed.") from exc

def _parse_video_payload(payload: bytes) -> bytes:
    """Validates and strips the video-level envelope, returning payload and frame-step."""
    if len(payload) < VIDEO_HEADER_SIZE:
        raise ValueError("Recovered payload is too small to contain video header.")
    magic, version, enc_len, frame_step = struct.unpack(VIDEO_HEADER_FORMAT, payload[:VIDEO_HEADER_SIZE])
    if magic != VIDEO_MAGIC:
        raise ValueError("Video payload marker not found. Not a valid stego payload.")
    if version != VIDEO_VERSION:
        raise ValueError(f"Unsupported video payload version: {version}")
    if frame_step <= 0:
        raise ValueError("Invalid frame-step value in video payload header.")
    end_idx = VIDEO_HEADER_SIZE + enc_len
    if end_idx > len(payload):
        raise ValueError("Video payload is truncated or corrupted.")
    return payload[VIDEO_HEADER_SIZE:end_idx], frame_step

def _frame_payload_capacity_bytes(frame_shape: Tuple[int, int, int]) -> int:
    """Returns safe payload bytes for one video frame (low-distortion LSB-1)."""
    height, width, channels = frame_shape
    total_carrier_bytes = height * width * channels
    bits_available = (total_carrier_bytes * VIDEO_LSB_BITS_PER_CHANNEL) - VIDEO_CHUNK_HEADER_BITS
    if bits_available <= 0:
        return 0
    max_payload = bits_available // 8
    # Keep per-frame utilization conservative to avoid visible artifacts.
    return max(1, int(max_payload * VIDEO_FRAME_CAPACITY_UTILIZATION))


def _embed_chunk_lsb1(frame: np.ndarray, chunk: bytes) -> np.ndarray:
    """Embeds one chunk in a frame using low-distortion 1-bit LSB."""
    payload_bits = np.unpackbits(np.frombuffer(chunk, dtype=np.uint8)) if chunk else np.array([], dtype=np.uint8)
    header_bits = np.unpackbits(np.frombuffer(np.array([len(payload_bits)], dtype=np.uint32).tobytes(), dtype=np.uint8))
    all_bits = np.concatenate([header_bits, payload_bits])
    stego = frame.copy().reshape(-1)
    if len(all_bits) > len(stego):
        raise ValueError("Chunk exceeds frame embedding capacity in LSB-1 mode.")
    stego[:len(all_bits)] = (stego[:len(all_bits)] & 0xFE) | all_bits.astype(np.uint8)
    return stego.reshape(frame.shape)


def _extract_chunk_lsb1(frame: np.ndarray) -> bytes:
    """Extracts one LSB-1 chunk from a frame; returns empty bytes if invalid."""
    flat = frame.reshape(-1)
    if len(flat) < VIDEO_CHUNK_HEADER_BITS:
        return b""
    header_bits = (flat[:VIDEO_CHUNK_HEADER_BITS] & 1).astype(np.uint8)
    header_bytes = np.packbits(header_bits).tobytes()
    bits_exp = int(np.frombuffer(header_bytes, dtype=np.uint32)[0])
    if bits_exp < 0:
        return b""
    total_bits = VIDEO_CHUNK_HEADER_BITS + bits_exp
    if total_bits > len(flat):
        return b""
    if bits_exp == 0:
        return b""
    payload_bits = (flat[VIDEO_CHUNK_HEADER_BITS:total_bits] & 1).astype(np.uint8)
    return np.packbits(payload_bits).tobytes()

def embed_video(video_path: str, secret_data: bytes, output_path: str, frame_step: int = 0) -> None:
    """
    Embeds secret data across video frames using the Hybrid Engine.
    Splits the payload into chunks distributed across the frames.
    """
    stego_temp_dir = "temp_stego_frames"
    os.makedirs(stego_temp_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {video_path}")
        
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if total_frames == 0:
        raise ValueError("Video has no frames.")
    frame_probe_cap = cv2.VideoCapture(video_path)
    ok, frame_probe = frame_probe_cap.read()
    frame_probe_cap.release()
    if not ok or frame_probe is None:
        raise ValueError("Failed to read first frame for capacity probing.")
    per_frame_capacity = _frame_payload_capacity_bytes(frame_probe.shape)
    if per_frame_capacity <= 0:
        raise ValueError("Frame capacity is too low for embedding.")
    min_payload_size = VIDEO_HEADER_SIZE + len(secret_data)
    required_frames_min = max(1, math.ceil(min_payload_size / per_frame_capacity))
    if frame_step <= 0:
        frame_step = max(1, total_frames // required_frames_min)
    selected_frame_count = (total_frames + frame_step - 1) // frame_step
    total_capacity = per_frame_capacity * selected_frame_count
    payload_for_video = _build_video_payload(secret_data, frame_step)
    payload_size = len(payload_for_video)
    if payload_size > total_capacity:
        raise ValueError(
            f"Payload too large for this video. Required={payload_size} bytes, "
            f"Available={total_capacity} bytes."
        )

    stego_frame_paths = []

    logger.info(
        "Embedding %d encrypted bytes (%d with video header) across %d/%d frames (step=%d)...",
        len(secret_data), payload_size, selected_frame_count, total_frames, frame_step
    )
    payload_offset = 0
    frame_idx = 0
    cap.release()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not reopen video stream {video_path} for embedding.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None:
            raise ValueError(f"Failed to decode frame index {frame_idx}.")
        should_embed = (frame_idx % frame_step == 0) and (payload_offset < payload_size)
        if should_embed:
            end_offset = min(payload_offset + per_frame_capacity, payload_size)
            chunk = payload_for_video[payload_offset:end_offset]
            stego_frame = _embed_chunk_lsb1(frame, chunk)
            payload_offset = end_offset
        else:
            stego_frame = frame
        stego_fpath = os.path.join(stego_temp_dir, f"frame_{frame_idx:06d}.png")
        cv2.imwrite(stego_fpath, stego_frame)
        stego_frame_paths.append(stego_fpath)
        if (frame_idx + 1) % 50 == 0 or (frame_idx + 1) == total_frames:
            logger.info("Processed frame %d/%d", frame_idx + 1, total_frames)
        frame_idx += 1
    cap.release()

    if payload_offset != payload_size:
        raise ValueError("Embedding ended before payload was fully distributed.")

    # Reconstruct
    logger.info("Reconstructing stego video...")
    reconstruct_video(stego_frame_paths, output_path, fps, (width, height))
    
    # Cleanup
    shutil.rmtree(stego_temp_dir)
    logger.info("Video embedding complete.")

def reconstruct_video(frame_paths: List[str], output_path: str, fps: float, size: Tuple[int, int]) -> None:
    """Attempts FFmpeg reconstruction, falls back to OpenCV VideoWriter."""
    # 1. Use portable FFmpeg from config
    ffmpeg_exe = getattr(config, 'FFMPEG_PATH', 'ffmpeg')
    
    temp_dir = os.path.dirname(frame_paths[0])
    temp_pattern = os.path.join(temp_dir, "frame_%06d.png")
    
    # Ensure output has a valid extension for the chosen codec
    output_base, output_ext = os.path.splitext(output_path)
    if output_ext.lower() != '.mp4':
        output_path = output_base + ".mp4"

    # libx264rgb with crf 0 is bit-perfect for RGB frames
    cmd = [
        ffmpeg_exe, '-y', 
        '-r', str(fps), 
        '-i', temp_pattern, 
        '-c:v', 'libx264rgb', 
        '-crf', '0', 
        '-preset', 'fast',
        '-movflags', '+faststart',
        output_path
    ]
    
    try:
        logger.info(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Reconstructed video using FFmpeg (Lossless MP4): {output_path}")
            return
        else:
            logger.warning(f"FFmpeg failed (Return code {result.returncode}): {result.stderr}")
    except FileNotFoundError:
        logger.warning(f"FFmpeg not found at {ffmpeg_exe}. Ensure it is installed or vendor directory is set up.")

    # 2. Fallback: OpenCV VideoWriter with HFYU (Huffman Lossless - AVI)
    fallback_path = output_base + "_fallback.avi"
    fourcc = cv2.VideoWriter_fourcc(*'HFYU')
    out = cv2.VideoWriter(fallback_path, fourcc, fps, size)
    for fpath in frame_paths:
        frame = cv2.imread(fpath)
        if frame is not None:
            out.write(frame)
    out.release()
    logger.info(f"Reconstructed video using OpenCV (HFYU Fallback): {fallback_path}")

def extract_video(stego_video_path: str) -> bytes:
    """Extracts secret data from a stego video by aggregating bits from frames."""
    cap = cv2.VideoCapture(stego_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {stego_video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError("Stego video has no frames.")

    full_payload = b""
    logger.info("Scanning %d frames for hidden chunks...", total_frames)
    
    target_payload_len = None
    expected_total_len = None
    payload_frame_step = 1

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is None:
            idx += 1
            continue
        if idx % payload_frame_step != 0:
            idx += 1
            if (idx % 50 == 0) or (idx == total_frames):
                logger.info("Scanned frame %d/%d", idx, total_frames)
            continue
        # For simplicity, we assume every frame might have a chunk
        # hybrid_engine.extract_hybrid will return an empty byte string if no header is found
        try:
            chunk = _extract_chunk_lsb1(frame)
            if chunk:
                full_payload += chunk
                if expected_total_len is None and len(full_payload) >= VIDEO_HEADER_SIZE:
                    try:
                        magic, version, target_payload_len, frame_step = struct.unpack(
                            VIDEO_HEADER_FORMAT, full_payload[:VIDEO_HEADER_SIZE]
                        )
                        if magic != VIDEO_MAGIC:
                            raise ValueError("Video payload marker mismatch.")
                        if version != VIDEO_VERSION:
                            raise ValueError(f"Unsupported video payload version: {version}")
                        if target_payload_len < 0:
                            raise ValueError("Invalid payload length from video header.")
                        if frame_step <= 0:
                            raise ValueError("Invalid frame-step from video header.")
                        payload_frame_step = frame_step
                        expected_total_len = VIDEO_HEADER_SIZE + target_payload_len
                    except struct.error as exc:
                        raise ValueError("Unable to parse video payload header.") from exc
                if expected_total_len is not None and len(full_payload) >= expected_total_len:
                    break
        except Exception:
            # Skip frames that might fail or have no payload
            pass
        idx += 1
        if (idx % 50 == 0) or (idx == total_frames):
            logger.info("Scanned frame %d/%d", idx, total_frames)

    cap.release()
    if expected_total_len is None:
        raise ValueError("No valid video payload header found in stego video.")
    if len(full_payload) < expected_total_len:
        raise ValueError(
            f"Recovered payload is incomplete. Expected={expected_total_len} bytes, "
            f"Got={len(full_payload)} bytes."
        )
    logger.info("Extracted %d bytes from video.", expected_total_len)
    payload, _ = _parse_video_payload(full_payload[:expected_total_len])
    return payload

if __name__ == "__main__":
    # Internal test
    print("--- Video Steganography Internal Test ---")
    test_video = "test_input.mp4"
    # Create a small random video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(test_video, fourcc, 10.0, (128, 128))
    for _ in range(30):
        frame = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    
    secret = b"This is a distributed secret across 30 frames!"
    output_stego = "test_stego.mp4"
    
    try:
        embed_video(test_video, secret, output_stego)
        # Locate the actual output (might be _fallback.avi)
        actual_stego = output_stego if os.path.exists(output_stego) else "test_stego_fallback.avi"
        extracted = extract_video(actual_stego)
        print(f"Original Secret: {secret}")
        print(f"Extracted Secret: {extracted}")
        assert secret == extracted
        print("SUCCESS: Video Steganography Verified!")
    except Exception as e:
        print(f"FAILURE: {e}")
    finally:
        if os.path.exists(test_video): os.remove(test_video)
        if os.path.exists("test_stego.mp4"): os.remove("test_stego.mp4")
        if os.path.exists("test_stego_fallback.avi"): os.remove("test_stego_fallback.avi")
