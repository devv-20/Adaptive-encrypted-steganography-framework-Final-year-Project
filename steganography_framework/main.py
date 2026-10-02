"""
Main Entry Point for the Adaptive Encrypted Steganographic Framework.
Provides a CLI for embedding and extracting secrets in images and videos.
"""

import argparse
import os
import logging
import time
import cv2
import numpy as np
from steganography_framework import config
from steganography_framework import encryption
from steganography_framework import hybrid_engine
from steganography_framework import video_steganography
from steganography_framework import extraction
from steganography_framework import evaluation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Adaptive Encrypted Steganography Framework")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Embed Command
    embed_parser = subparsers.add_parser("embed", help="Embed secret into cover")
    embed_parser.add_argument("--cover", required=True, help="Path to cover image or video")
    embed_parser.add_argument("--secret", required=True, help="Secret message (string) or path to secret file")
    embed_parser.add_argument("--output", required=True, help="Path for stego output")
    embed_parser.add_argument("--key", help="256-bit key (hex). If omitted, one will be generated.")
    embed_parser.add_argument("--nonce", help="96-bit nonce (hex). If omitted, one will be generated.")
    embed_parser.add_argument(
        "--video-frame-step",
        type=int,
        default=0,
        help="Frame interval for video embedding (0 = auto). Higher values improve speed."
    )
    embed_parser.add_argument("--evaluate", action="store_true", help="Run evaluation after embedding")

    # Extract Command
    extract_parser = subparsers.add_parser("extract", help="Extract secret from stego")
    extract_parser.add_argument("--stego", required=True, help="Path to stego image or video")
    extract_parser.add_argument("--key", required=True, help="256-bit key (hex)")
    extract_parser.add_argument("--nonce", required=True, help="96-bit nonce (hex)")
    extract_parser.add_argument("--output", help="Path to save extracted secret (optional)")
    extract_parser.add_argument("--cover", help="Path to original cover (for visualization)")
    extract_parser.add_argument("--secret", help="Path to original secret (for visualization)")
    extract_parser.add_argument("--visualize", action="store_true", help="Generate dashboard and comparison grid after extraction")

    args = parser.parse_args()

    if args.command == "embed":
        handle_embed(args)
    elif args.command == "extract":
        handle_extract(args)
    else:
        parser.print_help()

def handle_embed(args):
    # 1. Prepare Key and Nonce
    if args.key:
        key = bytes.fromhex(args.key)
    else:
        key, _ = encryption.generate_key_and_nonce()
        logger.info(f"Generated Key (hex): {key.hex()}")

    if args.nonce:
        nonce = bytes.fromhex(args.nonce)
    else:
        _, nonce = encryption.generate_key_and_nonce()
        logger.info(f"Generated Nonce (hex): {nonce.hex()}")

    # 2. Prepare Secret Payload
    if os.path.exists(args.secret):
        with open(args.secret, "rb") as f:
            secret_bytes = f.read()
        eval_label = os.path.basename(args.secret)
    else:
        secret_bytes = args.secret.encode()
        eval_label = os.path.basename(args.output)

    # 4. Embed based on file type
    is_video = args.cover.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    
    if is_video:
        logger.info("Processing Video Embedding...")
        embed_start = time.time()
        video_plain_payload = video_steganography.pack_video_secret(secret_bytes)
        encrypted_payload = encryption.encrypt_payload(video_plain_payload, key, nonce)
        video_steganography.embed_video(
            args.cover,
            encrypted_payload,
            args.output,
            frame_step=args.video_frame_step
        )
        embed_time_sec = time.time() - embed_start

        if args.evaluate:
            extracted_bytes = extraction.extract_from_video(args.output, key, nonce)
            metrics = evaluation.evaluate_video(
                args.cover,
                args.output,
                secret_bytes,
                extracted_bytes
            )
            metrics["embed_time_sec"] = float(embed_time_sec)
            metrics["throughput_bytes_per_sec"] = float(len(secret_bytes) / max(embed_time_sec, 1e-9))
            metrics["frame_step_requested"] = int(args.video_frame_step)
            logger.info(f"Video Evaluation Metrics: {metrics}")
            evaluation.save_video_metrics(metrics, eval_label)
            evaluation.plot_video_results()
    else:
        logger.info("Processing Image Embedding...")
        embed_start = time.time()
        encrypted_payload = encryption.encrypt_payload(secret_bytes, key, nonce)
        cover_img = cv2.imread(args.cover)
        if cover_img is None:
            logger.error("Failed to load cover image.")
            return

        stego_img, cmap, bits_used = hybrid_engine.embed_hybrid(cover_img, encrypted_payload)
        cv2.imwrite(args.output, stego_img)
        logger.info(f"Stego image saved to {args.output}")

        if args.evaluate:
            # Simple extract and compare
            extracted_enc = hybrid_engine.extract_hybrid(stego_img, cmap)
            extracted_bytes = encryption.decrypt_payload(extracted_enc, key, nonce)
            
            metrics = evaluation.evaluate_image(cover_img, stego_img, secret_bytes, extracted_bytes)
            embed_time_sec = time.time() - embed_start
            metrics["embed_time_sec"] = float(embed_time_sec)
            metrics["throughput_bytes_per_sec"] = float(len(secret_bytes) / max(embed_time_sec, 1e-9))
            logger.info(f"Evaluation Metrics: {metrics}")
            
            evaluation.save_image_metrics(metrics, eval_label)
            evaluation.plot_image_results()

def handle_extract(args):
    key = bytes.fromhex(args.key)
    nonce = bytes.fromhex(args.nonce)

    is_video = args.stego.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    
    if is_video:
        logger.info("Extracting from Video...")
        decrypted = extraction.extract_from_video(args.stego, key, nonce)
    else:
        logger.info("Extracting from Image...")
        decrypted = extraction.extract_from_image(args.stego, key, nonce)

    if decrypted:
        if args.output:
            with open(args.output, "wb") as f:
                f.write(decrypted)
            logger.info(f"Extracted secret saved to {args.output}")
            
            # Post-extraction visualization if requested
            if getattr(args, "visualize", False) and args.cover and args.secret:
                logger.info("Generating Final Visualization Dashboard and Comparison Grid...")
                import json
                psnr, ssim, ber = 0.0, 0.0, 0.0
                hist_file = evaluation.VIDEO_HISTORY_FILE if is_video else evaluation.IMAGE_HISTORY_FILE
                try:
                    with open(hist_file, "r") as f:
                        history = json.load(f)
                        if history:
                            last_met = history[-1]["metrics"]
                            psnr = last_met.get("psnr", last_met.get("avg_psnr", 0.0))
                            ssim = last_met.get("ssim", last_met.get("avg_ssim", 0.0))
                            ber = last_met.get("ber", last_met.get("avg_ber", 0.0))
                except Exception as e:
                    logger.warning(f"Could not read metrics history for visualization: {e}")
                
                # Plot the overall evaluation dashboard
                evaluation.plot_report_dashboard()
                
                # Plot the 2x2 physical grid via the visualizer script
                grid_out = os.path.join(evaluation.RESULTS_DIR, "comparison_grid.png")
                cmd = f'python steganography_framework/visualize_comparison.py --cover "{args.cover}" --secret "{args.secret}" --stego "{args.stego}" --extracted "{args.output}" --psnr {psnr} --ssim {ssim} --ber {ber} --output "{grid_out}"'
                os.system(cmd)
                logger.info(f"Done! ✅ Results saved to {evaluation.RESULTS_DIR}")
        else:
            # Only print if small and likely text
            if len(decrypted) < 1000:
                try:
                    print(f"Extracted Secret: {decrypted.decode()}")
                except UnicodeDecodeError:
                    print("Extracted secret is binary data. Use --output to save it.")
            else:
                print(f"Extracted {len(decrypted)} bytes of binary data. Use --output to save it.")
    else:
        logger.error("Extraction failed.")

if __name__ == "__main__":
    main()
