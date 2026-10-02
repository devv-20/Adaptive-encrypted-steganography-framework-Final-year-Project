"""
Evaluation module for image and video steganography.
Generates separate metrics histories and plots for image/video runs.
"""

import cv2
import numpy as np
import time
import logging
import matplotlib.pyplot as plt
import json
import os
import re
import math
from typing import Dict, List
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join('outputs', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

IMAGE_HISTORY_FILE = os.path.join(RESULTS_DIR, 'image_metrics_history.json')
VIDEO_HISTORY_FILE = os.path.join(RESULTS_DIR, 'video_metrics_history.json')
LEGACY_HISTORY_FILE = os.path.join(RESULTS_DIR, 'metrics_history.json')
MAX_HISTORY = 20
COMPARISON_RUNS = 8  # compare against last 6-8 runs
VIDEO_SAMPLE_MAX_FRAMES = 40


def _ber(original_secret: bytes, extracted_secret: bytes) -> float:
    orig_arr = np.unpackbits(np.frombuffer(original_secret, dtype=np.uint8))
    ext_arr = np.unpackbits(np.frombuffer(extracted_secret, dtype=np.uint8))
    max_len = max(len(orig_arr), len(ext_arr), 1)
    if len(orig_arr) < max_len:
        orig_arr = np.pad(orig_arr, (0, max_len - len(orig_arr)))
    if len(ext_arr) < max_len:
        ext_arr = np.pad(ext_arr, (0, max_len - len(ext_arr)))
    return float(np.count_nonzero(orig_arr != ext_arr) / max_len)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def _safe_float(val: float, fallback: float = 99.0) -> float:
    return float(val) if math.isfinite(val) else fallback


def _format_label(label: str) -> str:
    base = os.path.splitext(os.path.basename(label))[0]
    base = re.sub(r"^(run_|image_run_|video_run_)", "", base)
    base = re.sub(r"[_\\-]+", " ", base).strip()
    return base[:28] if base else "unnamed"


def evaluate_image(
    cover: np.ndarray,
    stego: np.ndarray,
    original_secret: bytes,
    extracted_secret: bytes
) -> Dict[str, float]:
    """Calculates image-specific steganography metrics."""
    metrics = {
        "psnr": _safe_float(psnr_metric(cover, stego)),
        "ssim": float(ssim_metric(cover, stego, channel_axis=2 if len(cover.shape) == 3 else None)),
        "mse": _mse(cover, stego),
        "ber": _ber(original_secret, extracted_secret),
        "bpp": float((len(original_secret) * 8) / max(cover.shape[0] * cover.shape[1], 1)),
        "payload_bytes": int(len(original_secret)),
        "extracted_bytes": int(len(extracted_secret)),
        "byte_match": bool(original_secret == extracted_secret),
    }
    return metrics


def evaluate_video(
    cover_video_path: str,
    stego_video_path: str,
    original_secret: bytes,
    extracted_secret: bytes
) -> Dict[str, float]:
    """Calculates video-specific metrics with frame-wise sampling."""
    cover_cap = cv2.VideoCapture(cover_video_path)
    stego_cap = cv2.VideoCapture(stego_video_path)
    if not cover_cap.isOpened() or not stego_cap.isOpened():
        raise ValueError("Could not open cover/stego video for evaluation.")

    total_frames = int(min(
        cover_cap.get(cv2.CAP_PROP_FRAME_COUNT),
        stego_cap.get(cv2.CAP_PROP_FRAME_COUNT),
    ))
    fps = float(stego_cap.get(cv2.CAP_PROP_FPS) or cover_cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(stego_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or cover_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(stego_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or cover_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    frame_step = max(1, total_frames // VIDEO_SAMPLE_MAX_FRAMES) if total_frames > 0 else 1
    psnr_values: List[float] = []
    ssim_values: List[float] = []
    mse_values: List[float] = []

    idx = 0
    sampled = 0
    while True:
        ret_c, cover_frame = cover_cap.read()
        ret_s, stego_frame = stego_cap.read()
        if not ret_c or not ret_s:
            break
        if idx % frame_step == 0:
            psnr_values.append(_safe_float(psnr_metric(cover_frame, stego_frame)))
            ssim_values.append(float(ssim_metric(cover_frame, stego_frame, channel_axis=2)))
            mse_values.append(_mse(cover_frame, stego_frame))
            sampled += 1
        idx += 1

    cover_cap.release()
    stego_cap.release()

    if not psnr_values:
        raise ValueError("No comparable frames found for video metrics.")

    pixels_per_frame = max(width * height, 1)
    metrics = {
        "frame_count": int(total_frames),
        "fps": fps,
        "resolution": f"{width}x{height}",
        "sampled_frames": int(sampled),
        "sample_step": int(frame_step),
        "avg_psnr": _safe_float(np.mean(psnr_values)),
        "min_psnr": _safe_float(np.min(psnr_values)),
        "avg_ssim": float(np.mean(ssim_values)),
        "min_ssim": float(np.min(ssim_values)),
        "avg_mse": float(np.mean(mse_values)),
        "ber": _ber(original_secret, extracted_secret),
        "payload_bytes": int(len(original_secret)),
        "extracted_bytes": int(len(extracted_secret)),
        "byte_match": bool(original_secret == extracted_secret),
        "payload_bpp": float((len(original_secret) * 8) / (pixels_per_frame * max(total_frames, 1))),
    }
    return metrics


def _read_history(history_file: str) -> List[Dict]:
    if not os.path.exists(history_file):
        return []
    try:
        with open(history_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _comparison_slice(history: List[Dict]) -> List[Dict]:
    """Returns current + last 3 runs for graph comparisons."""
    if not history:
        return []
    return history[-COMPARISON_RUNS:]


def _write_history(history_file: str, history: List[Dict]) -> None:
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)


def save_image_metrics(metrics: Dict, label: str) -> None:
    history = _read_history(IMAGE_HISTORY_FILE)
    history.append({
        "label": _format_label(label),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
    })
    history = history[-MAX_HISTORY:]
    _write_history(IMAGE_HISTORY_FILE, history)
    logger.info("Image metrics saved to %s", IMAGE_HISTORY_FILE)


def save_video_metrics(metrics: Dict, label: str) -> None:
    history = _read_history(VIDEO_HISTORY_FILE)
    history.append({
        "label": _format_label(label),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
    })
    history = history[-MAX_HISTORY:]
    _write_history(VIDEO_HISTORY_FILE, history)
    logger.info("Video metrics saved to %s", VIDEO_HISTORY_FILE)


def plot_image_results(output_path: str = None) -> None:
    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, "image_evaluation_results.png")
    history = _comparison_slice(_read_history(IMAGE_HISTORY_FILE))
    if not history:
        logger.warning("No image historical data to plot.")
        return

    labels = [h["label"] for h in history]
    psnr_vals = [h["metrics"]["psnr"] for h in history]
    ssim_vals = [h["metrics"]["ssim"] for h in history]
    ber_vals = [h["metrics"]["ber"] for h in history]
    bpp_vals = [h["metrics"]["bpp"] for h in history]

    mse_vals = [h["metrics"].get("mse", 0.0) for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    bars0 = axes[0].bar(labels, psnr_vals, color="skyblue", edgecolor="navy")
    axes[0].set_title("Image PSNR")
    axes[0].axhline(y=40, color="r", linestyle="--", linewidth=1)
    axes[0].bar_label(bars0, fmt="%.2f")

    bars1 = axes[1].bar(labels, ssim_vals, color="lightgreen", edgecolor="darkgreen")
    axes[1].set_title("Image SSIM")
    axes[1].axhline(y=0.95, color="r", linestyle="--", linewidth=1)
    axes[1].set_ylim(0, 1.05)
    axes[1].bar_label(bars1, fmt="%.4f")

    bars2 = axes[2].bar(labels, ber_vals, color="salmon", edgecolor="darkred")
    axes[2].set_title("Image BER")
    axes[2].bar_label(bars2, fmt="%.4f")

    bars3 = axes[3].bar(labels, bpp_vals, color="plum", edgecolor="purple")
    axes[3].set_title("Image Payload Capacity")
    axes[3].bar_label(bars3, fmt="%.4f")

    bars4 = axes[4].bar(labels, mse_vals, color="gold", edgecolor="darkgoldenrod")
    axes[4].set_title("Image MSE")
    axes[4].bar_label(bars4, fmt="%.2f")

    # Hide the 6th unused subplot
    axes[5].axis('off')

    for ax in axes[:5]:
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(output_path)
    logger.info("Image evaluation plots saved to %s", output_path)


def plot_video_results(output_path: str = None) -> None:
    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, "video_evaluation_results.png")
    history = _comparison_slice(_read_history(VIDEO_HISTORY_FILE))
    if not history:
        logger.warning("No video historical data to plot.")
        return

    labels = [h["label"] for h in history]
    avg_psnr_vals = [h["metrics"]["avg_psnr"] for h in history]
    avg_ssim_vals = [h["metrics"]["avg_ssim"] for h in history]
    ber_vals = [h["metrics"]["ber"] for h in history]
    payload_bpp_vals = [h["metrics"]["payload_bpp"] for h in history]

    avg_mse_vals = [h["metrics"].get("avg_mse", 0.0) for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    bars0 = axes[0].bar(labels, avg_psnr_vals, color="skyblue", edgecolor="navy")
    axes[0].set_title("Video Avg PSNR")
    axes[0].axhline(y=40, color="r", linestyle="--", linewidth=1)
    axes[0].bar_label(bars0, fmt="%.2f")

    bars1 = axes[1].bar(labels, avg_ssim_vals, color="lightgreen", edgecolor="darkgreen")
    axes[1].set_title("Video Avg SSIM")
    axes[1].axhline(y=0.95, color="r", linestyle="--", linewidth=1)
    axes[1].set_ylim(0, 1.05)
    axes[1].bar_label(bars1, fmt="%.4f")

    bars2 = axes[2].bar(labels, ber_vals, color="salmon", edgecolor="darkred")
    axes[2].set_title("Video BER")
    axes[2].bar_label(bars2, fmt="%.4f")

    bars3 = axes[3].bar(labels, payload_bpp_vals, color="plum", edgecolor="purple")
    axes[3].set_title("Video Payload Capacity")
    axes[3].bar_label(bars3, fmt="%.4f")

    bars4 = axes[4].bar(labels, avg_mse_vals, color="gold", edgecolor="darkgoldenrod")
    axes[4].set_title("Video Avg MSE")
    axes[4].bar_label(bars4, fmt="%.2f")

    # Hide the 6th unused subplot
    axes[5].axis('off')

    for ax in axes[:5]:
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(output_path)
    logger.info("Video evaluation plots saved to %s", output_path)


def plot_time_comparison(output_path: str = None) -> None:
    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, "time_comparison_results.png")
    """Plots embed time comparisons for image/video (last 4 runs each)."""
    img_hist = _comparison_slice(_read_history(IMAGE_HISTORY_FILE))
    vid_hist = _comparison_slice(_read_history(VIDEO_HISTORY_FILE))
    if not img_hist and not vid_hist:
        logger.warning("No historical data for time comparison.")
        return

    has_img = bool(img_hist)
    has_vid = bool(vid_hist)
    cols = sum([has_img, has_vid])
    if cols == 0:
        return
        
    fig, axes = plt.subplots(1, cols, figsize=(7 * cols, 5))
    if cols == 1:
        axes = [axes]
    
    idx = 0
    if has_img:
        labels = [h["label"] for h in img_hist]
        vals = [h["metrics"].get("embed_time_sec", 0.0) for h in img_hist]
        bars = axes[idx].bar(labels, vals, color="cornflowerblue", edgecolor="navy")
        axes[idx].set_title("Image Embed Time")
        axes[idx].set_ylabel("Seconds")
        axes[idx].tick_params(axis="x", rotation=15)
        axes[idx].bar_label(bars, fmt="%.2f")
        idx += 1

    if has_vid:
        labels = [h["label"] for h in vid_hist]
        vals = [h["metrics"].get("embed_time_sec", 0.0) for h in vid_hist]
        bars = axes[idx].bar(labels, vals, color="orange", edgecolor="darkorange")
        axes[idx].set_title("Video Embed Time")
        axes[idx].set_ylabel("Seconds")
        axes[idx].tick_params(axis="x", rotation=15)
        axes[idx].bar_label(bars, fmt="%.2f")

    plt.tight_layout()
    plt.savefig(output_path)
    logger.info("Time comparison plot saved to %s", output_path)


def plot_recovery_success_rate(output_path: str = None) -> None:
    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, "recovery_success_rate.png")
    """Plots recovery success rate for image/video histories."""
    img_hist = _comparison_slice(_read_history(IMAGE_HISTORY_FILE))
    vid_hist = _comparison_slice(_read_history(VIDEO_HISTORY_FILE))
    if not img_hist and not vid_hist:
        logger.warning("No historical data for recovery success rate.")
        return

    def success_rate(hist: List[Dict]) -> float:
        if not hist:
            return 0.0
        ok = sum(1 for h in hist if h["metrics"].get("byte_match", False))
        return (ok / len(hist)) * 100.0

    labels = []
    rates = []
    colors = []
    if img_hist:
        labels.append("Image")
        rates.append(success_rate(img_hist))
        colors.append("seagreen")
    if vid_hist:
        labels.append("Video")
        rates.append(success_rate(vid_hist))
        colors.append("mediumpurple")

    plt.figure(figsize=(max(4, len(labels)*3), 5))
    bars = plt.bar(labels, rates, color=colors)
    plt.title("Recovery Success Rate")
    plt.ylabel("Success (%)")
    plt.ylim(0, 105)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info("Recovery success plot saved to %s", output_path)


def plot_capacity_vs_quality(output_path: str = None) -> None:
    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, "capacity_vs_quality.png")
    """Plots payload capacity vs quality for image/video histories."""
    img_hist = _comparison_slice(_read_history(IMAGE_HISTORY_FILE))
    vid_hist = _comparison_slice(_read_history(VIDEO_HISTORY_FILE))
    if not img_hist and not vid_hist:
        logger.warning("No historical data for capacity vs quality.")
        return

    has_img = bool(img_hist)
    has_vid = bool(vid_hist)
    cols = sum([has_img, has_vid])
    if cols == 0:
        return
        
    fig, axes = plt.subplots(1, cols, figsize=(7 * cols, 5))
    if cols == 1:
        axes = [axes]
    
    idx = 0
    if has_img:
        x = [h["metrics"].get("bpp", 0.0) for h in img_hist]
        y = [h["metrics"].get("psnr", 0.0) for h in img_hist]
        labels = [h["label"] for h in img_hist]
        axes[idx].scatter(x, y, color="dodgerblue")
        axes[idx].set_title("Image Capacity vs Quality")
        axes[idx].set_xlabel("Payload Capacity (bpp)")
        axes[idx].set_ylabel("PSNR (dB)")
        for xi, yi, lb in zip(x, y, labels):
            axes[idx].annotate(lb, (xi, yi), fontsize=8, xytext=(3, 3), textcoords="offset points")
        idx += 1

    if has_vid:
        x = [h["metrics"].get("payload_bpp", 0.0) for h in vid_hist]
        y = [h["metrics"].get("avg_psnr", 0.0) for h in vid_hist]
        labels = [h["label"] for h in vid_hist]
        axes[idx].scatter(x, y, color="darkorange")
        axes[idx].set_title("Video Capacity vs Quality")
        axes[idx].set_xlabel("Payload Capacity (bpp)")
        axes[idx].set_ylabel("Avg PSNR (dB)")
        for xi, yi, lb in zip(x, y, labels):
            axes[idx].annotate(lb, (xi, yi), fontsize=8, xytext=(3, 3), textcoords="offset points")

    plt.tight_layout()
    plt.savefig(output_path)
    logger.info("Capacity vs quality plot saved to %s", output_path)


def plot_report_dashboard(output_path: str = None) -> None:
    if output_path is None:
        output_path = os.path.join(RESULTS_DIR, "evaluation_dashboard.png")
    """
    Generates one report-ready dashboard containing:
    1) image/video embed time comparison
    2) recovery success rate
    3) image capacity vs quality
    4) video capacity vs quality
    """
    img_hist = _comparison_slice(_read_history(IMAGE_HISTORY_FILE))
    vid_hist = _comparison_slice(_read_history(VIDEO_HISTORY_FILE))
    has_img = bool(img_hist)
    has_vid = bool(vid_hist)
    
    # Decide layout
    if has_img and has_vid:
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle("Steganography Evaluation Dashboard", fontsize=16, fontweight="bold")
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
    else:
        fig = plt.figure(figsize=(16, 5))
        title = "Steganography Evaluation Dashboard (Image)" if has_img else "Steganography Evaluation Dashboard (Video)"
        fig.suptitle(title, fontsize=16, fontweight="bold")
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)
        ax4 = None

    # 1) Time comparison
    if has_img and has_vid:
        x_img = np.arange(len(img_hist))
        img_vals = [h["metrics"].get("embed_time_sec", 0.0) for h in img_hist]
        ax1.bar(x_img - 0.2, img_vals, width=0.4, label="Image", color="cornflowerblue")
        x_vid = np.arange(len(vid_hist))
        vid_vals = [h["metrics"].get("embed_time_sec", 0.0) for h in vid_hist]
        ax1.bar(x_vid + 0.2, vid_vals, width=0.4, label="Video", color="orange")
        ax1.legend()
    elif has_img:
        x_img = np.arange(len(img_hist))
        img_vals = [h["metrics"].get("embed_time_sec", 0.0) for h in img_hist]
        bars = ax1.bar(x_img, img_vals, width=0.5, label="Image", color="cornflowerblue")
        ax1.set_xticks(x_img)
        ax1.set_xticklabels([h["label"] for h in img_hist], rotation=15)
        ax1.bar_label(bars, fmt="%.2f")
    elif has_vid:
        x_vid = np.arange(len(vid_hist))
        vid_vals = [h["metrics"].get("embed_time_sec", 0.0) for h in vid_hist]
        bars = ax1.bar(x_vid, vid_vals, width=0.5, label="Video", color="orange")
        ax1.set_xticks(x_vid)
        ax1.set_xticklabels([h["label"] for h in vid_hist], rotation=15)
        ax1.bar_label(bars, fmt="%.2f")
        
    ax1.set_title("Embed Time Comparison")
    ax1.set_ylabel("Seconds")

    # 2) Recovery success rate
    def _rate(hist):
        if not hist: return 0.0
        ok = sum(1 for h in hist if h["metrics"].get("byte_match", False))
        return (ok / len(hist)) * 100.0
        
    labels = []
    rates = []
    colors = []
    if has_img:
        labels.append("Image")
        rates.append(_rate(img_hist))
        colors.append("seagreen")
    if has_vid:
        labels.append("Video")
        rates.append(_rate(vid_hist))
        colors.append("mediumpurple")
        
    bars = ax2.bar(labels, rates, color=colors)
    ax2.set_title("Recovery Success Rate")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Success (%)")
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}%", ha="center")

    # 3) Capacity vs quality
    if has_img and has_vid:
        # ax3 image, ax4 video
        x = [h["metrics"].get("bpp", 0.0) for h in img_hist]
        y = [h["metrics"].get("psnr", 0.0) for h in img_hist]
        ax3.scatter(x, y, color="dodgerblue")
        for xi, yi, lb in zip(x, y, [h["label"] for h in img_hist]):
            ax3.annotate(lb, (xi, yi), fontsize=8, xytext=(3, 3), textcoords="offset points")
        ax3.set_title("Image Capacity vs Quality")
        ax3.set_xlabel("Payload Capacity (bpp)")
        ax3.set_ylabel("PSNR (dB)")
        
        x = [h["metrics"].get("payload_bpp", 0.0) for h in vid_hist]
        y = [h["metrics"].get("avg_psnr", 0.0) for h in vid_hist]
        ax4.scatter(x, y, color="darkorange")
        for xi, yi, lb in zip(x, y, [h["label"] for h in vid_hist]):
            ax4.annotate(lb, (xi, yi), fontsize=8, xytext=(3, 3), textcoords="offset points")
        ax4.set_title("Video Capacity vs Quality")
        ax4.set_xlabel("Payload Capacity (bpp)")
        ax4.set_ylabel("Avg PSNR (dB)")
    else:
        # only ax3 needed
        if has_img:
            x = [h["metrics"].get("bpp", 0.0) for h in img_hist]
            y = [h["metrics"].get("psnr", 0.0) for h in img_hist]
            ax3.scatter(x, y, color="dodgerblue")
            for xi, yi, lb in zip(x, y, [h["label"] for h in img_hist]):
                ax3.annotate(lb, (xi, yi), fontsize=8, xytext=(3, 3), textcoords="offset points")
            ax3.set_title("Capacity vs Quality")
            ax3.set_xlabel("Payload Capacity (bpp)")
            ax3.set_ylabel("PSNR (dB)")
        elif has_vid:
            x = [h["metrics"].get("payload_bpp", 0.0) for h in vid_hist]
            y = [h["metrics"].get("avg_psnr", 0.0) for h in vid_hist]
            ax3.scatter(x, y, color="darkorange")
            for xi, yi, lb in zip(x, y, [h["label"] for h in vid_hist]):
                ax3.annotate(lb, (xi, yi), fontsize=8, xytext=(3, 3), textcoords="offset points")
            ax3.set_title("Capacity vs Quality")
            ax3.set_xlabel("Payload Capacity (bpp)")
            ax3.set_ylabel("Avg PSNR (dB)")

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(output_path)
    logger.info("Evaluation dashboard saved to %s", output_path)


# Backward-compatible wrappers used by older call sites.
def calculate_metrics(cover: np.ndarray, stego: np.ndarray, original_secret: bytes, extracted_secret: bytes) -> dict:
    return evaluate_image(cover, stego, original_secret, extracted_secret)


def save_to_history(metrics: dict, label: str):
    save_image_metrics(metrics, label)
    if os.path.exists(LEGACY_HISTORY_FILE):
        legacy_history = _read_history(LEGACY_HISTORY_FILE)
    else:
        legacy_history = []
    legacy_history.append({
        "label": os.path.basename(label),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics
    })
    legacy_history = legacy_history[-MAX_HISTORY:]
    _write_history(LEGACY_HISTORY_FILE, legacy_history)


def load_history():
    return _read_history(IMAGE_HISTORY_FILE)


def plot_results():
    plot_image_results()

if __name__ == "__main__":
    print("--- Evaluation Module Hook Initialized ---")
    # Example usage placeholder
    # c = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    # s = c.copy(); s[0,0,0] ^= 1
    # m = calculate_metrics(c, s, b"abc", b"abc")
    # print(m)
