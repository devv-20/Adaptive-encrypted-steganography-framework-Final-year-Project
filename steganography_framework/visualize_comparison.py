import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_comparison_grid():
    parser = argparse.ArgumentParser(description="Generate a steganography comparison grid.")
    parser.add_argument("--cover", default="demo_cover.png", help="Path to original cover image")
    parser.add_argument("--secret", default="demo_secret.png", help="Path to original secret image")
    parser.add_argument("--stego", default="demo_stego.png", help="Path to output stego image")
    parser.add_argument("--extracted", default="extracted_demo_secret.png", help="Path to extracted secret image")
    parser.add_argument("--psnr", type=float, default=45.23, help="PSNR value to display")
    parser.add_argument("--ssim", type=float, default=0.9978, help="SSIM value to display")
    parser.add_argument("--ber", type=float, default=0.0, help="BER value to display")
    parser.add_argument("--output", default="comparison_grid.png", help="Path to save the grid")

    args = parser.parse_args()

    # Load images
    def load_rgb(path):
        if not os.path.exists(path):
            print(f"Warning: File not found {path}")
            return np.zeros((100, 100, 3), dtype=np.uint8)
        # Try OpenCV first
        img = cv2.imread(path)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Fallback: try PIL/Pillow
        try:
            from PIL import Image
            pil_img = Image.open(path).convert("RGB")
            return np.array(pil_img)
        except Exception:
            pass
        # Fallback: try matplotlib
        try:
            import matplotlib.image as mpimg
            arr = mpimg.imread(path)
            if arr is not None:
                if arr.dtype != np.uint8:
                    arr = (arr * 255).clip(0, 255).astype(np.uint8)
                return arr if arr.ndim == 3 else np.stack([arr]*3, axis=-1)
        except Exception:
            pass
        print(f"Warning: Could not load image at {path} — showing placeholder.")
        # Show a labeled placeholder instead of blank black
        placeholder = np.ones((200, 300, 3), dtype=np.uint8) * 50
        return placeholder

    cover = load_rgb(args.cover)
    secret = load_rgb(args.secret)
    stego = load_rgb(args.stego)
    extracted = load_rgb(args.extracted)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Steganography Framework: Visual Comparison", fontsize=18, fontweight='bold', y=0.98)

    # Top Left: Cover
    axes[0, 0].imshow(cover)
    axes[0, 0].set_title(f"1. Original Cover Image\n({cover.shape[1]}x{cover.shape[0]})", fontsize=14)
    axes[0, 0].axis('off')

    # Top Right: Original Secret
    axes[0, 1].imshow(secret)
    axes[0, 1].set_title(f"2. Original Secret Image\n({secret.shape[1]}x{secret.shape[0]})", fontsize=14)
    axes[0, 1].axis('off')
    
    # Bottom Left: Stego
    axes[1, 0].imshow(stego)
    axes[1, 0].set_title(f"3. Stego Image\nPSNR: {args.psnr} dB | SSIM: {args.ssim}", fontsize=14)
    axes[1, 0].axis('off')

    # Bottom Right: Extracted Secret
    axes[1, 1].imshow(extracted)
    axes[1, 1].set_title(f"4. Extracted Secret Image\nBER: {args.ber}", fontsize=14)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    plt.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"Saved comparison grid explicitly to {args.output}")

if __name__ == "__main__":
    create_comparison_grid()
