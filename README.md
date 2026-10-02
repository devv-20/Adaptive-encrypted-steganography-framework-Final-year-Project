# Adaptive Encrypted Steganographic Framework

A Python command-line project for embedding encrypted image or video payloads in cover media and extracting them later. It includes image/video evaluation and visualization utilities.

## Requirements

- Python 3.10 or later
- Packages listed in `requirements.txt`
- FFmpeg for video workflows. The bundled FFmpeg directory is intentionally excluded from this repository; install FFmpeg separately and update `FFMPEG_PATH` in `steganography_framework/config.py` if needed.

## Setup

From the project root, install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Create `inputs/covers`, `inputs/secrets`, `outputs/stego_images`, `outputs/extracted`, and `outputs/results` locally, then place media you own or are authorized to use in the input folders. Input media and generated outputs are ignored by Git and are not part of this repository.

## Usage

Run commands from the project root. To embed an image secret, allowing the program to generate an encryption key and nonce:

```powershell
python -m steganography_framework.main embed --cover "inputs/covers/cover.jpg" --secret "inputs/secrets/secret.jpg" --output "outputs/stego_images/stego.png"
```

Save the generated key and nonce securely and do not commit or share them. To extract the payload, supply those values locally:

```powershell
python -m steganography_framework.main extract --stego "outputs/stego_images/stego.png" --key "YOUR_KEY_HEX" --nonce "YOUR_NONCE_HEX" --output "outputs/extracted/recovered.jpg"
```

Use `--evaluate` when embedding to generate evaluation results. See `DEMO_COMMANDS.txt` for additional examples.

## Notes

- This repository intentionally excludes private input media, generated stego/extracted media, local results, the downloaded FFmpeg distribution, and the local research PDF/text extraction.
- Review the licenses and permissions for any third-party media or tools before publishing or redistributing them.