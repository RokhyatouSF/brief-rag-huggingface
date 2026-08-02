import os
import wave
import struct
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def generate_sample_wav(filename: Path):
    """Génère un fichier audio WAV de 3 secondes contant une onde sinusoïdale de test (440Hz)."""
    sample_rate = 16000
    duration = 3  # secondes
    frequency = 440.0  # Hz

    num_samples = int(sample_rate * duration)
    with wave.open(str(filename), 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(sample_rate)

        for i in range(num_samples):
            value = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * frequency * i / sample_rate))
            data = struct.pack('<h', value)
            wav_file.writeframesraw(data)

    print(f"[OK] Fichier audio de test genere: {filename}")


def generate_sample_image(filename: Path, is_damaged: bool = True):
    """Génère une image d'exemple représentant un produit avec ou sans fissure."""
    width, height = 400, 400
    color = (220, 220, 240)
    image = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(image)

    # Dessin d'un produit (boîte/smartphone)
    draw.rectangle([80, 50, 320, 350], fill=(50, 50, 50), outline=(0, 0, 0), width=4)
    draw.rectangle([100, 70, 300, 310], fill=(100, 180, 240))

    if is_damaged:
        # Dessin de fissures rouges/noires sur l'écran
        draw.line([120, 80, 220, 200], fill=(255, 0, 0), width=5)
        draw.line([220, 200, 180, 280], fill=(255, 0, 0), width=4)
        draw.line([220, 200, 290, 250], fill=(255, 0, 0), width=4)
        # Texte d'illustration
        draw.text((110, 320), "DEMO - PRODUIT CASSE", fill=(255, 0, 0))
    else:
        draw.text((110, 320), "DEMO - PRODUIT INTACT", fill=(0, 150, 0))

    image.save(filename, format="JPEG")
    print(f"[OK] Fichier image de test genere: {filename}")


def main():
    assets_dir = Path(__file__).resolve().parent.parent / "demo_assets"
    assets_dir.mkdir(exist_ok=True)

    generate_sample_wav(assets_dir / "demo_audio_casse.wav")
    generate_sample_image(assets_dir / "demo_photo_produit_casse.jpg", is_damaged=True)
    generate_sample_image(assets_dir / "demo_photo_produit_intact.jpg", is_damaged=False)

    print("\nAssets de démonstration prêts dans le dossier 'demo_assets/'. Utilise-les dans Swagger UI (/docs) !")


if __name__ == "__main__":
    main()
