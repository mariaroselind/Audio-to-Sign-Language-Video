import os
import zipfile
import subprocess

os.makedirs("data/ravdess", exist_ok=True)
os.makedirs("data/crema",   exist_ok=True)
os.makedirs("data/tess",    exist_ok=True)
os.makedirs("data/savee",   exist_ok=True)

# ── RAVDESS ────────────────────────────────────────────────────────────────
print("Downloading RAVDESS...")
os.system('wget -q "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip" -O ravdess.zip')
with zipfile.ZipFile("ravdess.zip", "r") as z:
    z.extractall("data/ravdess")
print("RAVDESS done!")

# ── CREMA-D ────────────────────────────────────────────────────────────────
print("Downloading CREMA-D...")
os.system("git clone --depth=1 -q https://github.com/CheyneyComputerScience/CREMA-D.git data/crema_repo")
os.system("cp data/crema_repo/AudioWAV/*.wav data/crema/ 2>/dev/null || true")
print("CREMA-D done!")

# ── TESS ───────────────────────────────────────────────────────────────────
print("Downloading TESS...")
os.system('pip install gdown -q')
os.system('gdown --fuzzy "https://drive.google.com/uc?id=1wWsrN2Ep7x6lWqOXfr4rpKGYrJhWc8z6" -O tess.zip 2>/dev/null || echo "TESS: use Kaggle download (instructions in README)"')
print("TESS attempted!")

# ── SAVEE ──────────────────────────────────────────────────────────────────
print("Downloading SAVEE...")
os.system('gdown --fuzzy "https://drive.google.com/uc?id=1Ps0QLMZQ4bHMPp4OBcAuBHEMzKNpVzV1" -O savee.zip 2>/dev/null || echo "SAVEE: use Kaggle download (instructions in README)"')
print("All downloads attempted!")