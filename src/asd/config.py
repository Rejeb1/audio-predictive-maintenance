import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Large files live outside the repo. On Colab: export MIMII_ROOT=/content/mimii
DATA = Path(os.environ.get("MIMII_ROOT", "D:/mimii"))
CACHE = DATA / "cache"
# Overridable so that Colab can write straight to Google Drive.
MODELS = Path(os.environ.get("MIMII_MODELS", DATA / "models"))
RESULTS = Path(os.environ.get("MIMII_RESULTS", ROOT / "results"))

SR = 16000
CLIP_SECONDS = 10
N_FFT, HOP, N_MELS = 1024, 512, 128
N_FRAMES = 1 + SR * CLIP_SECONDS // HOP  # 313
CONTEXT = 5                # consecutive frames per dense-AE input vector
PATCH, PATCH_HOP = 64, 32  # conv-AE patch width / inference stride
SEED = 42
VAL_FRACTION = 0.1         # normal clips held out to set the alert threshold
MACHINES = ("pump", "fan")
