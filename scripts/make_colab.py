"""Build colab/train_colab.ipynb and colab/code_bundle.zip from the current source."""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "colab"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


cells = [
    md("# Maintenance prédictive par l'audio - entraînement\n\n"
       "1. **Exécution > Modifier le type d'exécution > T4 GPU**\n"
       "2. **Exécution > Tout exécuter**, autoriser Google Drive, puis choisir `code_bundle.zip`.\n\n"
       "Les résultats vont dans `MyDrive/mimii_project`. Après une déconnexion, relancer "
       "« Tout exécuter » : les entraînements terminés sont ignorés. Espace Drive : environ 2,5 Go."),
    code("import torch\n"
         "assert torch.cuda.is_available(), 'Choisir un GPU T4 dans le type d execution'\n"
         "print(torch.cuda.get_device_name(), torch.__version__)"),
    code("import os\n"
         "from google.colab import drive\n"
         "drive.mount('/content/drive')\n"
         "DRIVE = '/content/drive/MyDrive/mimii_project'\n"
         "os.makedirs(DRIVE, exist_ok=True)\n"
         "os.environ.update(MIMII_ROOT='/content/mimii', MIMII_MODELS=f'{DRIVE}/models',\n"
         "                  MIMII_RESULTS=f'{DRIVE}/results')\n"
         "os.makedirs('/content/mimii/dcase', exist_ok=True)"),
    code("import shutil, zipfile\n"
         "bundle = f'{DRIVE}/code_bundle.zip'\n"
         "if not os.path.exists(bundle):   # pour mettre le code a jour : supprimer ce fichier du Drive\n"
         "    from google.colab import files\n"
         "    name, data = next(iter(files.upload().items()))\n"
         "    open(bundle, 'wb').write(data)\n"
         "shutil.rmtree('/content/project', ignore_errors=True)\n"
         "zipfile.ZipFile(bundle).extractall('/content/project')\n"
         "%cd /content/project\n"
         "!pip -q install librosa soundfile scikit-learn pytest\n"
         "!python -m pytest -q tests"),
    md("## Données (Zenodo, sommes MD5 vérifiées)"),
    code("import hashlib, subprocess\n\n"
         "def get(url, dest, md5):\n"
         "    def ok():\n"
         "        if not os.path.exists(dest): return False\n"
         "        h = hashlib.md5()\n"
         "        with open(dest, 'rb') as f:\n"
         "            for b in iter(lambda: f.read(1 << 24), b''): h.update(b)\n"
         "        return h.hexdigest() == md5\n"
         "    if not ok():\n"
         "        subprocess.run(['curl', '-fL', '--retry', '10', '--retry-all-errors', '-sS', '-o', dest, url], check=True)\n"
         "        assert ok(), f'MD5 incorrect : {dest}'\n"
         "    print('ok', dest)\n\n"
         "Z = 'https://zenodo.org/records'\n"
         "os.makedirs('/content/mimii/panns', exist_ok=True)\n"
         "get(f'{Z}/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1',\n"
         "    '/content/mimii/panns/Cnn14_16k.pth', '362fc5ff18f1d6ad2f6d464b45893f2c')\n"
         "for m, h in (('pump', '90e7091ef722b7238a7f1009365779cd'), ('fan', '649bdfc06263ae7a838963f43b6641e6')):\n"
         "    get(f'{Z}/3678171/files/dev_data_{m}.zip?download=1', f'/content/dev_data_{m}.zip', h)\n"
         "    subprocess.run(['unzip', '-q', '-n', f'/content/dev_data_{m}.zip', '-d', '/content/mimii/dcase'], check=True)\n"
         "counts = {(m, s): len(os.listdir(f'/content/mimii/dcase/{m}/{s}')) for m in ('pump', 'fan') for s in ('train', 'test')}\n"
         "print(counts)\n"
         "assert counts == {('pump', 'train'): 3349, ('pump', 'test'): 856, ('fan', 'train'): 3675, ('fan', 'test'): 1875}"),
    md("## DCASE : trois approches"),
    code("!python scripts/run_all.py --datasets dcase"),
    md("## Étude du bruit : sous-ensemble MIMII (-6, 0, +6 dB)\n\n"
       "Téléchargement long la première fois (environ 11 Go lus), puis archivé sur le Drive."),
    code("import glob, tarfile\n"
         "tar = f'{DRIVE}/mimii_subset.tar'\n"
         "if os.path.exists(tar):\n"
         "    tarfile.open(tar).extractall('/content/mimii', filter='data')\n"
         "!python scripts/download_mimii_subset.py\n"
         "n = len(glob.glob('/content/mimii/mimii_*dB/*/id_*/*/*.wav'))\n"
         "assert n == 6000, n\n"
         "if not os.path.exists(tar):\n"
         "    with tarfile.open(tar + '.tmp', 'w') as t:\n"
         "        for s in ('-6', '0', '6'):\n"
         "            t.add(f'/content/mimii/mimii_{s}dB', f'mimii_{s}dB')\n"
         "    os.rename(tar + '.tmp', tar)"),
    code("!python scripts/run_all.py"),
    md("## Récupérer les résultats\n\nTélécharge `colab_results.zip` (scores, historiques, modèles)."),
    code("done = {os.path.basename(p)[:-4] for p in glob.glob(f'{DRIVE}/results/scores/*.csv') if 'history' not in p}\n"
         "want = {f'{d}_{m}_{k}' for d in ('dcase', 'mimii_-6dB', 'mimii_0dB', 'mimii_6dB')\n"
         "        for m in ('pump', 'fan') for k in ('panns', 'dense', 'conv')}\n"
         "print(f'{len(done & want)}/24 termines ; manquants :', sorted(want - done))\n"
         "with zipfile.ZipFile('/content/colab_results.zip', 'w', zipfile.ZIP_DEFLATED) as z:\n"
         "    for d in ('results', 'models'):\n"
         "        for p in glob.glob(f'{DRIVE}/{d}/**/*', recursive=True):\n"
         "            if os.path.isfile(p): z.write(p, os.path.relpath(p, DRIVE))\n"
         "from google.colab import files\n"
         "files.download('/content/colab_results.zip')"),
]
for i, c in enumerate(cells):
    c["id"] = f"cell{i}"

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
          "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
    (OUT / "train_colab.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    with zipfile.ZipFile(OUT / "code_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for d in ("src", "scripts", "tests"):
            for p in sorted((ROOT / d).rglob("*.py")):
                z.write(p, p.relative_to(ROOT).as_posix())
    print([n for n in zipfile.ZipFile(OUT / "code_bundle.zip").namelist()])
