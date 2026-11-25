"""Build kaggle/train_kaggle.ipynb (self-contained: the source code is embedded)."""
import base64
import io
import json
import zipfile

from make_colab import ROOT, code, md

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for d in ("src", "scripts", "tests"):
        for p in sorted((ROOT / d).rglob("*.py")):
            if not p.name.startswith("make_"):
                z.write(p, p.relative_to(ROOT).as_posix())
B64 = base64.b64encode(buf.getvalue()).decode()

cells = [
    md("# Maintenance prédictive par l'audio - entraînement (Kaggle)\n\n"
       "Panneau *Settings* : **Accelerator = GPU T4 x2**, **Internet = On**. "
       "Puis **Save Version > Save & Run All (Commit)** : le calcul continue en arrière-plan.\n\n"
       "Pour reprendre une exécution incomplète : *Add Input* > sortie de ce notebook, puis relancer."),
    code("import os, torch\n"
         "assert torch.cuda.is_available(), 'Choisir GPU T4 x2'\n"
         "assert hasattr(torch.nn.utils, 'get_total_norm'), 'PyTorch >= 2.6 requis'\n"
         "torch.ones(2, device='cuda').sum().item()  # echoue si le GPU n'est pas supporte\n"
         "print(torch.cuda.get_device_name(), torch.__version__)\n"
         "OUT, DATA = '/kaggle/working', '/tmp/mimii'\n"
         "os.environ.update(MIMII_ROOT=DATA, MIMII_MODELS=f'{OUT}/models', MIMII_RESULTS=f'{OUT}/results',\n"
         "                  CUDA_VISIBLE_DEVICES='0')\n"
         "os.makedirs(f'{DATA}/dcase', exist_ok=True)"),
    code("import base64, glob, io, shutil, sys, subprocess, zipfile\n"
         f"CODE = '{B64}'\n"
         "shutil.rmtree('/tmp/project', ignore_errors=True)\n"
         "zipfile.ZipFile(io.BytesIO(base64.b64decode(CODE))).extractall('/tmp/project')\n"
         "os.chdir('/tmp/project')\n"
         "subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'librosa', 'soundfile', 'scikit-learn', 'pytest'], check=True)\n"
         "subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests'], check=True)\n"
         "# reprise : recopier les resultats d'une version precedente ajoutee en entree\n"
         "for z in glob.glob('/kaggle/input/**/training_results.zip', recursive=True):\n"
         "    zipfile.ZipFile(z).extractall(OUT); print('reprise depuis', z)"),
    md("## Données (Zenodo, sommes MD5 vérifiées)"),
    code("import hashlib\n\n"
         "def get(url, dest, md5):\n"
         "    subprocess.run(['curl', '-fL', '--retry', '10', '--retry-all-errors', '-sS', '-o', dest, url], check=True)\n"
         "    h = hashlib.md5()\n"
         "    with open(dest, 'rb') as f:\n"
         "        for b in iter(lambda: f.read(1 << 24), b''): h.update(b)\n"
         "    assert h.hexdigest() == md5, f'MD5 incorrect : {dest}'\n"
         "    print('ok', dest, flush=True)\n\n"
         "Z = 'https://zenodo.org/records'\n"
         "os.makedirs(f'{DATA}/panns', exist_ok=True)\n"
         "get(f'{Z}/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1', f'{DATA}/panns/Cnn14_16k.pth',\n"
         "    '362fc5ff18f1d6ad2f6d464b45893f2c')\n"
         "for m, h in (('pump', '90e7091ef722b7238a7f1009365779cd'), ('fan', '649bdfc06263ae7a838963f43b6641e6')):\n"
         "    zp = f'/tmp/dev_data_{m}.zip'\n"
         "    get(f'{Z}/3678171/files/dev_data_{m}.zip?download=1', zp, h)\n"
         "    zipfile.ZipFile(zp).extractall(f'{DATA}/dcase'); os.remove(zp)\n"
         "counts = {(m, s): len(os.listdir(f'{DATA}/dcase/{m}/{s}')) for m in ('pump', 'fan') for s in ('train', 'test')}\n"
         "print(counts)\n"
         "assert counts == {('pump', 'train'): 3349, ('pump', 'test'): 856, ('fan', 'train'): 3675, ('fan', 'test'): 1875}"),
    md("## Entraînements\n\nUne erreur n'arrête pas le notebook, pour que les résultats obtenus soient toujours sauvegardés."),
    code("def save():\n"
         "    with zipfile.ZipFile(f'{OUT}/training_results.zip', 'w', zipfile.ZIP_DEFLATED) as z:\n"
         "        for d in ('results', 'models'):\n"
         "            for p in glob.glob(f'{OUT}/{d}/**/*', recursive=True):\n"
         "                if os.path.isfile(p): z.write(p, os.path.relpath(p, OUT))\n\n"
         "def sh(*a):\n"
         "    p = subprocess.Popen([sys.executable, '-u', *a], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n"
         "    for line in p.stdout: print(line, end='', flush=True)\n"
         "    return p.wait()\n\n"
         "print('code retour', sh('scripts/run_all.py', '--datasets', 'dcase')); save()"),
    md("## Étude du bruit (sous-ensemble MIMII -6, 0, +6 dB)"),
    code("todo = [1 for d in ('mimii_-6dB', 'mimii_0dB', 'mimii_6dB') for m in ('pump', 'fan')\n"
         "        for k in ('panns', 'dense', 'conv') if not os.path.exists(f'{OUT}/results/scores/{d}_{m}_{k}.csv')]\n"
         "rc = sh('scripts/download_mimii_subset.py') if todo else -1  # rien a faire : pas de telechargement\n"
         "n = len(glob.glob(f'{DATA}/mimii_*dB/*/id_*/*/*.wav'))\n"
         "print('clips MIMII :', n, '/ 6000')\n"
         "if rc == 0 and n == 6000:\n"
         "    print('code retour', sh('scripts/run_all.py'))\n"
         "else:\n"
         "    print('MIMII deja termine' if not todo else 'MIMII incomplet : etude du bruit non lancee')\n"
         "save()"),
    code("done = {os.path.basename(p)[:-4] for p in glob.glob(f'{OUT}/results/scores/*.csv')}\n"
         "want = {f'{d}_{m}_{k}' for d in ('dcase', 'mimii_-6dB', 'mimii_0dB', 'mimii_6dB')\n"
         "        for m in ('pump', 'fan') for k in ('panns', 'dense', 'conv')}\n"
         "print(f'{len(done & want)}/24 termines ; manquants :', sorted(want - done))\n"
         "print('Telecharger training_results.zip dans l onglet Output')"),
]
for i, c in enumerate(cells):
    c["id"] = f"cell{i}"

if __name__ == "__main__":
    out = ROOT / "kaggle"
    out.mkdir(exist_ok=True)
    nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
          "metadata": {"language_info": {"name": "python"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True,
                                  "isInternetEnabled": True, "language": "python", "sourceType": "notebook"}}}
    (out / "train_kaggle.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
