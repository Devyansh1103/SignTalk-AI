#!/usr/bin/env python3
"""
SignTalk AI: pack everything ST-GCN training needs into one zip for Google Colab.

Includes code (src, scripts, configs), the vocabulary, the train/val/test manifests and
only the sequence .npz files those manifests reference. Raw videos are not needed.

Usage:
  python scripts/pack_for_colab.py            -> signtalk_colab.zip in the project root
Then upload the zip to Google Drive: MyDrive/SignTalk/signtalk_colab.zip
and open notebooks/train_stgcn_colab.ipynb in Colab.
"""
import os, sys, zipfile
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "signtalk_colab.zip")
CODE_DIRS = ["src", "scripts", "configs", "assets"]
SPLITS = ["train", "val", "test"]


def main():
    os.chdir(ROOT)
    files = []
    for d in CODE_DIRS:
        for base, dirs, names in os.walk(d):
            dirs[:] = [x for x in dirs if x != "__pycache__"]
            files += [os.path.join(base, n) for n in names if not n.endswith(".pyc")]
    missing = []
    for split in SPLITS:
        m = f"data/manifests/{split}.csv"
        files.append(m)
        for p in pd.read_csv(m)["file_path"]:
            (files if os.path.exists(p) else missing).append(p)
    if missing:
        print(f"[ERROR] {len(missing)} manifest files missing on disk, e.g. {missing[:3]}")
        sys.exit(1)

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(set(files)):
            z.write(f, f.replace(os.sep, "/"))
    print(f"[OK] {len(set(files))} files -> {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")
    print("Ab ise Google Drive me MyDrive/SignTalk/ folder me upload karo.")


if __name__ == "__main__":
    main()
