#!/usr/bin/env python3
"""
SignTalk AI: Build the class vocabulary (assets/vocabularies/signtalk_vocab.json)
from the class folders found in your INCLUDE dataset.

Every folder like  <Category>/<N. Word>/  that contains videos becomes one class.
Run this BEFORE convert_include_mvp10.py (it decides how many classes the model has).

Usage:
  python scripts/build_vocabulary.py --include-root "D:\\path\\to\\dataset"
  python scripts/build_vocabulary.py --include-root "..." --min-clips 8
  python scripts/build_vocabulary.py --default10       # reset to the original 10-class vocabulary
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.hindi_lexicon import HINDI_LEXICON

OUT = "assets/vocabularies/signtalk_vocab.json"
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv")
DEFAULT10 = [("hello", "HELLO", "Hello"), ("thankyou", "THANK_YOU", "Thank you"), ("good", "GOOD", "Good"),
             ("happy", "HAPPY", "Happy"), ("monday", "MONDAY", "Monday"), ("car", "CAR", "Car"),
             ("bird", "BIRD", "Bird"), ("house", "HOUSE", "House"), ("time", "TIME", "Time"),
             ("teacher", "TEACHER", "Teacher")]


def clean_name(folder):
    """'55. Thank you' -> 'Thank you'"""
    return re.sub(r"^\s*\d+\s*[.\-_)]\s*", "", folder).strip()


def norm_label(folder):
    """'55. Thank you' -> 'thankyou' (must match scripts/convert_include_mvp10.py)."""
    return re.sub(r"[^a-z]", "", clean_name(folder).lower())


def folder_number(folder):
    m = re.match(r"^\s*(\d+)", folder)
    return int(m.group(1)) if m else 10 ** 6


def make_gloss(name):
    return re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")


def build_json(entries, name="signtalk_vocab"):
    """entries: list of (label, gloss, english). Class ids follow list order."""
    special = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
    token_to_id, id_to_token = dict(special), {str(v): k for k, v in special.items()}
    d = {"vocabulary_name": name, "version": "2.0.0", "language": "ISL", "num_classes": len(entries),
         "special_tokens": special, "class_id_to_token_id": {}, "class_id_to_gloss": {},
         "class_id_to_translation": {}, "class_id_to_label": {}, "class_id_to_hindi": {}}
    for i, (label, gloss, eng) in enumerate(entries):
        tok = 4 + i
        token_to_id[gloss] = tok
        id_to_token[str(tok)] = gloss
        d["class_id_to_token_id"][str(i)] = tok
        d["class_id_to_gloss"][str(i)] = gloss
        d["class_id_to_translation"][str(i)] = eng
        d["class_id_to_label"][str(i)] = label
        d["class_id_to_hindi"][str(i)] = HINDI_LEXICON.get(label, eng)
    d["token_to_id"], d["id_to_token"] = token_to_id, id_to_token
    return d


def discover(include_root, min_clips):
    found = {}  # label -> dict(folder, path, cat, n)
    for dirpath, _, files in os.walk(include_root):
        n = sum(f.lower().endswith(VIDEO_EXT) for f in files)
        if not n:
            continue
        folder = os.path.basename(dirpath)
        label = norm_label(folder)
        if not label:
            continue
        cat = os.path.basename(os.path.dirname(dirpath))
        if label in found:
            print(f"[WARN] duplicate class '{label}': {dirpath} ignored (using {found[label]['path']})")
            continue
        found[label] = dict(folder=folder, path=dirpath, cat=cat, n=n)
    keep, skipped = [], []
    for label, v in found.items():
        (keep if v["n"] >= min_clips else skipped).append((label, v))
    keep.sort(key=lambda lv: (lv[1]["cat"].lower(), folder_number(lv[1]["folder"]), lv[0]))
    return keep, skipped


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-root", default=os.path.join("dataset", "INCLUDE"))
    ap.add_argument("--min-clips", type=int, default=6, help="classes with fewer clips are skipped")
    ap.add_argument("--default10", action="store_true")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    if a.default10:
        entries = list(DEFAULT10)
    else:
        keep, skipped = discover(a.include_root, a.min_clips)
        if not keep:
            sys.exit(f"[ERROR] {a.include_root} me koi class folder (videos ke saath) nahi mila.")
        entries = [(l, make_gloss(clean_name(v["folder"])), clean_name(v["folder"])[:1].upper() + clean_name(v["folder"])[1:])
                   for l, v in keep]
        print(f"\n{'ID':>3}  {'label':14s} {'category':22s} clips  hindi")
        for i, (l, v) in enumerate(keep):
            print(f"{i:3d}  {l:14s} {v['cat'][:22]:22s} {v['n']:5d}  {HINDI_LEXICON.get(l, '(english fallback)')}")
        if skipped:
            print(f"\n[SKIPPED: <{a.min_clips} clips] " + ", ".join(f"{l}({v['n']})" for l, v in skipped))
        no_hi = [l for l, _, _ in entries if l not in HINDI_LEXICON]
        if no_hi:
            print(f"\n[INFO] Hindi missing (English use hoga): {no_hi}  -> src/data/hindi_lexicon.py me add kar sakte ho")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(build_json(entries), f, ensure_ascii=False, indent=2)
    print(f"\nWrote {a.out}: {len(entries)} classes")
    print("NOTE: classes badalne par model dobara train karna padega (purana checkpoint kaam nahi karega).")
