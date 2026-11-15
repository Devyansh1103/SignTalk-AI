#!/usr/bin/env python3
"""
SignTalk AI: export the gloss -> sentence Transformer to ONNX for the browser build.
Writes to frontend/public/models/:
  gloss_translator.onnx   (src [1,12] int64, tgt [1,24] int64, both <pad>ded) -> probabilities [1,24,V];
                          row k is the next word after the first k+1 target tokens
  gloss_translator.json   source / target vocabularies, decoding length and confidence gate
                          (the model answers <unk> for sign sequences that follow no pattern)
Checks greedy decoding with onnxruntime against PyTorch on sentences from the corpus.

Usage:
  python scripts/export_gloss_onnx.py
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch

from src.models.gloss_translator import GlossTranslator, NextTokenProbs, BOS, EOS, UNK, LANG_TOKENS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="experiments/gloss_translator/best.pt")
    ap.add_argument("--out", default="frontend/public/models/gloss_translator.onnx")
    ap.add_argument("--max-len", type=int, default=24, help="target positions, <bos> included")
    ap.add_argument("--max-src", type=int, default=12, help="source positions: language tag + up to 11 signs")
    ap.add_argument("--min-confidence", type=float, default=0.5)
    a = ap.parse_args()

    ckpt = torch.load(a.checkpoint, map_location="cpu")
    src_itos, tgt_itos = ckpt["src_itos"], ckpt["tgt_itos"]
    model = GlossTranslator(len(src_itos), len(tgt_itos), **ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.eval()
    wrapped = NextTokenProbs(model).eval()

    src = torch.zeros(1, a.max_src, dtype=torch.long); tgt = torch.zeros(1, a.max_len, dtype=torch.long)
    src[0, :3] = torch.tensor([1, 5, 6]); tgt[0, :2] = torch.tensor([1, 9])
    torch.onnx.export(wrapped, (src, tgt), a.out, input_names=["src", "tgt"], output_names=["probs"],
                      opset_version=17, do_constant_folding=True, dynamo=False)
    print(f"[OK] wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")

    # Parity: greedy decoding in onnxruntime vs PyTorch, both languages
    import onnxruntime as ort
    sess = ort.InferenceSession(a.out)
    s2i = {t: i for i, t in enumerate(src_itos)}
    bos, eos = tgt_itos.index(BOS), tgt_itos.index(EOS)
    glosses = [["i", "mother", "sick"], ["hello", "teacher"], ["i", "car", "new"], ["today", "monday"],
               ["book", "table"], ["she", "doctor"], ["yesterday", "we", "school"], ["cow"]]
    same = 0
    for g in glosses:
        for lang in ("en", "hi"):
            ids = [s2i[LANG_TOKENS[lang]]] + [s2i[x] for x in g]
            ref, _ = model.greedy(torch.tensor([ids]), bos, eos, a.max_len - 1)
            src_np = np.zeros((1, a.max_src), dtype=np.int64); src_np[0, :len(ids)] = ids
            ys = [bos]
            while len(ys) < a.max_len:
                tgt_np = np.zeros((1, a.max_len), dtype=np.int64); tgt_np[0, :len(ys)] = ys
                p = sess.run(None, {"src": src_np, "tgt": tgt_np})[0][0, len(ys) - 1]
                nxt = int(p.argmax())
                if nxt == eos:
                    break
                ys.append(nxt)
            same += ys[1:] == ref
            print(f"  {' '.join(g).upper():25s} [{lang}] {' '.join(tgt_itos[i] for i in ys[1:])}")
    print(f"[CHECK] onnxruntime == PyTorch greedy on {same}/{2 * len(glosses)} sentences")

    meta = {"src": src_itos, "tgt": tgt_itos, "bos": bos, "eos": eos, "unk": tgt_itos.index(UNK),
            "lang_tokens": LANG_TOKENS,
            "max_len": a.max_len, "max_src": a.max_src, "min_confidence": a.min_confidence}
    meta_path = os.path.join(os.path.dirname(a.out), "gloss_translator.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    print(f"[OK] wrote {meta_path}")


if __name__ == "__main__":
    main()
