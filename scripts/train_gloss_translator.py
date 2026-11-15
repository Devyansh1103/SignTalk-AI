#!/usr/bin/env python3
"""
SignTalk AI: train the gloss -> sentence Transformer (src/models/gloss_translator.py).

Builds the synthetic corpus (src/nlp/gloss_grammar.py) for the 260 vocabulary words, holds out
5% of the gloss sequences for validation and 5% for test (never seen in training), trains one
model for English and Hindi, and reports exact-match sentence accuracy on the test glosses.
Also checks the confidence gate: random gloss sequences that follow no pattern should mostly
fall below --min-confidence, so the browser falls back to plain word-by-word text for them.

Usage:
  python -u scripts/train_gloss_translator.py
  python -u scripts/train_gloss_translator.py --sentences 60000 --epochs 30
Output:
  experiments/gloss_translator/{best.pt, vocab.json, corpus_sample.tsv, test_report.txt}
"""
import argparse, json, os, random, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn

from src.data.label_map import CANONICAL_CLASSES
from src.nlp.gloss_grammar import FIXED, GREETINGS, build_corpus, clothing_pairs
from src.models.gloss_translator import (GlossTranslator, SPECIALS, LANG_TOKENS, PAD, BOS, EOS, UNK,
                                         tokenize_sentence, detokenize)
from src.utils.reproducibility import set_seed

OUT_DIR = "experiments/gloss_translator"
# Clearer single-word English than the vocabulary display names ("You (plural)", "Street or Road")
SINGLE_EN = {"youplural": "You all", "raceethnicity": "Race", "streetorroad": "Road", "storeorshop": "Shop",
             "biglarge": "Big", "smalllittle": "Small", "trainstation": "Train station"}


def load_vocab():
    """Labels in class-id order with their English / Hindi display text."""
    classes = [CANONICAL_CLASSES[i] for i in sorted(CANONICAL_CLASSES)]
    labels = [c.label for c in classes]
    english = {c.label: SINGLE_EN.get(c.label, c.translation) for c in classes}
    hindi = {c.label: c.hindi or c.translation for c in classes}
    return labels, english, hindi


def encode(pairs, src_itos, tgt_itos):
    s2i = {t: i for i, t in enumerate(src_itos)}
    t2i = {t: i for i, t in enumerate(tgt_itos)}
    data = []
    for gloss, en, hi in pairs:
        for lang, text in (("en", en), ("hi", hi)):
            src = [s2i[LANG_TOKENS[lang]]] + [s2i[g] for g in gloss]
            tgt = [t2i[BOS]] + [t2i[t] for t in tokenize_sentence(text)] + [t2i[EOS]]
            data.append((src, tgt, gloss, lang, text))
    return data


def make_junk(labels, known, n, rng, exclude=()):
    """Random gloss sequences that follow no corpus pattern, taught to translate to <unk>.
    Lengths 2-5: also covers phrases longer than any pattern (the browser allows 10 signs)."""
    junk = set()
    while len(junk) < n:
        g = tuple(rng.sample(labels, rng.choice([2, 2, 3, 3, 4, 5])))
        if g not in known and g not in exclude:
            junk.add(g)
    return [(g, UNK, UNK) for g in sorted(junk)]


def batches(data, bs, shuffle, rng):
    idx = list(range(len(data)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [data[j] for j in idx[i:i + bs]]
        S = max(len(c[0]) for c in chunk); T = max(len(c[1]) for c in chunk)
        src = torch.tensor([c[0] + [0] * (S - len(c[0])) for c in chunk])
        tgt = torch.tensor([c[1] + [0] * (T - len(c[1])) for c in chunk])
        yield src, tgt


def decode_all(model, data, tgt_itos, device, limit=None):
    """[(gloss, lang, reference, prediction, min_prob)] by greedy decoding."""
    model.eval()
    bos, eos = tgt_itos.index(BOS), tgt_itos.index(EOS)
    out = []
    for src, _, gloss, lang, text in data[:limit]:
        ids, p = model.greedy(torch.tensor([src], device=device), bos, eos)
        out.append((gloss, lang, text, detokenize([tgt_itos[i] for i in ids]), p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentences", type=int, default=60000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--min-confidence", type=float, default=0.5)
    ap.add_argument("--junk-ratio", type=float, default=0.15, help="random <unk> glosses per training sentence")
    ap.add_argument("--fixed-repeat", type=int, default=10,
                    help="copies of single words, fixed phrases and greetings (few next to 25k patterns)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUT_DIR, exist_ok=True)

    labels, english, hindi = load_vocab()
    pairs = build_corpus(labels, english, hindi, n=a.sentences, seed=a.seed)
    rng = random.Random(a.seed)
    # Single words, fixed phrases, greetings and the pronoun + clothing set always train;
    # sampled sentences split 90 / 5 / 5 by gloss
    fixed_keys = ({p[0] for p in pairs if len(p[0]) == 1 or p[0][0] in GREETINGS}
                  | {p[0] for p in FIXED} | {p[0] for p in clothing_pairs()})
    fixed = [p for p in pairs if p[0] in fixed_keys]
    rest = [p for p in pairs if p[0] not in fixed_keys]
    rng.shuffle(rest)
    n_hold = len(rest) // 20
    test_p, val_p, train_p = rest[:n_hold], rest[n_hold:2 * n_hold], rest[2 * n_hold:] + fixed
    known = {p[0] for p in pairs}
    junk_train = make_junk(labels, known, int(a.junk_ratio * len(train_p)), rng)
    train_p = train_p + fixed * (a.fixed_repeat - 1) + junk_train
    print(f"[DATA] {len(pairs)} gloss sequences: train {len(train_p)}, val {len(val_p)}, test {len(test_p)}")

    src_itos = [PAD, *LANG_TOKENS.values()] + labels
    tgt_words = sorted({t for _, en, hi in pairs for t in tokenize_sentence(en) + tokenize_sentence(hi)})
    tgt_itos = SPECIALS + tgt_words
    print(f"[DATA] source vocab {len(src_itos)}, target vocab {len(tgt_itos)}")
    with open(os.path.join(OUT_DIR, "corpus_sample.tsv"), "w", encoding="utf-8") as f:
        for g, en, hi in rng.sample(pairs, 300):
            f.write(f"{' '.join(g).upper()}\t{en}\t{hi}\n")

    train, val, test = (encode(p, src_itos, tgt_itos) for p in (train_p, val_p, test_p))
    model = GlossTranslator(len(src_itos), len(tgt_itos)).to(device)
    print(f"[MODEL] {sum(p.numel() for p in model.parameters()):,} parameters on {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * ((len(train) + a.batch_size - 1) // a.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    best, best_path = -1.0, os.path.join(OUT_DIR, "best.pt")
    for ep in range(1, a.epochs + 1):
        model.train(); t0 = time.time(); total = 0.0; n = 0
        for src, tgt in batches(train, a.batch_size, True, rng):
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, tgt[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            total += float(loss) * src.size(0); n += src.size(0)
        res = decode_all(model, val, tgt_itos, device, limit=600)
        acc = sum(r[2] == r[3] for r in res) / len(res)
        flag = ""
        if acc > best:
            best = acc; flag = " [*BEST]"
            torch.save({"model": model.state_dict(), "src_itos": src_itos, "tgt_itos": tgt_itos,
                        "config": {"d_model": 128, "nhead": 4, "encoder_layers": 2, "decoder_layers": 2,
                                   "ff_dim": 512}}, best_path)
        print(f"Epoch {ep:02d}/{a.epochs} [{time.time() - t0:.1f}s] loss {total / n:.4f} | val exact match {acc:.3f}{flag}")

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    res = decode_all(model, test, tgt_itos, device)
    fixed_res = decode_all(model, encode(fixed, src_itos, tgt_itos), tgt_itos, device)
    lines = [f"single words / fixed phrases / greetings reproduced exactly: "
             f"{sum(x[2] == x[3] for x in fixed_res) / len(fixed_res):.3f} of {len(fixed_res)}"]
    for lang in ("en", "hi"):
        r = [x for x in res if x[1] == lang]
        acc = sum(x[2] == x[3] for x in r) / len(r)
        passed = [x for x in r if x[4] >= a.min_confidence]
        lines.append(f"{lang}: exact match {acc:.3f} on {len(r)} unseen gloss sequences; "
                     f"{len(passed) / len(r):.3f} pass the confidence gate, exact among those "
                     f"{sum(x[2] == x[3] for x in passed) / max(1, len(passed)):.3f}")
    # Gibberish: random glosses that follow no pattern should mostly be rejected
    s2i = {t: i for i, t in enumerate(src_itos)}
    held_junk = make_junk(labels, known, 400, rng, exclude={g for g, _, _ in junk_train})
    junk = [([s2i["<2en>"]] + [s2i[x] for x in g], None, g, "en", "") for g, _, _ in held_junk]
    jr = decode_all(model, junk, tgt_itos, device)
    rejected = sum(x[3] == UNK or x[4] < a.min_confidence for x in jr) / len(jr)
    lines.append(f"unseen random gloss sequences rejected (<unk> or below the confidence gate): {rejected:.3f}")
    lines.append("\nExamples (test, gloss | reference | prediction | min prob):")
    for x in rng.sample(res, 30):
        lines.append(f"  {' '.join(x[0]).upper()} [{x[1]}] | {x[2]} | {x[3]} | {x[4]:.2f}")
    lines.append("\nRandom glosses (prediction | min prob):")
    for x in jr[:15]:
        lines.append(f"  {' '.join(x[0]).upper()} | {x[3]} | {x[4]:.2f}")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(OUT_DIR, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    with open(os.path.join(OUT_DIR, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"src": src_itos, "tgt": tgt_itos}, f, ensure_ascii=False)
    print(f"[OK] best checkpoint {best_path} (val exact match {best:.3f})")


if __name__ == "__main__":
    main()
