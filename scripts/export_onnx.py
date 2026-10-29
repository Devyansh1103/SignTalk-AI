#!/usr/bin/env python3
"""
SignTalk AI: export the deployed ST-GCN to ONNX for the browser build (onnxruntime-web).

Writes to frontend/public/models/:
  stgcn.onnx     model (softmax output), checked against PyTorch on random and real inputs
  vocab.json     the vocabulary classes with English / Hindi translations
  pipeline.json  the live settings from configs/realtime.yaml that the browser pipeline uses
Re-run after every new checkpoint (retrain or fine-tune) so the website uses it too.

Usage:
  python scripts/export_onnx.py
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import glob
import numpy as np
import torch

from src.realtime.model_runner import STGCNRunner
from src.realtime.realtime_config import RealTimeConfig
from src.data.label_map import CANONICAL_CLASSES


class Wrapped(torch.nn.Module):
    """Logits -> probabilities, so the browser only has to read the output."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, mask):
        return torch.softmax(self.model(x, mask), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="experiments/stgcn/checkpoints/best_checkpoint.pt")
    ap.add_argument("--out", default="frontend/public/models/stgcn.onnx")
    ap.add_argument("--config", default="configs/realtime.yaml")
    a = ap.parse_args()

    runner = STGCNRunner(checkpoint_path=a.checkpoint, device="cpu")
    model = Wrapped(runner.model).eval()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    x = torch.randn(1, 3, 45, 93)
    mask = torch.ones(1, 1, 45, 93)
    torch.onnx.export(model, (x, mask), a.out, input_names=["x", "mask"], output_names=["probs"],
                      opset_version=17, do_constant_folding=True)
    print(f"[OK] wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB)")

    # Parity: random inputs plus real processed clips
    import onnxruntime as ort
    sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    samples = [(torch.randn(1, 3, 45, 93), (torch.rand(1, 1, 45, 93) > 0.2).float()) for _ in range(5)]
    for path in sorted(glob.glob("data/processed/**/*.npz", recursive=True))[:50]:
        with np.load(path) as d:
            samples.append((torch.from_numpy(d["data"]).float()[None], torch.from_numpy(d["mask"]).float()[None]))
    worst, agree = 0.0, 0
    with torch.no_grad():
        for xs, ms in samples:
            ref = model(xs, ms).numpy()
            out = sess.run(None, {"x": xs.numpy(), "mask": ms.numpy()})[0]
            worst = max(worst, float(np.abs(ref - out).max()))
            agree += int(ref.argmax() == out.argmax())
    print(f"[CHECK] {len(samples)} inputs: max |prob diff| {worst:.2e}, same top-1 {agree}/{len(samples)}")
    if worst > 1e-4 or agree != len(samples):
        sys.exit("[ERROR] ONNX output differs from PyTorch")

    write_browser_files(os.path.dirname(a.out), a.config)


def write_browser_files(out_dir, config_path):
    from src.realtime.translator import HINDI_LEXICON
    vocab = [{"class_id": cid, "label": i.label, "gloss": i.gloss, "english_translation": i.translation,
              "hindi_translation": HINDI_LEXICON.get(i.label, "")} for cid, i in sorted(CANONICAL_CLASSES.items())]
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    c = RealTimeConfig.from_yaml(config_path)
    mp, n, s, t = c.mediapipe, c.normalization, c.segmentation, c.translation
    pipeline = {
        "mediapipe": {"pose_confidence": mp.pose_detection_confidence, "hand_confidence": mp.hand_detection_confidence,
                      "hand_tracking_confidence": mp.hand_tracking_confidence},
        "normalization": {"min_scale_epsilon": n.min_scale_epsilon, "z_scale_factor": n.z_scale_factor,
                          "use_temporal_anchor_smoothing": n.use_temporal_anchor_smoothing,
                          "anchor_smoothing_alpha": n.anchor_smoothing_alpha, "reference_aspect": n.reference_aspect},
        "quality": {"weights": c.quality.weights, "good": c.quality.classification_thresholds.get("good", 0.75),
                    "acceptable": c.quality.classification_thresholds.get("acceptable", 0.50)},
        "segmentation": {"active_max_y": s.active_max_y, "min_active_sec": s.min_active_sec,
                         "end_hold_sec": s.end_hold_sec, "pad_before_sec": s.pad_before_sec,
                         "pad_after_sec": s.pad_after_sec, "max_segment_sec": s.max_segment_sec,
                         "min_confidence": s.min_confidence, "min_margin": c.confidence.min_margin},
        "sequence_length": c.buffer.max_length,
        "max_consecutive_missing": 10,
        "translation": {"pause_threshold_sec": t.pause_threshold_sec, "min_confidence": t.min_confidence,
                        "max_phrase_signs": t.max_phrase_signs},
        "max_events": c.sequence.max_events,
        "top_k": c.inference.top_k,
    }
    with open(os.path.join(out_dir, "pipeline.json"), "w", encoding="utf-8") as f:
        json.dump(pipeline, f, indent=2)
    print(f"[OK] wrote vocab.json ({len(vocab)} classes) and pipeline.json")


if __name__ == "__main__":
    main()
