#!/usr/bin/env python3
"""
SignTalk AI: record your own signs from the webcam for fine-tuning.

Runs the exact live pipeline (same MediaPipe settings, same segmentation, same 45-frame
model input as the backend), so the saved samples look like what the model sees live.
Before each word its INCLUDE reference video plays in the corner; press SPACE to start, then
sign the word and lower your hands. Each finished sign is saved.

Resumes by itself: takes already in the manifest count towards --reps, so after quitting just
run the same command again and it continues with the remaining words.

Keys: SPACE = start the word, u = undo last sample, s = skip word, c = next camera, q = quit.
Stop the backend first (both need the webcam and the CPU).

Usage:
  python scripts/record_signs.py --words hello alright fan cow thankyou --reps 15
  python scripts/record_signs.py --all --reps 10        # all vocabulary words
Output:
  data/user_recordings/<signer>/<word>/*.npz  +  data/manifests/user_recordings.csv
"""
import argparse, glob, logging, os, re, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import numpy as np
import pandas as pd

from src.realtime.realtime_config import RealTimeConfig
from src.realtime.realtime_pipeline import RealtimePipeline
from src.realtime.sign_segmenter import resample_by_time
from src.data.label_map import LABEL_TO_ID, get_gloss, get_translation

MANIFEST = "data/manifests/user_recordings.csv"
DEFAULT_REFERENCE_ROOT = "D:/data convertion for mini project/dataset"


def save_sample(pipe, segment, word, signer, out_dir):
    """Store the model input exactly as the live backend builds it, plus the raw landmarks."""
    cfg = pipe.config
    window = resample_by_time(segment, segment[0].timestamp, segment[-1].timestamp, cfg.buffer.max_length)
    x, mask = pipe.model_runner.window_to_tensor(window, offline_postprocess=cfg.inference.offline_postprocess)
    cid = LABEL_TO_ID[word]
    seq_id = f"user_{signer}_{word}_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}"
    path = os.path.join(out_dir, f"{seq_id}.npz")
    np.savez_compressed(
        path,
        data=x[0].cpu().numpy().astype(np.float32),        # (3, 45, 93)
        mask=mask[0].cpu().numpy().astype(np.float32),     # (1, 45, 93)
        label=np.int64(cid),
        gloss_id=np.int64(cid + 4),                        # same offset as the INCLUDE sequences
        raw_coords=np.stack([f.raw_coords for f in segment]).astype(np.float32),
        raw_mask=np.stack([f.mask for f in segment]),
        timestamps=np.array([f.timestamp for f in segment]),
        aspect_y_scale=np.array([f.aspect_y_scale for f in segment]),
    )
    return {
        "sequence_id": seq_id, "source_dataset": "USER", "source_recording_id": seq_id,
        "source_video_id": "webcam", "signer_id": f"user_{signer}", "split": "train",
        "class_id": cid, "label": word, "gloss": get_gloss(cid), "translation": get_translation(cid),
        "duration_seconds": round(segment[-1].timestamp - segment[0].timestamp, 2),
        "quality_score": 0.95, "quality_status": "GOOD", "file_path": path.replace("\\", "/"),
    }


def find_references(root):
    """word -> INCLUDE videos, from <root>/<Category>/<N. Word>/*.MOV (labels are letters only)."""
    refs = {}
    for d in glob.glob(os.path.join(root, "*", "*")):
        word = re.sub(r"[^a-z]", "", re.sub(r"^\d+\.\s*", "", os.path.basename(d)).lower())
        videos = sorted(glob.glob(os.path.join(d, "*.MOV")) + glob.glob(os.path.join(d, "*.mp4")) + glob.glob(os.path.join(d, "*.MP4")))
        if videos:
            refs.setdefault(word, videos)
    return refs


class ReferencePlayer:
    """Loops one INCLUDE clip of the current word, scaled down for a corner of the window.
    Mirrored like the camera preview, so copying what you see uses the signer's hand, not the
    opposite one."""

    def __init__(self, videos, width=256):
        self.frames = []
        if videos:
            cap = cv2.VideoCapture(videos[0])
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                h, w = f.shape[:2]
                self.frames.append(cv2.flip(cv2.resize(f, (width, int(h * width / w))), 1))
            cap.release()
        self.i = 0

    def draw(self, view):
        if not self.frames:
            return
        f = self.frames[self.i % len(self.frames)]
        self.i += 1
        h, w = f.shape[:2]
        H, W = view.shape[:2]
        y0, x0 = H - h - 8, W - w - 8
        view[y0:y0 + h, x0:x0 + w] = f
        cv2.rectangle(view, (x0 - 1, y0 - 1), (x0 + w, y0 + h), (0, 200, 255), 1)
        cv2.putText(view, "reference", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)


def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", nargs="+", default=["hello", "alright", "fan", "cow", "thankyou"])
    ap.add_argument("--all", action="store_true", help="record every vocabulary word")
    ap.add_argument("--reps", type=int, default=15, help="target takes per word (existing takes count)")
    ap.add_argument("--signer", default="me")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--reference-root", default=DEFAULT_REFERENCE_ROOT,
                    help="INCLUDE dataset folder, for the reference clip of each word")
    a = ap.parse_args()

    words = sorted(LABEL_TO_ID, key=LABEL_TO_ID.get) if a.all else a.words
    unknown = [w for w in words if w not in LABEL_TO_ID]
    if unknown:
        sys.exit(f"[ERROR] Not in the model vocabulary: {unknown}. Check assets/ vocabulary for exact spelling.")

    rows = pd.read_csv(MANIFEST).to_dict("records") if os.path.exists(MANIFEST) else []
    have = pd.Series([r["label"] for r in rows if r["signer_id"] == f"user_{a.signer}"]).value_counts()
    todo = [(w, a.reps - int(have.get(w, 0))) for w in words if a.reps - int(have.get(w, 0)) > 0]
    print(f"[INFO] {len(words) - len(todo)} of {len(words)} words already have {a.reps} takes; "
          f"{sum(n for _, n in todo)} takes left over {len(todo)} words")
    if not todo:
        return
    refs = find_references(a.reference_root) if os.path.isdir(a.reference_root) else {}
    if not refs:
        print(f"[WARN] No reference videos under {a.reference_root}; recording without them")

    logging.disable(logging.INFO)
    cfg = RealTimeConfig.from_yaml("configs/realtime.yaml")
    cfg.debug.save_segments = False
    pipe = RealtimePipeline(config=cfg, camera=None)
    pipe.initialize()

    # Capture each finished sign the segmenter hands to the classifier
    finished = []
    original_update = pipe.segmenter.update
    def update(frame):
        seg = original_update(frame)
        if seg is not None:
            finished.append(seg)
        return seg
    pipe.segmenter.update = update

    cam = a.camera
    cap = open_camera(cam)
    if not cap.isOpened():
        sys.exit("[ERROR] Webcam not available. Close the browser tab / backend that is using it.")

    saved = []  # (row, path) of this run, for undo
    last_msg, t_fps, n_fps, fps = "", time.time(), 0, 0.0

    quit_requested = False
    for wi, (word, need) in enumerate(todo):
        if quit_requested:
            break
        out_dir = os.path.join("data", "user_recordings", a.signer, word)
        os.makedirs(out_dir, exist_ok=True)
        ref = ReferencePlayer(refs.get(word))
        done_before = a.reps - need
        count = 0
        started = False  # waits for SPACE, so the reference clip can be watched first
        pipe.segmenter.reset()
        while count < need:
            ok, frame = cap.read()
            if not ok:
                continue
            if started:
                pipe.step(frame)
                n_fps += 1
            if time.time() - t_fps >= 1.0:
                fps, n_fps, t_fps = n_fps / (time.time() - t_fps), 0, time.time()

            if finished:
                seg = finished.pop()
                finished.clear()
                row = save_sample(pipe, seg, word, a.signer, out_dir)
                rows.append(row)
                saved.append(row)
                count += 1
                pred = pipe.latest_prediction
                guess = f"{pred.label} {pred.confidence:.0%}" if pred is not None else "-"
                last_msg = f"saved #{done_before + count} ({row['duration_seconds']:.1f}s) | model abhi: {guess}"
                print(f"[{word}] {last_msg}")
                pd.DataFrame(rows).to_csv(MANIFEST, index=False)

            view = cv2.flip(frame, 1)
            ref.draw(view)
            signing = pipe.segmenter.in_segment
            cv2.rectangle(view, (0, 0), (640, 92), (0, 0, 0), -1)
            cv2.putText(view, f"{word.upper()}   {done_before + count}/{a.reps}   (word {wi + 1}/{len(todo)})",
                        (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            status = ("reference dekho, SPACE = shuru, s = skip" if not started
                      else "SIGNING..." if signing else "ready: sign karo, phir haath neeche")
            cv2.putText(view, status, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 200, 255) if not started else (0, 0, 255) if signing else (0, 255, 0), 2)
            cv2.putText(view, f"{last_msg}   fps {fps:.0f}   cam {cam}   [u]ndo [s]kip [c]amera [q]uit", (10, 84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.imshow("SignTalk recorder", view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                quit_requested = True
                break
            elif key == ord("c"):
                # Laptops can list virtual cameras (e.g. Lenovo Smart Connect) before the webcam
                cap.release()
                for nxt in (cam + 1, cam + 2, 0):
                    cap = open_camera(nxt)
                    if cap.isOpened() and cap.read()[0]:
                        cam = nxt
                        break
                    cap.release()
                else:
                    cap = open_camera(cam)
                pipe.segmenter.reset()
                print(f"[INFO] camera {cam}")
            elif key == ord("s"):
                break
            elif key == ord(" ") and not started:
                started = True
                pipe.segmenter.reset()
            elif key == ord("u") and saved and saved[-1]["label"] == word and count > 0:
                row = saved.pop()
                rows.remove(row)
                os.remove(row["file_path"])
                count -= 1
                last_msg = f"undo: #{done_before + count + 1} deleted"
                pd.DataFrame(rows).to_csv(MANIFEST, index=False)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[OK] This run: {len(saved)} samples. Total in {MANIFEST}: {len(rows)}")
    if rows:
        counts = pd.DataFrame(rows)["label"].value_counts()
        print(f"{len(counts)} words recorded, {int((counts >= a.reps).sum())} with {a.reps}+ takes")
    if quit_requested:
        print("Baaki words ke liye wahi command dobara chalao; jo ho chuke hain wo skip honge.")


if __name__ == "__main__":
    main()
