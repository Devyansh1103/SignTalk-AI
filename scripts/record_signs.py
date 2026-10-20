#!/usr/bin/env python3
"""
SignTalk AI: record your own signs from the webcam for fine-tuning.

Runs the exact live pipeline (same MediaPipe settings, same segmentation, same 45-frame
model input as the backend), so the saved samples look like what the model sees live.
Sign the word shown on screen, then lower your hands; each finished sign is saved.

Keys: u = undo last sample, s = skip to next word, q = quit.
Stop the backend first (both need the webcam and the CPU).

Usage:
  python scripts/record_signs.py --words hello alright fan cow thankyou --reps 15
Output:
  data/user_recordings/<signer>/<word>/*.npz  +  data/manifests/user_recordings.csv
"""
import argparse, logging, os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
import numpy as np
import pandas as pd

from src.realtime.realtime_config import RealTimeConfig
from src.realtime.realtime_pipeline import RealtimePipeline
from src.realtime.sign_segmenter import resample_by_time
from src.data.label_map import LABEL_TO_ID, get_gloss, get_translation

MANIFEST = "data/manifests/user_recordings.csv"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", nargs="+", default=["hello", "alright", "fan", "cow", "thankyou"])
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--signer", default="me")
    ap.add_argument("--camera", type=int, default=0)
    a = ap.parse_args()

    unknown = [w for w in a.words if w not in LABEL_TO_ID]
    if unknown:
        sys.exit(f"[ERROR] Not in the model vocabulary: {unknown}. Check assets/ vocabulary for exact spelling.")

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

    cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        sys.exit("[ERROR] Webcam not available. Close the browser tab / backend that is using it.")

    rows = pd.read_csv(MANIFEST).to_dict("records") if os.path.exists(MANIFEST) else []
    saved = []  # (row, path) of this run, for undo
    last_msg, t_fps, n_fps, fps = "", time.time(), 0, 0.0

    quit_requested = False
    for wi, word in enumerate(a.words):
        if quit_requested:
            break
        out_dir = os.path.join("data", "user_recordings", a.signer, word)
        os.makedirs(out_dir, exist_ok=True)
        count = 0
        pause_until = time.time() + 3.0  # time to read the next word
        while count < a.reps:
            ok, frame = cap.read()
            if not ok:
                continue
            if time.time() >= pause_until:
                pipe.step(frame)
                n_fps += 1
            else:
                pipe.segmenter.reset()
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
                last_msg = f"saved #{count} ({row['duration_seconds']:.1f}s) | model abhi: {guess}"
                print(f"[{word}] {last_msg}")
                pd.DataFrame(rows).to_csv(MANIFEST, index=False)

            view = cv2.flip(frame, 1)
            signing = pipe.segmenter.in_segment
            cv2.rectangle(view, (0, 0), (640, 92), (0, 0, 0), -1)
            cv2.putText(view, f"{word.upper()}   {count}/{a.reps}   (word {wi + 1}/{len(a.words)})",
                        (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            status = ("agla word padho..." if time.time() < pause_until
                      else "SIGNING..." if signing else "ready: sign karo, phir haath neeche")
            cv2.putText(view, status, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255) if signing else (0, 255, 0), 2)
            cv2.putText(view, f"{last_msg}   fps {fps:.0f}   [u]ndo [s]kip [q]uit", (10, 84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            cv2.imshow("SignTalk recorder", view)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                quit_requested = True
                break
            elif key == ord("s"):
                break
            elif key == ord("u") and saved and saved[-1]["label"] == word:
                row = saved.pop()
                rows.remove(row)
                os.remove(row["file_path"])
                count -= 1
                last_msg = f"undo: #{count + 1} deleted"
                pd.DataFrame(rows).to_csv(MANIFEST, index=False)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[OK] This run: {len(saved)} samples. Total in {MANIFEST}: {len(rows)}")
    if rows:
        print(pd.DataFrame(rows)["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
