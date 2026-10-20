#!/usr/bin/env python3
"""
SignTalk AI: Webcam recorder for clean, correctly-labelled training clips.

Saves UNMIRRORED clips (same as the web app sends to the backend) as
  data/raw/videos/{label}_signer_{id}_rep{n}.mp4   (25 FPS, 640x480)

Keys:  0-9 = class select | SPACE = 3-2-1 countdown then record 2.2 s | q = quit
Usage: python scripts/record_clips.py --signer 01
"""
import argparse, os, re, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cv2
from src.data.label_map import ID_TO_LABEL

ap = argparse.ArgumentParser()
ap.add_argument("--signer", default="01")
ap.add_argument("--out", default="data/raw/videos")
ap.add_argument("--camera", type=int, default=0)
ap.add_argument("--seconds", type=float, default=2.2)
ap.add_argument("--fps", type=float, default=25.0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

def next_rep(label):
    pat = re.compile(rf"^{label}_signer_{a.signer}_rep(\d+)\.mp4$")
    reps = [int(m.group(1)) for f in os.listdir(a.out) if (m := pat.match(f))]
    return max(reps, default=0) + 1

cap = cv2.VideoCapture(a.camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    sys.exit("Camera open nahi hua (--camera 1 try karo)")
cur = 0
state, t0, writer, frames, path = "idle", 0, None, [], ""
interval = 1.0 / a.fps; last = 0
while True:
    ok, fr = cap.read()
    if not ok: break
    now = time.time()
    label = ID_TO_LABEL[cur]
    view = cv2.flip(fr, 1).copy()  # mirror only for display
    msg = f"[{cur}] {label.upper()}  rep#{next_rep(label)}  signer_{a.signer}"
    if state == "countdown":
        left = 3 - int(now - t0)
        if left <= 0:
            state, t0, frames = "rec", now, []; last = 0
        else:
            cv2.putText(view, str(left), (270, 260), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 8)
    elif state == "rec":
        if now - last >= interval:
            frames.append(fr.copy()); last = now
        cv2.circle(view, (30, 70), 12, (0, 0, 255), -1)
        if now - t0 >= a.seconds:
            label = ID_TO_LABEL[cur]
            path = os.path.join(a.out, f"{label}_signer_{a.signer}_rep{next_rep(label)}.mp4")
            w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (640, 480))
            for f in frames: w.write(cv2.resize(f, (640, 480)))
            w.release(); print("saved", path, len(frames), "frames"); state = "idle"
    cv2.putText(view, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("SignTalk recorder  (SPACE record, 0-9 class, q quit)", view)
    k = cv2.waitKey(1) & 0xFF
    if k == ord("q"): break
    if k == ord(" ") and state == "idle": state, t0 = "countdown", time.time()
    if ord("0") <= k <= ord("9") and state == "idle": cur = k - ord("0")
cap.release(); cv2.destroyAllWindows()
