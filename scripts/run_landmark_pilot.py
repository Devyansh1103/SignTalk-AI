"""
run_landmark_pilot.py - Landmark feasibility check using MediaPipe Holistic
Measures real detection rates, coordinates, latency, and failure cases on representative samples.
"""

import cv2
import os
import time
import json
import numpy as np
import mediapipe as mp

def main():
    mp_holistic = mp.solutions.holistic

    pilot_dir = 'data/interim/landmark_pilot'
    video_files = sorted([f for f in os.listdir(pilot_dir) if f.endswith('.mp4')])

    pilot_metrics = {}
    print(f"Starting Landmark Pilot across {len(video_files)} test clips...")

    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    for vf in video_files:
        vpath = os.path.join(pilot_dir, vf)
        cap = cv2.VideoCapture(vpath)
        total_frames = 0
        pose_detected = 0
        lh_detected = 0
        rh_detected = 0
        face_detected = 0
        frame_times = []
        landmarks_record = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            total_frames += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            t0 = time.perf_counter()
            results = holistic.process(rgb)
            t1 = time.perf_counter()
            frame_times.append((t1 - t0) * 1000)

            p_ok = 1 if results.pose_landmarks else 0
            lh_ok = 1 if results.left_hand_landmarks else 0
            rh_ok = 1 if results.right_hand_landmarks else 0
            f_ok = 1 if results.face_landmarks else 0

            pose_detected += p_ok
            lh_detected += lh_ok
            rh_detected += rh_ok
            face_detected += f_ok

            lh_coords = [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark] if lh_ok else [[0.0, 0.0, 0.0]] * 21
            rh_coords = [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark] if rh_ok else [[0.0, 0.0, 0.0]] * 21
            pose_coords = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark] if p_ok else [[0.0, 0.0, 0.0]] * 33
            landmarks_record.append({'pose': pose_coords, 'lh': lh_coords, 'rh': rh_coords})

        cap.release()

        npy_path = os.path.join(pilot_dir, vf.replace('.mp4', '_landmarks.npy'))
        np.save(npy_path, np.array(landmarks_record, dtype=object))

        avg_latency = float(np.mean(frame_times)) if frame_times else 0.0
        p95_latency = float(np.percentile(frame_times, 95)) if frame_times else 0.0

        rh_rate = round(rh_detected / total_frames * 100, 2) if total_frames else 0.0
        lh_rate = round(lh_detected / total_frames * 100, 2) if total_frames else 0.0
        pose_rate = round(pose_detected / total_frames * 100, 2) if total_frames else 0.0
        face_rate = round(face_detected / total_frames * 100, 2) if total_frames else 0.0

        pilot_metrics[vf] = {
            'total_frames': total_frames,
            'pose_success_rate': pose_rate,
            'left_hand_success_rate': lh_rate,
            'right_hand_success_rate': rh_rate,
            'face_success_rate': face_rate,
            'avg_frame_latency_ms': round(avg_latency, 2),
            'p95_frame_latency_ms': round(p95_latency, 2),
            'fps_equivalent': round(1000.0 / avg_latency, 2) if avg_latency > 0 else 0.0
        }
        print(f"Finished {vf}: {total_frames} frames, {avg_latency:.1f}ms/frame ({1000/avg_latency:.1f} FPS), RH={rh_rate}%, Face={face_rate}%, Pose={pose_rate}%")

    holistic.close()

    summary_file = os.path.join(pilot_dir, 'pilot_results.json')
    with open(summary_file, 'w') as out_f:
        json.dump(pilot_metrics, out_f, indent=2)

    print(f"Landmark pilot completed successfully. Summary saved to {summary_file}")

if __name__ == '__main__':
    main()
