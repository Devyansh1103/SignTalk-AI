# SignTalk AI — Sahi retraining guide (Hinglish)

## Problem kya tha
1. `data/raw/videos/` ke purane videos ki labels galat thi (PDF audit: "hello" file me asal sign "Good afternoon").
2. `data/manifests/*.csv` me poora INCLUDE dataset (~58 classes, class_id 243 tak) mixed tha, jabki model 10 classes ka hai. `happy`/`teacher` ke 2-2 alag class_id the.
3. `hello` ke sirf 4 train clips the, wo bhi wahi galat videos.
4. Playbook ka `python scripts/run_preprocessing.py` default **pilot** mode me chalta hai (sirf 10 clips). Full ke liye `--mode full` chahiye.

## Steps (project root se)
```bash
# 0. Purana sab hatao (galat labels wala)
rmdir /s /q data\processed          # Linux/Mac: rm -rf data/processed
del data\manifests\*.csv            # Linux/Mac: rm data/manifests/*.csv
del experiments\stgcn\checkpoints\*.pt

# 1. Naye clean clips record karo (har class ke 20-30 reps, 2-3 alag distance/lighting; ho sake to 2+ signers)
python scripts/record_clips.py --signer 01
#    0-9 = class, SPACE = record. Naming khud ban jati hai: hello_signer_01_rep1.mp4

# 2. Audit: counts + contact sheet dekho (data/interim/audit/*.png) — har class ka sign khud verify karo
python scripts/audit_dataset.py

# 3. Raw manifest banao
python scripts/make_raw_manifest.py

# 4. Landmarks (FULL mode zaroori)
python scripts/run_preprocessing.py --mode full

# 5. Sequences
python scripts/build_sequences.py

# 6. Audit dobara (manifest check) + train
python scripts/audit_dataset.py
python scripts/train_stgcn.py
python scripts/evaluate_stgcn.py --split test

# 7. App
python -m uvicorn backend.app.main:app --port 8000
cd frontend && npm run dev
```

## Dhyan rakho
- Test accuracy tabhi bharose layak hai jab test clips alag signer / alag session ke ho. Same session ke reps me split karoge to accuracy fake-high aayegi.
- Live me bhi wahi camera distance, lighting aur seedha camera jaisa training me tha, rakho.
- Agar kisi class me confusion ho to `scripts/generate_stgcn_confusion_matrix.py` chalao.
