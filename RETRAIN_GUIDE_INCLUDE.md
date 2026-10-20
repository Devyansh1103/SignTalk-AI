# INCLUDE dataset se sabhi classes ke saath training (Hinglish)

Ab class count fixed 10 nahi hai. Tumhare dataset ke jitne class folders hain (`<Category>/<N. Word>/*.MOV`),
utni classes ban jati hain. Dataset project ke bahar rakhna theek hai.

## Steps (project root `D:\SignTalk-AI`, venv active)
```
# 0. Purana data/model hatao (class badle hain, purana checkpoint kaam nahi karega)
rmdir /s /q data\processed
del data\manifests\*.csv
del experiments\stgcn\checkpoints\*.pt

# 1. Vocabulary banao (dataset ke class folders se). Ye list print karega: id, class, clips, Hindi
python scripts/build_vocabulary.py --include-root "D:\data convertion for mini project\dataset"

# 2. Dry-run: har class ke clips (train/val/test)
python scripts/convert_include_mvp10.py --dry-run --resplit --include-root "D:\data convertion for mini project\dataset"

# 3. Conversion (MediaPipe). BAHUT LAMBA: ~1700 videos ~ 3-4 ghante CPU pe. Beech me ruk jaye to
#    SAME command dobara chalao: jo videos ho chuke wo skip ho jate hain.
python scripts/convert_include_mvp10.py --resplit --include-root "D:\data convertion for mini project\dataset"

# 4. Manifest check
python scripts/audit_dataset.py --videos NONE

# 5. Train, evaluate
python scripts/train_stgcn.py
python scripts/evaluate_stgcn.py --split test

# 6. App
python -m uvicorn backend.app.main:app --port 8000
cd frontend && npm run dev
```

## Dhyan rakho
- `--min-clips 6` (default): jis class me 6 se kam clips hain wo skip ho jati hai (split nahi ban sakta).
  Kam clips wali classes pe accuracy waise bhi buri aayegi.
- Naya dataset baad me download karo to bas step 0-5 dobara (vocabulary naye folders ko khud jod degi).
  Class ids badal jate hain, isliye retrain zaroori hai.
- Hindi shabd `src/data/hindi_lexicon.py` me hain. Galat lage to wahin badlo, phir step 1 dobara chalao
  (conversion dobara nahi chalega). Jis class ka Hindi wahan nahi hai, English dikhega.
- 100+ classes aur ~15 clips/class: accuracy 10-class se kaafi kam aayegi, aur milte-julte signs
  (jaise Good morning / Good afternoon) confuse honge. Test accuracy sirf ek signer-mix pe hai.
- Vocabulary 10-class pe wapas lani ho: `python scripts/build_vocabulary.py --default10`
