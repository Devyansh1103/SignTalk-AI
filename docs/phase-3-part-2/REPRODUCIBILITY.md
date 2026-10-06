# SignTalk AI — ST-GCN Reproducibility & Determinism Specification

**Document ID:** `DOC-P3P2-REPRO-001`  
**Phase:** Phase 3 — Part 2 (ST-GCN Development & Training)  
**Configuration Reference:** [`configs/stgcn.yaml`](file:///d:/SignAI/configs/stgcn.yaml)  
**Script Reference:** [`scripts/verify_stgcn_reproducibility.py`](file:///d:/SignAI/scripts/verify_stgcn_reproducibility.py)  

---

## 1. Environment & Software Stack Specification

Exact environmental specifications for reproducing the ST-GCN experimental results:

| Environment Component | Exact Version / Specification | Verification Command |
| :--- | :--- | :--- |
| **Operating System** | Windows 11 64-bit (AMD64) | `systeminfo` |
| **Python Runtime** | Python 3.13.12 (Miniconda) | `python --version` |
| **PyTorch** | 2.13.0+cpu | `python -c "import torch; print(torch.__version__)"` |
| **NumPy** | 2.5.1 | `python -c "import numpy; print(numpy.__version__)"` |
| **Pandas** | 2.2.3 | `python -c "import pandas; print(pandas.__version__)"` |
| **Matplotlib** | 3.10.0 | `python -c "import matplotlib; print(matplotlib.__version__)"` |
| **PyYAML** | 6.0.3 | `python -c "import yaml; print(yaml.__version__)"` |
| **Hardware Device** | Intel/AMD x86_64 Multi-Core CPU | [`src/utils/device.py`](file:///d:/SignAI/src/utils/device.py) |

---

## 2. Dataset & Artifact Lineage

| Artifact Component | Version / Identifier | Storage Location |
| :--- | :--- | :--- |
| **Dataset Release** | `signTalk-seq-v1.0.0` | `data/processed/sequences/` |
| **Landmark Schema** | `93-node-v1` (21 LH, 21 RH, 11 Pose, 40 Face) | [`docs/phase-2-part-2/LANDMARK_DATA_SCHEMA.md`](file:///d:/SignAI/docs/phase-2-part-2/LANDMARK_DATA_SCHEMA.md) |
| **Vocabulary Token Map**| `mvp_10:v1.0.0` (10 classes) | [`assets/vocabularies/mvp_10.json`](file:///d:/SignAI/assets/vocabularies/mvp_10.json) |
| **Partition Manifests** | `train.csv` (36), `val.csv` (12), `test.csv` (12) | `data/manifests/` |

---

## 3. Seed Synchronization & Determinism Management

Deterministic initialization is enforced by [`src/utils/reproducibility.py`](file:///d:/SignAI/src/utils/reproducibility.py):
```python
def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

---

## 4. Empirical Dual-Trial Verification

Twin training runs (`Trial 1` vs `Trial 2`) executed using `scripts/verify_stgcn_reproducibility.py` across 3 epochs demonstrated bit-for-bit identical loss curves and 100% sample prediction matches:
- Maximum Float Discrepancy: **0.00e+00**
- Prediction Match Rate: **100.0%**
- Verification Status: **PASSED (PERFECT DETERMINISM)**
- Audit Report: [`experiments/stgcn/metrics/reproducibility_report.json`](file:///d:/SignAI/experiments/stgcn/metrics/reproducibility_report.json)
