# SignTalk AI — Model & Experiment Versioning Ledger

**Document ID:** `DOC-P3P2-VERSIONING-001`  
**Phase:** Phase 3 — Part 2 (ST-GCN Development & Training)  
**Status:** ACTIVE SPECIFICATION  

---

## 1. Versioning Hierarchy

To maintain complete experimental lineage from raw dataset extraction through production checkpoints, SignTalk AI enforces a multi-tier versioning convention:

```
[Model Family] - [Architecture Type] - [Model Version]
       ├── Dataset Version
       ├── Preprocessing Version
       ├── Landmark Schema Version
       ├── Configuration Version
       └── Experiment Identifier
```

---

## 2. Active Model Version Registry

| Component | Identifier | Repository Location | Notes |
| :--- | :--- | :--- | :--- |
| **Model Family** | `SignTalk-ISL` | Top-level system name | Indian Sign Language Recognition & Translation |
| **Model Name** | `SignTalk-STGCN` | [`src/models/stgcn.py`](file:///d:/SignAI/src/models/stgcn.py) | 6-Block Spatial-Temporal Graph Convolutional Network |
| **Model Version** | `v1.0.0` | Production release | First stabilized ST-GCN graph architecture |
| **Experiment ID** | `stgcn_v1` | `experiments/stgcn/` | Primary full-dataset training experiment |
| **Configuration Version** | `configs/stgcn.yaml:v1.0` | [`configs/stgcn.yaml`](file:///d:/SignAI/configs/stgcn.yaml) | Configuration spec for training and architecture |
| **Dataset Release** | `signTalk-seq-v1.0.0` | `data/processed/sequences/` | 60 sequence archives ($36$ train, $12$ val, $12$ test) |
| **Landmark Schema** | `93-node-v1` | [`docs/phase-2-part-2/LANDMARK_DATA_SCHEMA.md`](file:///d:/SignAI/docs/phase-2-part-2/LANDMARK_DATA_SCHEMA.md) | 21 Left Hand, 21 Right Hand, 11 Pose, 40 Face |
| **Vocabulary Version** | `mvp_10:v1.0.0` | [`assets/vocabularies/mvp_10.json`](file:///d:/SignAI/assets/vocabularies/mvp_10.json) | 10 isolated core Indian Sign Language glosses |

---

## 3. Comparative Version Ledger

| Attribute | Baseline Reference (Phase 3 Part 1) | Spatial-Temporal Graph Model (Phase 3 Part 2) |
| :--- | :---: | :---: |
| **Model Name** | `SignTalk-Baseline-BiGRU` | `SignTalk-STGCN` |
| **Model Version** | `v1.0.0` | `v1.0.0` |
| **Experiment ID** | `baseline_bigru_v1` | `stgcn_v1` |
| **Parameter Count** | $597,898$ | $2,137,818$ |
| **Checkpoint Artifact**| `experiments/baseline/checkpoints/best_checkpoint.pt` | `experiments/stgcn/checkpoints/best_checkpoint.pt` |
