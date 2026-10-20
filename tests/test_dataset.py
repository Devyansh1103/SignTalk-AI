

def test_live_robustness_augmentations():
    """Rest-hand dropout hides only low hands; low-FPS hold repeats frames; masked nodes stay 0."""
    import numpy as np
    from src.data.sign_sequence_dataset import SignSequenceDataset

    ds = SignSequenceDataset.__new__(SignSequenceDataset)
    ds.rng = np.random.default_rng(0)
    data = np.zeros((3, 45, 93), np.float32)
    mask = np.ones((1, 45, 93), np.float32)
    data[1, :, 0:21] = 2.0          # left hand resting at the hips
    data[1, 20:25, 0:21] = 0.5      # ...raised for 5 frames
    data[1, :, 21:42] = 0.8         # right hand always signing
    d, m = ds._drop_rest_hands(data, mask, p=1.0)
    assert m[0, 20:25, 0:21].all() and not m[0, :20, 0:21].any() and not m[0, 25:, 0:21].any()
    assert m[0, :, 21:42].all() and (d[:, :20, 0:21] == 0).all()

    data[0] = np.arange(45, dtype=np.float32)[:, None]
    d, _ = ds._low_fps(data, mask, p=1.0)
    assert len(np.unique(d[0, :, 50])) < 45
