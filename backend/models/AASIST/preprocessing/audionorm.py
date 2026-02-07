import numpy as np

def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 2:
        return x.mean(axis=1)
    return x

def pad_or_crop(x: np.ndarray, target_len: int, train: bool, rng: np.random.Generator | None = None) -> np.ndarray:
    if len(x) == target_len:
        return x
    if len(x) < target_len:
        return np.pad(x, (0, target_len - len(x)), mode="constant")

    if train:
        if rng is None:
            rng = np.random.default_rng()
        start = int(rng.integers(0, len(x) - target_len + 1))
    else:
        start = (len(x) - target_len) // 2

    return x[start:start + target_len]
