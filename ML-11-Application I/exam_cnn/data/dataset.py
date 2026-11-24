"""Load A-Z images for PyTorch, with augmentation during training only."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from exam_cnn.data.augmentation import augment_image
from exam_cnn.data.preprocessing import normalize_pixels

DEFAULT_BATCH_SIZE = 128


class AZCharacterDataset(Dataset):
    def __init__(
        self, X: np.ndarray, y: np.ndarray, train: bool = False, seed: int = 42
    ):
        self.X = X  ## uint8, character-normalized, shape (N, 28, 28)
        self.y = y.astype(np.int64)
        self.train = train
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        img = self.X[idx]
        if self.train:
            img = augment_image(img, self._rng)
        img = normalize_pixels(img)[None, :, :]  ## -> (1, 28, 28) float32
        return torch.from_numpy(img), int(self.y[idx])


def make_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    train: bool,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    dataset = AZCharacterDataset(X, y, train=train, seed=seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )
