from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import scipy.io as sio
import scipy.sparse
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


LABEL_KEYS = ("Y", "y", "gt", "gnd", "label", "labels", "truelabel")


def _unwrap_object(value):
    while isinstance(value, np.ndarray) and value.dtype == np.object_ and value.size == 1:
        value = value.flat[0]
    return value


def _to_dense_2d(value, n_samples: int) -> np.ndarray | None:
    value = _unwrap_object(value)
    if scipy.sparse.issparse(value):
        value = value.toarray()

    arr = np.asarray(value)
    if arr.dtype == np.object_:
        return None

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)

    if arr.shape[0] != n_samples and arr.shape[1] == n_samples:
        arr = arr.T

    if arr.shape[0] != n_samples:
        return None

    return arr.astype(np.float32)


def _extract_labels(mat: dict) -> np.ndarray:
    for key in LABEL_KEYS:
        if key in mat:
            labels = np.asarray(_unwrap_object(mat[key])).squeeze()
            if labels.dtype == np.object_ and labels.size > 0:
                labels = np.asarray(_unwrap_object(labels.flat[0])).squeeze()
            labels = labels.reshape(-1).astype(np.int64)
            min_label = int(labels.min())
            if min_label > 0:
                labels = labels - min_label
            return labels
    raise ValueError(f"DDR-MVC label key not found. Available keys: {[k for k in mat.keys() if not k.startswith('__')]}")


def _extract_view_candidates(mat: dict) -> List[np.ndarray]:
    x_keys = sorted(
        [k for k in mat.keys() if k.startswith("X") and len(k) > 1 and k[1:].isdigit()],
        key=lambda k: int(k[1:]),
    )
    if x_keys:
        return [mat[k] for k in x_keys]

    if "X" in mat:
        raw_x = mat["X"]
        if isinstance(raw_x, np.ndarray) and raw_x.dtype == np.object_:
            return [item for item in raw_x.flat]
        if isinstance(raw_x, np.ndarray) and raw_x.ndim == 3:
            return [raw_x[i] for i in range(raw_x.shape[0])]
        return [raw_x]

    raise ValueError(f"DDR-MVC view key not found. Available keys: {[k for k in mat.keys() if not k.startswith('__')]}")


class DDRMVCDataset(Dataset):
    """Minimal dataset wrapper used by DDR-MVC training."""

    def __init__(self, dataset_name: str, data_dir: str):
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.views: List[torch.Tensor] = []
        self.labels: torch.Tensor
        self._load()

    def _load(self) -> None:
        mat_path = self.data_dir / f"{self.dataset_name}.mat"
        if not mat_path.exists():
            raise FileNotFoundError(f"DDR-MVC dataset file not found: {mat_path}")

        mat = sio.loadmat(mat_path)
        labels = _extract_labels(mat)
        n_samples = labels.shape[0]
        raw_views = _extract_view_candidates(mat)

        for raw in raw_views:
            dense = _to_dense_2d(raw, n_samples)
            if dense is None:
                continue
            scaled = StandardScaler().fit_transform(dense).astype(np.float32)
            self.views.append(torch.from_numpy(scaled))

        if not self.views:
            raise ValueError(f"DDR-MVC could not parse any valid views from: {mat_path}")

        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        xs = [view[idx] for view in self.views]
        return xs, self.labels[idx]

    def get_dims(self) -> List[int]:
        return [int(view.shape[1]) for view in self.views]

    def get_num_clusters(self) -> int:
        return int(torch.unique(self.labels).numel())
