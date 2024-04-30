"""Fast I/O functions to load data into memory."""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from typing import List, Callable, Union, Tuple, Any, Optional, Dict


def load_raw_features_slide(
    features_path: Union[str, Path], n_tiles: int, extract_slidename: Callable
):
    """Load raw features slide."""
    X_slidename = extract_slidename(Path(features_path))
    X = np.load(features_path, mmap_mode="r", allow_pickle=True).astype(np.float32)
    if n_tiles:
        X = X[:n_tiles].copy()
    X_coords = X[:, :3]
    X = X[:, 3:]
    return X, X_coords, X_slidename


def load_features_to_mem(
    features_paths: List[Path],
    extract_slidename: Callable,
    extract_id: Callable,
    n_tiles: int,
    num_workers: int,
    as_list: bool = False,
):
    """Load features in a concurrent fashion from disk into memory.

    Parameters
    ----------
    features_paths : list
        Paths to features.

    extract_slidename : Callable
        Function to extract slidename from the file path.

    extract_id : Callable
        Function to extract slide id from slidename.

    n_tiles : int
        Number of tiles to include in every slide.

    num_workers : int
        Number of workers.

    as_list : bool
        If True, outputs X and X_coords as list for space efficiency.

    Returns
    -------
    X : array, shape (n_samples, n_tiles, embed_dim)
        Features.

    X_coords : array, shape (n_samples, n_tiles, 3)
        X, Y coordinates of every tile.

    X_slidenames : array, shape (n_samples)
        Slide names.

    X_ids : array, shape (n_samples)
        Slide IDs.

    """
    # Pre-allocating a large array to store all slides and avoid I/O bottleneck
    if as_list:
        X, X_coords = [], []  # List[np.ndarray], List[np.ndarray]
    else:
        # Load one sample to get dimension
        dim = np.load(features_paths[0], mmap_mode="r").shape[1] - 3
        X = np.zeros((len(features_paths), n_tiles, dim), dtype=np.float32)
        X_coords = np.zeros((len(features_paths), n_tiles, 3), dtype=np.float32)

    with tqdm(total=len(features_paths)) as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            running_tasks = [
                executor.submit(load_raw_features_slide, path, n_tiles, extract_slidename)
                for path in features_paths
            ]
            X_slidenames = []
            for i, running_task in enumerate(running_tasks):
                x, x_coords, x_slidename = running_task.result()
                if as_list:
                    X.append(x)
                    X_coords.append(x_coords)
                else:
                    X[i, : len(x), :] = x
                    X_coords[i, : len(x), :] = x_coords
                X_slidenames.append(x_slidename)
                pbar.update(1)

    X_slidenames = np.squeeze(X_slidenames)
    X_ids = np.squeeze([extract_id(s) for s in X_slidenames])
    return X, X_coords, X_slidenames, X_ids


@lru_cache(maxsize=None)
def load_ckpt(model_filename, device):
    """Load checkpoint."""
    return torch.load(model_filename, map_location=device)


def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))


def pad_collate_fn(
    batch: List[Tuple[torch.Tensor, Any]],
    batch_first: bool = True,
    max_len: Optional[int] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    Pads together sequences of arbitrary lengths
    Adds a mask of the padding to the samples that can later be used
    to ignore padding in activation functions.

    Expected to be used in combination of a torch.utils.datasets.DataLoader.

    Expect the sequences to be padded to be the first one in the sample tuples.
    Others members will be batched using default_collate

    Parameters
    ----------
    batch: List[Tuple[torch.Tensor, Any]]
    batch_first: bool = True
        Either return (B, N_TILES, F) or (N_TILES, B, F)
    max_len: int

    Returns
    -------
    padded_sequences, masks, Any: Tuple[torch.Tensor, torch.BoolTensor, Any]
        - if batch_first: Tuple[(B, N_TILES, F), (B, N_TILES, 1), ...]
        - else: Tuple[(N_TILES, B, F), (N_TILES, B, 1), ...]

        with N_TILES = max_len if max_len is not None
        or N_TILES = max length of the training samples.

    """
    sequences = []
    others = []
    for sample in batch:
        sequences.append(sample[0])
        others.append(sample[1:])

    if max_len is None:
        max_len = max([s.size(0) for s in sequences])

    trailing_dims = sequences[0].size()[1:]

    if batch_first:
        padded_dims = (len(sequences), max_len) + trailing_dims
        masks_dims = (len(sequences), max_len, 1)
    else:
        padded_dims = (max_len, len(sequences)) + trailing_dims
        masks_dims = (max_len, len(sequences), 1)

    padded_sequences = sequences[0].data.new(*padded_dims).fill_(0.0)
    masks = torch.ones(*masks_dims, dtype=torch.bool)

    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        if batch_first:
            padded_sequences[i, :length, ...] = tensor[:max_len, ...]
            masks[i, :length, ...] = False
        else:
            padded_sequences[:length, i, ...] = tensor[:max_len, ...]
            masks[:length, i, ...] = False

    others = default_collate(others)

    return (padded_sequences, masks, *others)


class SlideFeaturesDataset(Dataset):
    """
    SlideFeaturesDataset.

    From either a list of path to numpy files or loaded numpy features,
    create a `torch.utils.data.Dataset` that samples over the features and labels.

    Parameters
    ----------
    features: Union[np.array, List[Path], List[np.array]]
        Either a list of path to numpy files or loaded numpy features.
    labels: Union[np.array, List[Any]]
    metadata: Optional[List[Dict]] = None
        Optional metadata.
    """

    def __init__(
        self,
        features: Union[np.array, List[Path], List[np.array]],
        labels: Union[np.array, List[Any]],
        metadata: Optional[List[Dict]] = None,
    ):
        if len(features) != len(labels):
            raise ValueError(
                f"features and labels must have the same length.\
            Given {len(features)} and {len(labels)}."
            )

        if metadata is not None and len(metadata) != len(features):
            raise ValueError(
                f"features, labels and metadata must have the same length.\
            Given {len(features)}, {len(labels)}, {len(metadata)}."
            )

        self.features = features
        self.labels = np.array(labels)
        self.labels = (
            np.expand_dims(labels, axis=1) if self.labels.ndim == 1 else self.labels
        )  # set array shape to (n_labels, n_classes)
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(
        self, item: int
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict]]:
        """

        Parameters
        ----------
        item: int
            Index of item

        Returns
        -------
        Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict]]
            - if metadata is not None: (N_TILES, N_FEATURES), (), {}
            - else (N_tiles, N_FEATURES), ()
        """
        slide_features = self.features[item]
        slide_label = torch.tensor(self.labels[item])

        # Load the whole np.array
        if isinstance(slide_features, Path) or isinstance(slide_features, str):
            slide_features = np.load(slide_features)

        slide_features = torch.from_numpy(slide_features.astype(np.float32))

        if self.metadata is not None:
            slide_metadata = self.metadata[item]

            return slide_features, slide_label, slide_metadata

        return slide_features, slide_label
