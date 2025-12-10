import enum
import numpy as np
from sklearn.metrics import pairwise_distances
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, TensorDataset, DataLoader
from PIL import Image
import os
from datetime import datetime
from pathlib import Path


def manifold_matching_reg(x, x_embed, alpha=1.0):
    """
    Compute a manifold matching regularization term between original data and embeddings.
    This encourages the embedding to preserve local neighborhood structure using Euclidean distances.

    Args:
        x: Original data tensor of shape (N, D1)
        x_embed: Embedded data tensor of shape (N, D2)
        alpha: Regularization weight

    Returns:
        Scalar tensor representing the regularization loss
    """

    N = x.shape[0]
    x = x.reshape(N, -1)
    xdist = torch.cdist(x, x)
    xdist = xdist / torch.norm(xdist, p="fro")
    x_embeddist = torch.cdist(x_embed, x_embed)
    x_embeddist = x_embeddist / torch.norm(x_embeddist, p="fro")
    reg_term = alpha * (1 / N ** 2) * torch.norm(xdist - x_embeddist, p="fro")

    return reg_term


class TopoAlgoType(enum.Enum):
    """
    Which type of topological data analysis algorithm to use.
    """
    PCA = enum.auto()
    UMAP = enum.auto()
    TSNE = enum.auto()


def get_dataloader(data_path: Path, dtype, topo_algo: TopoAlgoType,
        n_components: int, batch_size: int, dataset_size: int):
    data = np.load(data_path)[:dataset_size]
    if data.ndim < 4:
        data = np.reshape(data, (data.shape[0], 1) + data.shape[1:])
    topo_repr = topo_representation(data, topo_algo, n_components)
    data = torch.from_numpy(data).to(dtype)

    # pad
    pad_fn = PadToDivisible(divisor=8)
    data = pad_fn(data)

    topo_repr = torch.from_numpy(topo_repr).to(dtype)
    dataset = TensorDataset(data, topo_repr)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


class PadToDivisible(nn.Module):
    """
    Pad image to make dimensions divisible by a given divisor.
    Useful for models that require specific dimension constraints (e.g., VAE with divisor=8).

    Args:
        divisor: The number by which dimensions should be divisible (default: 8)
        fill: Pixel fill value for padding (default: 0)
        padding_mode: Padding mode - 'constant', 'edge', 'reflect', 'symmetric' (default: 'constant')
    """

    def __init__(self, divisor=8, fill=0, padding_mode='constant'):
        super().__init__()
        self.divisor = divisor
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, img):
        """
        Args:
            img: PIL Image or Tensor

        Returns:
            Padded PIL Image or Tensor
        """
        if isinstance(img, torch.Tensor):
            h, w = img.shape[-2:]
            is_tensor = True
        else:
            w, h = img.size
            is_tensor = False

        # Calculate padding needed
        pad_h = (self.divisor - h % self.divisor) % self.divisor
        pad_w = (self.divisor - w % self.divisor) % self.divisor

        # Distribute padding evenly on both sides
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        if is_tensor:
            # Use torch padding for tensors
            padding = (pad_left, pad_right, pad_top, pad_bottom)
            return torch.nn.functional.pad(img, padding, mode=self.padding_mode, value=self.fill)
        else:
            # Use PIL padding for images
            padding = (pad_left, pad_top, pad_right, pad_bottom)
            return transforms.functional.pad(img, padding, fill=self.fill, padding_mode=self.padding_mode)

    def __repr__(self):
        return f"{self.__class__.__name__}(divisor={self.divisor}, fill={self.fill}, padding_mode='{self.padding_mode}')"


class CombinedDataset(Dataset):
    def __init__(self, dataset_a, dataset_b):
        assert len(dataset_a) == len(
            dataset_b), "Datasets must have same length"
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b

    def __len__(self):
        return len(self.dataset_a)

    def __getitem__(self, idx):
        return self.dataset_a[idx], self.dataset_b[idx]


class UnlabeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx][0]


class TensorWrapper(Dataset):
    def __init__(self, tensor):
        self.tensor = tensor

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return self.tensor[idx]


def topo_representation(data: np.ndarray, algo: TopoAlgoType, n_dimensions:
                        int = 2) -> np.ndarray:
    """
    Generate topological data augmentation using the specified algorithm.

    Parameters:
    data (np.ndarray): The input data to augment.
    algo (TopoAlgoType): The topological data analysis algorithm to use.
    n_dimensions (int): Number of dimensions for the output data.
    """

    N = data.shape[0]
    data = np.reshape(data, (N, -1))

    if algo == TopoAlgoType.PCA:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_dimensions)
        transformed_data = pca.fit_transform(data)
    elif algo == TopoAlgoType.UMAP:
        import umap
        reducer = umap.UMAP(n_components=n_dimensions)
        transformed_data = reducer.fit_transform(data)
    elif algo == TopoAlgoType.TSNE:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=n_dimensions)
        transformed_data = tsne.fit_transform(data)

    # compute the pairwise distances in the transformed space
    distances = pairwise_distances(transformed_data)

    return distances


def save_npy_images_as_png(npy_path: str = "data/photo/images.npy", output_dir: str = "data/photo/class1"):
    """
    Load images from a .npy file and save each one as a PNG file.

    Parameters:
    npy_path (str): Path to the .npy file containing images.
    output_dir (str): Directory where PNG files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    images = np.load(npy_path)

    for i, img in enumerate(images):
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.uint8)

        if len(img.shape) == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))

        if img.shape[-1] == 1:
            img = img.squeeze(-1)

        pil_img = Image.fromarray(img)
        pil_img.save(os.path.join(output_dir, f"image_{i:04d}.png"))

    print(f"Saved {len(images)} images to {output_dir}")


def create_run_directory():
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("output", run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    samples_dir = os.path.join(run_dir, "samples")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)
    return checkpoint_dir, samples_dir


if __name__ == "__main__":
    save_npy_images_as_png()
