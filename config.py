import torch
from model import DiT
from pathlib import Path
from utils import TopoAlgoType

class DiTConfig:
    """
    a simple small config
    """
    def __init__(self, debug):
        self.debug = debug
        self.device = "cuda"

        # Data
        self.img_size = (101, 91) # Size of the input images
        self.dataset_size = -1 if not self.debug else 32

        # Model
        self.patch_size = 4   # Size of each patch
        self.dim = 768        # Embedding dimension
        self.depth = 12       # Number of transformer blocks
        self.heads = 12       # Number of attention heads
        self.mlp_dim = 3072   # Dimensions of the multilayer perceptron in the transformer block
        self.in_channels = 1  # Number of input channels
        self.dtype = torch.float32

        # Diffusion
        self.timesteps = 100 # Number of timesteps for diffusion
        self.beta_start = 0.0001
        self.beta_end = 0.02

        # Training
        self.epochs = 1000 if not self.debug else 1
        self.batch_size = 32
        self.lr = 0.001

        # Manifold matching configs
        self.mm_reg_alpha = 1.0
        self.data_path = Path("./data/photo/images.npy")
        self.topo_algo = TopoAlgoType["PCA"]
        self.n_components = 100 if not self.debug else 16

        # Logger
        self.log_step = 1

        # Sampling
        self.sample = True
        self.eval_step = 100
        self.n_gen = 16


    def start(self):
        return DiT(self)
