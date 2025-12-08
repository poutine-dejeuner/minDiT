import torch
from config import DiTConfig
from diffusion import Diffusion

def test_diffusion_size_mismatch():
    """Test that reproduces the RuntimeError with tensor size mismatch"""
    cfg = DiTConfig()
    cfg.device = "cpu"
    diffusion = Diffusion(cfg)
    batch_size = 32
    x0 = torch.randn(batch_size, 1, 28, 28)  # 28x28 images
    t = torch.randint(0, cfg.timesteps, (batch_size,))
    
    xt, noise = diffusion.diffuse(x0, t)
    selected = diffusion.sqrt_alphas_cumprod[t]

if __name__ == "__main__":
    test_diffusion_size_mismatch()
