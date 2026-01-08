import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from model import DiT
from config import DiTConfig
from utils import (PadToDivisible, mmreg, manifold_matching_reg, topo_representation,
                   create_run_directory, get_dataset_and_topo_repr,
                   IndexDataset)
import wandb
import os
from tqdm import tqdm

from icecream import ic, install
import sys
import builtins

install()
builtins.sys = sys

def sample_model(model, epoch, samples_dir, cfg):
    print('Sampling')
    samples = model.sample(num_samples=cfg.n_gen, steps=cfg.timesteps)
    sample_path = os.path.join(samples_dir, f'epoch_{epoch}.png')
    save_image(samples, sample_path, normalize=True)
    # wandb.log({"samples": wandb.Image(sample_path)})
    del samples
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def train_model(args):
    # cfguration
    cfg = DiTConfig(args.d)

    # initialize wandb
    wandb.init(entity='nanophoto', project="minDiT", config=vars(cfg))
    checkpoint_dir, samples_dir = create_run_directory()

    # data
    dataset, topo_repr = get_dataset_and_topo_repr(cfg.data_path, cfg.dtype,
                                cfg.topo_algo,
                                cfg.n_components,
                                cfg.dataset_size)
    cfg.img_size = tuple(dataset.tensors[0].shape[-2:])
    dataset = IndexDataset(dataset)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size)


    # model
    model = DiT(cfg).to(cfg.dtype).to(cfg.device)
    
    # Load checkpoint from most recent output subdirectory if exists
    import glob
    output_dirs = sorted(glob.glob('output/*/'), key=os.path.getmtime, reverse=True)
    if output_dirs:
        checkpoint_path = os.path.join(output_dirs[0], 'checkpoint.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=cfg.device)
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f'Loaded checkpoint from {checkpoint_path}')
            except:
                print('Could not load checkpoint')

    # loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)


    # training loop
    for epoch in range(cfg.epochs):
        epoch_loss = 0
        # for x, topo_repr in tqdm(dataloader, disable=not cfg.debug):
        for (x,), indices in tqdm(dataloader):
            x = x.to(cfg.device)
            batch_topo_repr = topo_repr[indices][:, indices].to(cfg.device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # random input and target
            t = torch.randint(0, cfg.timesteps, (x.shape[0],),
                    device=cfg.device)

            # forward pass where we get the noisy image and the noise
            xt, noise = model.diffusion.diffuse(x, t)
            output = model(xt, t)

            # loss calculation to get the loss between the noise and the output
            loss = criterion(output, noise)
            reg = mmreg(model.last_repr(), batch_topo_repr)
            loss = loss + reg
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            # Clear references to prevent memory leak
            del xt, noise, output, loss
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            if cfg.debug:
                break

        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch}, Loss: {avg_loss}')

        # log to wandb
        if epoch % cfg.log_step == 0:
            wandb.log({"epoch": epoch, "loss": avg_loss})

        # save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, os.path.join(checkpoint_dir, 'checkpoint.pt'))

        # sample and save images
        if epoch % cfg.eval_step == 0 and cfg.sample:
            sample_model(model, epoch, samples_dir, cfg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', action="store_true", default=False)
    args = parser.parse_args()
    train_model(args)
