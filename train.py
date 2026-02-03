#!/usr/bin/env python
"""
Training Script for Rough Volatility Neural SDE.

This script trains a Neural SDE driven by fractional Brownian motion
to match the distribution of real market paths using signature-based loss.

Usage:
    python train.py --epochs 100 --lr 0.001 --H 0.1
    python train.py --epochs 100 --lr 0.001 --learnable-H  # Learn H

See docs/train.md for details.
"""

import argparse
import sys
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from noise.fbm import DaviesHarte
from models.generator import RoughNeuralSDE
from loss.signature import DifferentiablePathLoss
from data.market_data import MarketDataset, get_synthetic_data


class TrainableH(nn.Module):
    """
    Learnable Hurst parameter constrained to (0, 0.5) via sigmoid.
    
    The Hurst exponent H controls the roughness of fBM:
    - H = 0.5: Standard Brownian motion
    - H < 0.5: Rough (anti-persistent) paths
    - H > 0.5: Smooth (persistent) paths
    
    For rough volatility, we constrain H ∈ (0, 0.5).
    """
    
    def __init__(self, init_H: float = 0.1):
        super().__init__()
        # Initialize so that sigmoid(h_raw) * 0.5 ≈ init_H
        # sigmoid^{-1}(init_H / 0.5) = logit(2 * init_H)
        init_raw = np.log(2 * init_H / (1 - 2 * init_H))
        self.h_raw = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))
    
    def forward(self) -> float:
        """Return H constrained to (0, 0.5)."""
        return torch.sigmoid(self.h_raw).item() * 0.5


class RoughVolatilityTrainer:
    """
    Trainer for the Rough Volatility Neural SDE.
    
    Parameters
    ----------
    hidden_dim : int
        Hidden dimension for the neural networks.
    H : float or None
        Fixed Hurst parameter. If None, H is learned.
    sig_depth : int
        Signature truncation depth for loss.
    n_steps : int
        Number of time steps for generated paths.
    device : str
        Device to train on ('cpu' or 'cuda').
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        H: float = None,
        sig_depth: int = 4,
        n_steps: int = 50,
        device: str = 'cpu'
    ):
        self.device = device
        self.n_steps = n_steps
        self.H_fixed = H
        
        # Initialize model
        self.model = RoughNeuralSDE(hidden_dim=hidden_dim).to(device)
        
        # Initialize learnable H if not fixed
        if H is None:
            self.trainable_H = TrainableH(init_H=0.1)
            self.H_is_learnable = True
        else:
            self.trainable_H = None
            self.H_is_learnable = False
        
        # Initialize loss - use differentiable loss for gradient flow
        self.loss_fn = DifferentiablePathLoss(include_autocorr=True, max_lag=5)
        
        # Training history
        self.history = {
            'loss': [],
            'H': [],
            'epoch': []
        }
    
    def get_H(self) -> float:
        """Get current Hurst parameter value."""
        if self.H_is_learnable:
            return self.trainable_H()
        return self.H_fixed
    
    def create_fbm_generator(self, batch_size: int) -> DaviesHarte:
        """Create fBM generator with current H."""
        H = self.get_H()
        return DaviesHarte(
            n_steps=self.n_steps,
            batch_size=batch_size,
            H=H,
            T=1.0
        )
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: optim.Optimizer
    ) -> float:
        """
        Train for one epoch.
        
        Parameters
        ----------
        dataloader : DataLoader
            DataLoader for real market paths.
        optimizer : optim.Optimizer
            Optimizer for model parameters.
        
        Returns
        -------
        float
            Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for real_paths in dataloader:
            real_paths = real_paths.to(self.device)
            batch_size = real_paths.shape[0]
            
            # Create fBM generator
            fbm_gen = self.create_fbm_generator(batch_size)
            
            # Generate paths
            gen_paths = self.model(fbm_gen)
            
            # Compute loss
            loss = self.loss_fn(real_paths, gen_paths)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def train(
        self,
        dataloader: DataLoader,
        n_epochs: int = 100,
        lr: float = 1e-3,
        print_every: int = 10,
        save_dir: str = None
    ):
        """
        Full training loop.
        
        Parameters
        ----------
        dataloader : DataLoader
            DataLoader for real market paths.
        n_epochs : int
            Number of training epochs.
        lr : float
            Learning rate.
        print_every : int
            Print progress every N epochs.
        save_dir : str, optional
            Directory to save checkpoints.
        """
        # Collect parameters for optimization
        params = list(self.model.parameters())
        if self.H_is_learnable:
            params.append(self.trainable_H.h_raw)
        
        optimizer = optim.Adam(params, lr=lr)
        
        print(f"Training Rough Volatility Neural SDE")
        print(f"  - Epochs: {n_epochs}")
        print(f"  - Learning rate: {lr}")
        print(f"  - H: {'learnable' if self.H_is_learnable else self.H_fixed}")
        print(f"  - Device: {self.device}")
        print("-" * 50)
        
        start_time = time.time()
        
        for epoch in range(n_epochs):
            epoch_loss = self.train_epoch(dataloader, optimizer)
            current_H = self.get_H()
            
            # Record history
            self.history['loss'].append(epoch_loss)
            self.history['H'].append(current_H)
            self.history['epoch'].append(epoch)
            
            if (epoch + 1) % print_every == 0 or epoch == 0:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch+1:4d} | Loss: {epoch_loss:.6f} | "
                      f"H: {current_H:.4f} | Time: {elapsed:.1f}s")
            
            # Save checkpoint
            if save_dir is not None and (epoch + 1) % 50 == 0:
                self.save_checkpoint(save_dir, epoch + 1)
        
        print("-" * 50)
        print(f"Training complete. Final loss: {self.history['loss'][-1]:.6f}")
        
        # Save final model
        if save_dir is not None:
            self.save_checkpoint(save_dir, 'final')
    
    def save_checkpoint(self, save_dir: str, tag: str):
        """Save model checkpoint."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'H': self.get_H(),
            'history': self.history,
        }
        
        if self.H_is_learnable:
            checkpoint['trainable_H_state_dict'] = self.trainable_H.state_dict()
        
        path = save_dir / f"checkpoint_{tag}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.H_is_learnable and 'trainable_H_state_dict' in checkpoint:
            self.trainable_H.load_state_dict(checkpoint['trainable_H_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        print(f"Loaded checkpoint from {path}")


def main():
    parser = argparse.ArgumentParser(description="Train Rough Volatility Neural SDE")
    
    # Data arguments
    parser.add_argument('--ticker', type=str, default='^GSPC',
                        help='Yahoo Finance ticker (default: ^GSPC)')
    parser.add_argument('--start-date', type=str, default='2015-01-01',
                        help='Training data start date')
    parser.add_argument('--window-size', type=int, default=50,
                        help='Path window size')
    parser.add_argument('--use-synthetic', action='store_true',
                        help='Use synthetic data instead of real market data')
    
    # Model arguments
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension for networks')
    parser.add_argument('--H', type=float, default=0.1,
                        help='Fixed Hurst parameter (ignored if --learnable-H)')
    parser.add_argument('--learnable-H', action='store_true',
                        help='Make Hurst parameter learnable')
    parser.add_argument('--sig-depth', type=int, default=4,
                        help='Signature truncation depth')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--print-every', type=int, default=10,
                        help='Print progress every N epochs')
    
    # Output arguments
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Create dataset
    print("Loading data...")
    if args.use_synthetic:
        print("Using synthetic data")
        data = get_synthetic_data(n_samples=5000, n_steps=args.window_size)
        dataset = torch.utils.data.TensorDataset(data)
        # Wrapper to return just the tensor
        class SyntheticDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]
        dataset = SyntheticDataset(data)
    else:
        try:
            dataset = MarketDataset(
                ticker=args.ticker,
                start_date=args.start_date,
                window_size=args.window_size
            )
            print(f"Loaded {len(dataset)} market path windows from {args.ticker}")
        except Exception as e:
            print(f"Failed to load market data: {e}")
            print("Falling back to synthetic data")
            data = get_synthetic_data(n_samples=5000, n_steps=args.window_size)
            class SyntheticDataset(torch.utils.data.Dataset):
                def __init__(self, data):
                    self.data = data
                def __len__(self):
                    return len(self.data)
                def __getitem__(self, idx):
                    return self.data[idx]
            dataset = SyntheticDataset(data)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )
    
    # Create trainer
    H = None if args.learnable_H else args.H
    trainer = RoughVolatilityTrainer(
        hidden_dim=args.hidden_dim,
        H=H,
        sig_depth=args.sig_depth,
        n_steps=args.window_size,
        device=args.device
    )
    
    # Train
    trainer.train(
        dataloader=dataloader,
        n_epochs=args.epochs,
        lr=args.lr,
        print_every=args.print_every,
        save_dir=args.save_dir
    )
    
    print("\nTraining complete!")
    print(f"Final H: {trainer.get_H():.4f}")


if __name__ == "__main__":
    main()
