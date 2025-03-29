import os
import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

from lib.NormFlow import NormalizingFlowModel
from lib.data_utils import createJetData


def prepare_dataloader(args):
    # Create jet data using the provided utility
    j2_data, j2_detlabel, _, _ = createJetData(1, False, seed=78,root_dir=args.root_dir)

    # Convert data to tensors
    S_tensor = torch.tensor(j2_data[j2_detlabel == 1], dtype=torch.float32)
    BG_tensor = torch.tensor(j2_data[j2_detlabel == 0], dtype=torch.float32)

    # Equalize dataset size between signal and background
    max_size = np.min([len(S_tensor), len(BG_tensor)])
    if args.s:
        dataset = TensorDataset(S_tensor[:max_size], BG_tensor[:max_size])
    else:
        dataset = TensorDataset(BG_tensor[:max_size],S_tensor[:max_size])

    # Split dataset into training and validation sets (80/20 split)
    n_val = int(0.2 * len(dataset))
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])

    batch_size = 1000
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Calculate mean and standard deviation of the signal tensor
    X_mean = torch.mean(S_tensor, dim=0)
    X_std = torch.std(S_tensor, dim=0)

    return train_loader, val_loader, X_mean, X_std

def main(args):
    # Setup root directory and add necessary paths
    root_dir = args.root_dir
    print("HEP Challenge Root directory is", root_dir)

    train_loader, val_loader, X_mean, X_std = prepare_dataloader(args)

    # Setup callbacks: early stopping and model checkpointing
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=100, mode="min"
    )
    if args.s:
        file_name = f"s_advNF_c_{args.c}"
    else:
        file_name = f"/bg_advNF_c_{args.c}"

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        filename=file_name
    )

    bg_model = NormalizingFlowModel(
        c=args.c,
        input_dim=20,
        clamp_val=-50,
        n_layers=30,
        X_mean=X_mean,
        X_std=X_std
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu' 

    # Create the trainer and fit the model
    trainer = pl.Trainer(
        max_epochs=500,default_root_dir=args.checkpoint_path,
        log_every_n_steps=10,
        accelerator=device,
        callbacks=[checkpoint_callback, early_stop_callback]
    )
    trainer.fit(bg_model, train_loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Normalizing Flow Model")
    parser.add_argument(
        "--root-dir",
        type=str,
        default="/Users/ibrahim/HEP-Challenge/",
        help="Root directory containing input_data, ingestion_program, and scoring_program folders"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="Test/",
        help="Path to a model checkpoint to load (optional)"
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1,
    )

    parser.add_argument(
        "-s",
        type=bool,
        default=True,
    )
    args = parser.parse_args()
    main(args)