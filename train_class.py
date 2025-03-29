import os
import argparse
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Import custom libraries
from lib.class_model import CombinedClassifier
from lib.data_utils import createMultiJetMultiNuanData, Dataset1j2j, load_nf_models

def main(args):
    # Load Normalizing Flow models from the given directory.
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    nf_models = load_nf_models(args.models_dir,device)
    if len(nf_models) != 8:
        raise ValueError("Expected to load exactly 8 models (4 for 1-jet and 4 for 2-jet).")

    # Create multi-jet data using the provided root directory.
    # j2 corresponds to 2-jet data and j1 corresponds to 1-jet data.
    j2_data, j2_detlabel, _, _ = createMultiJetMultiNuanData(2, False, seed=0, root_dir=args.root_dir)
    j1_data, j1_detlabel, _, _ = createMultiJetMultiNuanData(1, False, seed=0, root_dir=args.root_dir)
    
    # Extract features from the loaded models. For 1-jet models, indices 0-3 are used.
    # For 2-jet models, indices 4-7 are used.
    with torch.no_grad():
        # Process 1-jet data.
        NF_feat_s1j    = torch.sigmoid(nf_models[0](j1_data)).cpu().unsqueeze(1)
        NF_feat_b1j    = torch.sigmoid(nf_models[1](j1_data)).cpu().unsqueeze(1)
        NF_feat_s1j_3  = torch.sigmoid(nf_models[2](j1_data)).cpu().unsqueeze(1)
        NF_feat_b1j_3  = torch.sigmoid(nf_models[3](j1_data)).cpu().unsqueeze(1)
        
        # Process 2-jet data.
        NF_feat_s2j    = torch.sigmoid(nf_models[4](j2_data)).cpu().unsqueeze(1)
        NF_feat_b2j    = torch.sigmoid(nf_models[5](j2_data)).cpu().unsqueeze(1)
        NF_feat_s2j_3  = torch.sigmoid(nf_models[6](j2_data)).cpu().unsqueeze(1)
        NF_feat_b2j_3  = torch.sigmoid(nf_models[7](j2_data)).cpu().unsqueeze(1)
    
        # Append the Normalizing Flow features to the original data.
        j1_data = torch.cat([j1_data, NF_feat_s1j, NF_feat_s1j_3, NF_feat_b1j, NF_feat_b1j_3], dim=1)
        j2_data = torch.cat([j2_data, NF_feat_s2j, NF_feat_s2j_3, NF_feat_b2j, NF_feat_b2j_3], dim=1)
    
    # Ensure the number of data points match by using the minimum length.
    max_shape = min(len(j1_data), len(j2_data))
    print(f"Number of data points used: {max_shape}")
    j1_data = j1_data[:max_shape]
    j2_data = j2_data[:max_shape]
    j1_detlabel = j1_detlabel[:max_shape]
    j2_detlabel = j2_detlabel[:max_shape]

    # Create a dataset for the combined jets.
    all_jet_dataset = Dataset1j2j(j2_data, j1_data, j2_detlabel, j1_detlabel)

    # Split the dataset into training and validation sets.
    n_val = int(0.1 * len(all_jet_dataset))
    n_train = len(all_jet_dataset) - n_val
    train_dataset, val_dataset = random_split(all_jet_dataset, [n_train, n_val])
    
    # Create DataLoaders.
    batch_size = 1000
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Setup callbacks: early stopping and model checkpointing.
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=50,
        verbose=False,
        mode="min"
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",     # Metric to monitor.
        mode="min",             # Lower validation loss is better.
        save_top_k=3,           # Save only the best 3 models.
        filename="DNNclass"
    )

    # Initialize the classifier model.
    class_model = CombinedClassifier()

    # Initialize the PyTorch Lightning trainer.
    trainer = pl.Trainer(
        max_epochs=500,default_root_dir=args.checkpoint_path,
        log_every_n_steps=10,
        accelerator=device,
        callbacks=[checkpoint_callback, early_stop_callback]
    )
    
    # Start training.
    trainer.fit(class_model, train_loader, val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CombinedClassifier with appended Normalizing Flow features")
    # Argument for the root directory for data.
    parser.add_argument(
        "--root_dir", 
        type=str, 
        default="/Users/ibrahim/HEP-Challenge/",
        help="Root directory path for data files."
    )
    # Argument for the directory containing Normalizing Flow model checkpoints.
    parser.add_argument(
        "--models_dir", 
        type=str, 
        default="PreTrained/Models/",
        help="Path to the directory containing the '1_jet' and '2_jet' subdirectories with checkpoint files."
    )
    # Argument for saving checkpoints during training.
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        default="Test/",
        help="Directory where model checkpoints will be saved during training."
    )
    
    args = parser.parse_args()
    main(args)