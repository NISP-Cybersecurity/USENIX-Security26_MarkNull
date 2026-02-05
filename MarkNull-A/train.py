from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import csv
import os

from WRN import Restormer
import argparse
import os
import torch
import random 
from pytorch_lightning import Trainer
import time
from pytorch_lightning.callbacks import LearningRateMonitor
from utils.dataload import WatermarkDataModule


torch.autograd.set_detect_anomaly(True)
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # Set to False to ensure complete determinism

def get_device(gpu_id):
    return f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'


def parse_train_args():
    parser = argparse.ArgumentParser(description="Train Noise Estimation Model")


    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (default: 300)')
    parser.add_argument('--lr', '--learning_rate', type=float, default=0.0005,
                        help='Initial learning rate (default: 0.001)')
    parser.add_argument('--batch_size', type=int, default=5,
                        help='Batch size for data loaders (default: 100)')

    parser.add_argument('--output_dir', type=str, default='onlyhigh_logs',
                        help='Base directory for saving logs and models (default: logs)')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Optional name for this run (used in log/model paths)')

    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to checkpoint file to resume training from')
    parser.add_argument('--auto_resume', action='store_true',
                        help='Automatically resume from the latest checkpoint in the log directory')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use (default: 0)')



    args = parser.parse_args()
    return args


def one_train(args):
    set_seed(args.seed)

    device = get_device(args.gpu_id)
    print(f"Using device: {device}")
    print("Creating data loaders...")
    
    ATTACKED_PATH = "./Dataset/Watermark/SD1.5"
    WATERMARKED_PATH = "./Dataset/Watermark/SD1.5"

    dm = WatermarkDataModule(
        clean_root=ATTACKED_PATH,
        watermarked_root=WATERMARKED_PATH,
        batch_size=7,
        patch_size=256
    )

    dm.setup()
    
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    
    print("Data loaders created.")



    model = Restormer(args.lr, args.epochs).to(device)
    logger = TensorBoardLogger(save_dir=args.output_dir, name=os.path.basename(args.log_path))
    print(f"TensorBoard logs will be saved to: {logger.log_dir}")
    
    checkpoint_path = None
    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        print(f"Resuming training from specified checkpoint: {checkpoint_path}")
    elif args.auto_resume:
        checkpoint_dir = os.path.join(logger.log_dir, 'checkpoints')
        if os.path.exists(checkpoint_dir):
            last_ckpt = os.path.join(checkpoint_dir, 'last.ckpt')
            if os.path.exists(last_ckpt):
                checkpoint_path = last_ckpt
                print(f"Resuming training from the latest checkpoint: {checkpoint_path}")
            else:
                print("No last.ckpt found, starting training from scratch.")
        else:
            print("Checkpoint directory does not exist, starting training from scratch.")
    
    # Verify if checkpoint file exists
    if checkpoint_path and not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint file {checkpoint_path} does not exist, starting training from scratch.")
        checkpoint_path = None
    
    # 8. Setup checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', # Monitor validation loss
        dirpath=os.path.join(logger.log_dir, 'checkpoints'), # Save to 'checkpoints' directory inside log directory
        filename=f"{args.base_save_name}-{{epoch:02d}}-{{val_loss:.4f}}", # Filename format
        save_top_k=1, # Save the best model
        mode='min', # 'min' mode (lower metric is better)
        save_last=True # Always save the last epoch model
    )
    
    # 9. Initialize Trainer
    print("Initializing Pytorch Lightning Trainer...")
    trainer = Trainer(
        max_epochs=args.epochs,
        devices=[args.gpu_id] if torch.cuda.is_available() else None,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=logger,
        callbacks=checkpoint_callback,
        val_check_interval=1.0,
        # Log model structure and other info
        log_every_n_steps=50,
    )
    
    print("Trainer initialized.")

  
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
   
    
    model_path = os.path.join("trained_model", f"Restormer.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")


if __name__ == '__main__':
    
    args = parse_train_args()
   
    one_train(args)
    #progressive_training(args)
