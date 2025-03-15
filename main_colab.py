import os
import torch
import config
import logging
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint 
import torch.multiprocessing as mp

from helpers.utils import create_path
from helpers.data_processor import ESC_Dataset
from helpers.data_loader import DataPrerparation
from models.sed_model import SEDWrapper
from models.htsat import HTSAT_Swin_Transformer

device_num = torch.cuda.device_count()

# create folder to save result  
exp_dir = os.path.join(config.workspace, "results", config.exp_name)
checkpoint_dir = os.path.join(config.workspace, "results", config.exp_name, "checkpoint")
if not config.debug:
    create_path(config.workspace)
    create_path(os.path.join(config.workspace, "results"))
    create_path(exp_dir)
    create_path(checkpoint_dir)


def train():
    # load dataset 
    # full_dataset = np.load(
    #     os.path.join("esc-50-data.npy", "esc-50-data.npy"), allow_pickle=True
    # )
    full_dataset = np.load("esc-50-data.npy",  allow_pickle=True)
    dataset = ESC_Dataset(
        dataset = full_dataset,
        config = config,
        eval_mode = False
    )
    eval_dataset = ESC_Dataset(
        dataset = full_dataset,
        config = config,
        eval_mode = True
    )

    audioset_data = DataPrerparation(
        train_dataset = dataset, 
        eval_dataset = eval_dataset, 
        device_num = device_num
    )
    
    # initialize model checkpoint which supports to save the best versions of the model 
    # optimizing storage and training efficiency. 
    checkpoint_callback = ModelCheckpoint(
        monitor= "acc",                         # evaluate metric during training.
        filename="epoch_{epoch:02d}",     # save checkpoint with epoch and accuracy
        save_top_k= 20,                         # keep only the top 20 best checkpoints 
        mode= "max",                            # save checkpoints only when accuracy improves 
        save_last= True 
    )

    # intialize trainer 
    trainer = pl.Trainer(
        deterministic=False,
        default_root_dir=checkpoint_dir,
        devices=device_num,
        accelerator="gpu" if device_num > 0 else "cpu",
        val_check_interval=1.0,
        max_epochs=config.max_epochs,
        sync_batchnorm=True,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        gradient_clip_val=1.0
    )

    # initialize model 
    sed_model = HTSAT_Swin_Transformer(
        spec_size=config.htsat_spec_size,
        patch_size=config.htsat_patch_size,
        in_chans=1,
        num_classes=config.classes_num,
        window_size=config.htsat_window_size,
        config = config,
        depths = config.htsat_depth,
        embed_dim = config.htsat_dim,
        patch_stride=config.htsat_stride,
        num_heads=config.htsat_num_head
    )

    model = SEDWrapper(
        sed_model= sed_model, 
        config = config, 
        dataset= dataset
    )

    if config.resume_checkpoint:
        logging.info(f"Load Checkpoint from {config.resume_checkpoint}")
        ckpt = torch.load(config.resume_checkpoint, map_location="cpu")
        ckpt["state_dict"].pop("sed_model.head.weight")
        ckpt["state_dict"].pop("sed_model.head.bias")
        ckpt["state_dict"].pop("sed_model.tscam_conv.weight")
        ckpt["state_dict"].pop("sed_model.tscam_conv.bias")
        model.load_state_dict(ckpt["state_dict"], strict=False)

    trainer.fit(model, audioset_data)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,  # Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
        datefmt="%Y-%m-%d %H:%M:%S"  # Timestamp format
    )
    mp.set_start_method("spawn", force=True)
    train()
