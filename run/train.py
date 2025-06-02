import argparse
import os

import torch
import utils
from paths import DATASET_DIR, LOG_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-inference-steps", type=int, required=False, default=50)
    parser.add_argument("--guidance-scale", type=float, required=False, default=7.5)
    parser.add_argument("--batch-size", type=int, required=False, default=16)
    parser.add_argument("--num-workers", type=int, required=False, default=4)
    parser.add_argument("--num-epochs", type=int, required=False, default=100)
    parser.add_argument("--lr", type=float, required=False, default=1e-4)
    parser.add_argument("--lr-warmup-steps", type=int, required=False, default=20)
    parser.add_argument("--devices", type=str, required=False, default="auto")
    args = parser.parse_args()

    # Load dataset
    train_dataset = utils.datasets.cifar10.load_dataset(
        root=os.path.join(DATASET_DIR, "cifar10"),
        train=True,
    )
    val_dataset = utils.datasets.cifar10.load_dataset(
        root=os.path.join(DATASET_DIR, "cifar10"),
        train=False,
    )
    # Create subset of dataset using indices
    train_indices = list(range(1))
    val_indices = list(range(1))

    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    # Load pipeline
    pipe = utils.pipe.StableDiffusionCIFAR10(
        version="1.4",
        scheduler="DDIM",
        variant="fp16",
        verbose=True,
        weights_path=None,
    )

    # Train
    trainer = utils.train.Trainer(
        log_root=LOG_DIR,
        name="cifar10",
        version=None,
        **vars(args),
    )
    trainer.run(train_dataset, val_dataset, pipe)
