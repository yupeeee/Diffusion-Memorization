import argparse
import os

import utils
from paths import DATASET_DIR, LOG_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-inference-steps", type=int, required=False, default=50)
    parser.add_argument("--guidance-scale", type=float, required=False, default=7.5)
    parser.add_argument("--seed", type=int, required=False, default=0)
    parser.add_argument("--batch-size", type=int, required=False, default=16)
    parser.add_argument("--num-workers", type=int, required=False, default=4)
    parser.add_argument("--devices", type=str, required=False, default="auto")
    args = parser.parse_args()

    # Load dataset
    prompts = utils.datasets.sdv1_memorization.Prompts(
        root=os.path.join(DATASET_DIR, "sdv1-memorization"),
    )

    # Load pipeline
    pipe = utils.pipe.StableDiffusion(
        version="1.4",
        scheduler="DDIM",
        variant="fp16",
        verbose=False,
    )

    # Generate
    runner = utils.generate.Runner(**vars(args))
    runner.run(
        prompts,
        pipe,
        save_dir=os.path.join(LOG_DIR, "sdv1-memorization"),
    )
