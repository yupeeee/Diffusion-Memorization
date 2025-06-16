import os

import utils
from paths import DATASET_DIR, LOG_DIR

if __name__ == "__main__":
    args = utils.args.load()

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
        save_dir=os.path.join(
            LOG_DIR,
            "sdv1-memorization",
        ),
    )
