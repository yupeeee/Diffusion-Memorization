"""
[Memorized]
Detecting, Explaining, and Mitigating Memorization in Diffusion Models
Yuxin Wen, Yuchen Liu, Chen Chen, Lingjuan Lyu
https://github.com/YuxinWenRick/diffusion_memorization/blob/main/examples/sdv1_500_memorized.jsonl

[Normal]
Finding NeMo: Localizing Neurons Responsible For Memorization in Diffusion Models
Dominik Hintersdorf, Lukas Struppek, Kristian Kersting, Adam Dziedzic, Franziska Boenisch
https://github.com/ml-research/localizing_memorization_in_diffusion_models/tree/main/prompts
"""

import json
import os
from pathlib import Path
from typing import List

from torch.utils.data import DataLoader, Dataset

__all__ = [
    "Prompts",
]


def _load_json(
    path: str,
) -> List[str]:
    prompts = []
    with open(path, "r") as f:
        for line in f:
            entry = json.loads(line)
            # prompt = (entry["caption"], entry["url"], int(entry["index"]))
            prompts.append(entry["caption"])
    return prompts


class Prompts(Dataset):
    def __init__(
        self,
        root: Path,
    ) -> None:
        super().__init__()

        self.prompts = []

        # load data/targets
        for fname in [
            "sdv1_500_memorized.jsonl",
            "sdv1_500_normal.jsonl",
        ]:
            fpath = os.path.join(root, fname)
            prompts = _load_json(fpath)
            self.prompts.extend(prompts)

    def __getitem__(
        self,
        index: int,
    ) -> str:
        return self.prompts[index]

    def __len__(
        self,
    ) -> int:
        return len(self.prompts)

    def memorized_prompts(self) -> List[str]:
        return self.prompts[:500]

    def normal_prompts(self) -> List[str]:
        return self.prompts[500:]

    def dataloader(
        self,
        batch_size: int,
        num_workers: int,
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
