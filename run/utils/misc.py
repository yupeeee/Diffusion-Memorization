import torch.nn as nn

__all__ = [
    "get_submodule",
]


def get_submodule(
    module: nn.Module,
    attr: str,
) -> nn.Module:
    try:
        for a in attr.split("."):
            if a.isnumeric():
                module = module[int(a)]
            else:
                module = getattr(module, a)
        return module
    except AttributeError:
        raise AttributeError(f"Could not find attribute '{attr}' in module")
