import argparse

__all__ = [
    "load",
]


def load() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--num-inference-steps", type=int, required=False, default=50)
    parser.add_argument("--guidance-scale", type=float, required=False, default=7.5)
    parser.add_argument("--seed", type=int, required=False, default=0)
    parser.add_argument("--num-seeds", type=int, required=False, default=50)
    parser.add_argument("--batch-size", type=int, required=False, default=16)
    parser.add_argument("--num-workers", type=int, required=False, default=4)
    parser.add_argument("--device", type=str, required=False, default="cuda")
    parser.add_argument("--devices", type=str, required=False, default="auto")
    parser.add_argument("--plot", action="store_true", default=False)

    args = parser.parse_args()

    return args
