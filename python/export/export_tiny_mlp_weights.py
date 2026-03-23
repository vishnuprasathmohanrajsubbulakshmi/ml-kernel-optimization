from pathlib import Path
import sys
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from python.models.tiny_mlp import TinyMLP


def save_array_as_c(name: str, array: np.ndarray, f) -> None:
    flat = array.flatten()
    f.write(f"static const float {name}[{len(flat)}] = {{\n")
    for i, val in enumerate(flat):
        f.write(f"    {val:.9f}f")
        if i != len(flat) - 1:
            f.write(",")
        if (i + 1) % 8 == 0:
            f.write("\n")
        else:
            f.write(" ")
    f.write("\n};\n\n")


def main() -> None:
    model = TinyMLP()

    state_path = PROJECT_ROOT / "models" / "tiny_mlp_state.pt"
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.eval()

    fc1_w = model.fc1.weight.detach().cpu().numpy().T.astype(np.float32)
    fc1_b = model.fc1.bias.detach().cpu().numpy().astype(np.float32)
    fc2_w = model.fc2.weight.detach().cpu().numpy().T.astype(np.float32)
    fc2_b = model.fc2.bias.detach().cpu().numpy().astype(np.float32)

    out_path = PROJECT_ROOT / "src" / "runtime" / "tiny_mlp_weights.h"

    with open(out_path, "w") as f:
        f.write("#ifndef TINY_MLP_WEIGHTS_H\n")
        f.write("#define TINY_MLP_WEIGHTS_H\n\n")

        save_array_as_c("fc1_weights", fc1_w, f)
        save_array_as_c("fc1_bias", fc1_b, f)
        save_array_as_c("fc2_weights", fc2_w, f)
        save_array_as_c("fc2_bias", fc2_b, f)

        f.write("#endif\n")

    print(f"Exported weights to: {out_path}")


if __name__ == "__main__":
    main()
