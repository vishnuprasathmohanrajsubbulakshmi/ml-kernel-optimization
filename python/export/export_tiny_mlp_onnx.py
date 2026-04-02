from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from python.models.tiny_mlp import TinyMLP


def main() -> None:
    model = TinyMLP()
    state_path = PROJECT_ROOT / "models" / "tiny_mlp_state.pt"
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 1, 28, 28)

    output_path = PROJECT_ROOT / "models" / "onnx" / "tiny_mlp.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
    )

    print(f"Exported ONNX model to: {output_path}")


if __name__ == "__main__":
    main()
