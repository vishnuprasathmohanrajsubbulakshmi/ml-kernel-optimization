from pathlib import Path
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from python.models.tiny_mlp import TinyMLP


def main() -> None:
    torch.manual_seed(42)

    model = TinyMLP()
    model.eval()

    out_path = PROJECT_ROOT / "models" / "tiny_mlp_state.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), out_path)
    print(f"Saved model state to: {out_path}")


if __name__ == "__main__":
    main()
