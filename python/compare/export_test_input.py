from pathlib import Path
import numpy as np


def main() -> None:
    x = np.array([(i % 10) / 10.0 for i in range(28 * 28)], dtype=np.float32)
    out_path = Path("python/compare/test_input.npy")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, x.reshape(1, 1, 28, 28))
    print(f"Saved test input to: {out_path}")


if __name__ == "__main__":
    main()
