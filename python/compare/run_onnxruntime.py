import numpy as np
import onnxruntime as ort


def main() -> None:
    model_path = "models/onnx/tiny_mlp.onnx"
    input_path = "python/compare/test_input.npy"

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    x = np.load(input_path).astype(np.float32)
    y = session.run([output_name], {input_name: x})[0]

    print("ONNX Runtime output:")
    for i, val in enumerate(y[0]):
        print(f"output[{i}] = {val:.6f}")


if __name__ == "__main__":
    main()
