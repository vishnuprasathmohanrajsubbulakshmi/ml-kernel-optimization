from pathlib import Path
import onnx


def main() -> None:
    model_path = Path("models/onnx/tiny_mlp.onnx")
    model = onnx.load(str(model_path))

    onnx.checker.check_model(model)
    print(f"Model checked successfully: {model_path}")
    print()

    print("Inputs:")
    for inp in model.graph.input:
        print(f"  - {inp.name}")

    print("\nOutputs:")
    for out in model.graph.output:
        print(f"  - {out.name}")

    print("\nNodes:")
    for i, node in enumerate(model.graph.node):
        print(f"  [{i}] op_type={node.op_type}, name={node.name or '<unnamed>'}")
        print(f"      inputs={list(node.input)}")
        print(f"      outputs={list(node.output)}")
        if node.attribute:
            print("      attributes:")
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.INT:
                    print(f"        - {attr.name} = {attr.i}")
                elif attr.type == onnx.AttributeProto.FLOAT:
                    print(f"        - {attr.name} = {attr.f}")
                elif attr.type == onnx.AttributeProto.STRING:
                    print(f"        - {attr.name} = {attr.s.decode()}")
                else:
                    print(f"        - {attr.name} = <unsupported attr type>")


if __name__ == "__main__":
    main()
