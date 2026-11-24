"""Export CNN15 to TorchScript, ONNX, and quantized TorchScript, then compare predictions."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from exam_cnn.data.preprocessing import character_normalize_batch, normalize_pixels
from exam_cnn.models.cnn15 import CNN15, IMAGE_SIZE, IN_CHANNELS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / "exam_cnn/models_export/cnn15_best.pt"
TORCHSCRIPT_PATH = PROJECT_ROOT / "exam_cnn/models_export/cnn15.torchscript.pt"
ONNX_PATH = PROJECT_ROOT / "exam_cnn/models_export/cnn15.onnx"
QUANTIZED_PATH = PROJECT_ROOT / "exam_cnn/models_export/cnn15.quantized.torchscript.pt"
DATA_PATH = PROJECT_ROOT / "exam_cnn/datasets/processed/az_dataset.npz"
N_VERIFY_IMAGES = 1000
ONNX_OPSET = 13


def export_torchscript(model: torch.nn.Module) -> torch.jit.ScriptModule:
    model.eval()
    example = torch.randn(1, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    scripted = torch.jit.trace(model, example)
    scripted.save(str(TORCHSCRIPT_PATH))
    return scripted


def export_onnx(model: torch.nn.Module):
    model.eval()
    example = torch.randn(1, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(
        model,
        example,
        str(ONNX_PATH),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=ONNX_OPSET,
        dynamo=False,  ## Use the exporter supported by this Windows environment.
    )


def export_quantized(model: torch.nn.Module):
    model.eval()
    quantized = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    example = torch.randn(1, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    scripted = torch.jit.trace(quantized, example)
    scripted.save(str(QUANTIZED_PATH))


def main():
    model = CNN15()
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")

    print("Exporting TorchScript ...")
    export_torchscript(model)
    print("Exporting ONNX ...")
    export_onnx(model)
    print("Exporting dynamically-quantized TorchScript ...")
    export_quantized(model)

    print(f"Verifying agreement across formats on {N_VERIFY_IMAGES} test images ...")
    d = np.load(DATA_PATH)
    X = d["X_test"][:N_VERIFY_IMAGES]
    Xn = normalize_pixels(character_normalize_batch(X))[:, None, :, :].astype(
        np.float32
    )
    x_t = torch.from_numpy(Xn)

    with torch.no_grad():
        pred_original = model(x_t).argmax(1).numpy()

        scripted = torch.jit.load(str(TORCHSCRIPT_PATH))
        pred_scripted = scripted(x_t).argmax(1).numpy()

        quantized = torch.jit.load(str(QUANTIZED_PATH))
        pred_quantized = quantized(x_t).argmax(1).numpy()

    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"input": Xn})[0]
    pred_onnx = onnx_out.argmax(1)

    agree_scripted = (pred_original == pred_scripted).mean()
    agree_onnx = (pred_original == pred_onnx).mean()
    agree_quantized = (pred_original == pred_quantized).mean()

    print(f"  original vs TorchScript agreement: {agree_scripted:.4f}")
    print(f"  original vs ONNX agreement:        {agree_onnx:.4f}")
    print(f"  original vs quantized agreement:   {agree_quantized:.4f}")

    assert agree_scripted == 1.0, "TorchScript export diverges from the original model"
    assert agree_onnx >= 0.99, "ONNX export diverges too much from the original model"

    for p in [TORCHSCRIPT_PATH, ONNX_PATH, QUANTIZED_PATH]:
        print(f"  {p.name}: {p.stat().st_size / 1e6:.2f} MB")

    print("All export formats verified.")


if __name__ == "__main__":
    main()
