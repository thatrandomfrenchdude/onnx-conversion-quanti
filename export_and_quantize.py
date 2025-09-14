"""
Script to export a YOLOv8 PyTorch model to ONNX with a fixed input
resolution and then quantize the resulting model to 8‑bit using the
ONNX Runtime QNN quantization utilities.

Usage:
    python export_and_quantize.py \
        --weights path/to/yolov8.pt \
        --img-size 640 \
        --output-onnx my_model.onnx \
        --quantized-model my_model_qdq.onnx \
        --calibration-data path/to/calibration/images

The script performs the following steps:
  1. Loads a YOLOv8 model (either from a local .pt file or one of
     Ultralytics' built‑in model names) using the Ultralytics API.
  2. Exports the model to ONNX using a fixed input shape to avoid
     dynamic axes, which are not supported by the QNN execution provider.
  3. Preprocesses the ONNX model to insert quantize/dequantize (QDQ)
     nodes required by QNN and then quantizes the model to 8‑bit.

The quantization step uses a `CalibrationDataReader` subclass to feed
calibration samples into ONNX Runtime.  You can provide a directory of
images for calibration or omit the `--calibration-data` option to use
synthetic random data (not recommended for production).  Quantization
requires an x86_64 environment; the resulting quantized model can then
be run on Windows ARM64 devices (or other platforms) with the QNN
execution provider.

Note: This script assumes `ultralytics` and `onnxruntime` are
installed.  Install them via `pip install ultralytics onnxruntime
onnxruntime-qnn`.
"""

import argparse
import pathlib
import sys
from typing import Dict, Iterator, List, Optional

import numpy as np

try:
    import cv2  # Used for reading calibration images
except ImportError:
    cv2 = None  # If OpenCV is unavailable, we can still generate random data

# Ultralytics is used for exporting the YOLO model to ONNX.
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # We'll raise an error at runtime if this isn't available

try:
    import onnxruntime
    from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize
    from onnxruntime.quantization.execution_providers.qnn import (
        get_qnn_qdq_config,
        qnn_preprocess_model,
    )
except ImportError as e:
    raise SystemExit(
        "onnxruntime and onnxruntime-qnn must be installed to run this script."
    ) from e


class ImageFolderDataReader(CalibrationDataReader):
    """Feeds calibration images into ONNX Runtime for quantization.

    This data reader iterates over a directory of images, resizes them
    to the specified input size and normalises pixel values to the range
    [0, 1].  It yields a dictionary mapping input tensor names to
    NumPy arrays.

    If no directory is provided, it will generate random samples to
    demonstrate the API.  Using random data is not recommended for
    production quantization; calibration should reflect real inputs.
    """

    def __init__(
        self,
        model_path: str,
        img_size: int,
        calibration_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._session = onnxruntime.InferenceSession(
            model_path, providers=["CPUExecutionProvider"]
        )
        self._inputs = self._session.get_inputs()
        self._input_name = self._inputs[0].name
        self._input_shape = self._inputs[0].shape
        self._img_size = img_size
        self._calibration_images: List[np.ndarray] = []

        if calibration_dir is not None:
            image_paths = list(pathlib.Path(calibration_dir).glob("*.jpg"))
            image_paths += list(pathlib.Path(calibration_dir).glob("*.png"))
            for p in image_paths:
                img = self._read_image(str(p))
                if img is not None:
                    self._calibration_images.append(img)

        # If no images found, fall back to synthetic data
        if not self._calibration_images:
            # We'll create a few random inputs to satisfy the API
            for _ in range(5):
                dummy = np.random.rand(*self._input_shape).astype(np.float32)
                self._calibration_images.append(dummy)

        self._data_iter: Optional[Iterator[Dict[str, np.ndarray]]] = None

    def _read_image(self, path: str) -> Optional[np.ndarray]:
        if cv2 is None:
            return None
        img = cv2.imread(path)
        if img is None:
            return None
        # Convert to RGB and resize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self._img_size, self._img_size))
        img = img.astype(np.float32) / 255.0  # normalise to [0,1]
        # Reorder to CHW
        img = np.transpose(img, (2, 0, 1))
        # Expand batch dimension
        return img[np.newaxis, ...]

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self._data_iter is None:
            # Prepare an iterator over the calibration data
            self._data_iter = iter(
                {self._input_name: data} for data in self._calibration_images
            )
        return next(self._data_iter, None)

    def rewind(self) -> None:
        # Reset the iterator so quantization can restart
        self._data_iter = None


def export_to_onnx(
    model_weights: str,
    onnx_path: str,
    img_size: int,
    opset: int = 14,
) -> None:
    """Export a YOLOv8 model to ONNX with a fixed input shape.

    Args:
        model_weights: Path to the .pt file or model name (e.g. 'yolov8n').
        onnx_path: Destination file for the exported ONNX model.
        img_size: Input resolution for the model (width and height).
        opset: ONNX opset version to use for export.  Ultralytics
            recommends opset>=14 for YOLOv8.
    """
    if YOLO is None:
        raise RuntimeError(
            "Ultralytics is required to export the model. Install it via 'pip install ultralytics'."
        )

    model = YOLO(model_weights)
    # Create a dummy input for export
    dummy = np.zeros((1, 3, img_size, img_size), dtype=np.float32)
    # Export with dynamic axes disabled to generate a static shape
    model.export(
        format="onnx",
        opset=opset,
        dynamic=False,
        imgsz=(img_size, img_size),
        simplify=True,
        half=False,
        device="cpu",
        verbose=False,
    )
    # The export function writes to a default name (e.g. model.onnx) next to the weights.
    # Find the exported ONNX file and move/rename it.
    exported = pathlib.Path(model_weights).with_suffix("")
    # When exporting using Ultralytics, the ONNX file has the same stem with '.onnx' suffix.
    expected_onnx = exported.with_suffix(".onnx")
    if not expected_onnx.exists():
        # Try default path if provided name is a built‑in model (e.g. yolov8n)
        expected_onnx = pathlib.Path(model.weights_dir) / f"{model.model_name}.onnx"
    expected_onnx = expected_onnx.resolve()
    if not expected_onnx.exists():
        raise FileNotFoundError(f"Could not locate exported ONNX model at {expected_onnx}")
    # Move/rename to the requested location
    pathlib.Path(onnx_path).write_bytes(expected_onnx.read_bytes())


def quantize_model(
    input_model: str,
    output_model: str,
    img_size: int,
    calibration_dir: Optional[str] = None,
) -> None:
    """Quantize an ONNX model using ONNX Runtime QNN utilities.

    Args:
        input_model: Path to the float32 ONNX model.
        output_model: Path to write the quantized model.
        img_size: Input resolution for calibration data reader.
        calibration_dir: Optional directory containing calibration images. If
            not provided, random data will be used (not ideal for production).
    """
    # Preprocess to insert QDQ nodes
    preproc_model_path = pathlib.Path(output_model).with_suffix(".preproc.onnx")
    model_changed = qnn_preprocess_model(input_model, str(preproc_model_path))
    model_to_quantize = str(preproc_model_path) if model_changed else input_model

    # Prepare calibration data reader
    dr = ImageFolderDataReader(
        model_path=model_to_quantize,
        img_size=img_size,
        calibration_dir=calibration_dir,
    )

    # Build a quantization configuration.  Use uint8 for both activations and weights.
    qnn_config = get_qnn_qdq_config(
        model_to_quantize,
        dr,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QUInt8,
    )

    # Perform quantization
    quantize(
        model_input=model_to_quantize,
        model_output=output_model,
        quant_format=qnn_config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to the YOLOv8 .pt file or model name to export",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="Input image size used for export and calibration",
    )
    parser.add_argument(
        "--output-onnx",
        required=True,
        help="Path where the exported ONNX model will be saved",
    )
    parser.add_argument(
        "--quantized-model",
        required=True,
        help="Path where the quantized (QDQ) ONNX model will be saved",
    )
    parser.add_argument(
        "--calibration-data",
        default=None,
        help="Directory containing calibration images for quantization (optional)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Step 1: Export the model to ONNX
    print(f"Exporting {args.weights} to {args.output_onnx}...")
    export_to_onnx(
        model_weights=args.weights,
        onnx_path=args.output_onnx,
        img_size=args.img_size,
    )
    # Step 2: Quantize the model to 8‑bit
    print(f"Quantizing {args.output_onnx} to {args.quantized_model} with 8‑bit precision...")
    quantize_model(
        input_model=args.output_onnx,
        output_model=args.quantized_model,
        img_size=args.img_size,
        calibration_dir=args.calibration_data,
    )
    print("Quantization complete. Saved quantized model to:", args.quantized_model)


if __name__ == "__main__":
    main()