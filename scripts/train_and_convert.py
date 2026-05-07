#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import shutil
import struct

import numpy as np
import tensorflow as tf


# UE7: Parser is extended to configure pruning parameters.
# UE8: Parser is kept minimal: FP16 export, INT8 export and K-Means weight sharing.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one MNIST model and export baseline, pruning and simple quantization variants."
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Directory where .tflite files, metrics and test_digit.bmp are written.",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Baseline training epochs.")
    parser.add_argument(
        "--finetune-epochs",
        type=int,
        default=1,
        help="Short retraining after pruning. Use 0 for pure pruning without retraining.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--conv-filters", type=int, default=16)
    parser.add_argument("--dense-units", type=int, default=64)
    parser.add_argument(
        "--synapse-prune-ratio",
        type=float,
        default=0.70,
        help="Fraction of individual weights set to zero by magnitude pruning.",
    )
    parser.add_argument(
        "--neuron-prune-ratio",
        type=float,
        default=0.50,
        help="Fraction of hidden dense neurons removed.",
    )
    parser.add_argument(
        "--channel-prune-ratio",
        type=float,
        default=0.50,
        help="Fraction of convolution output channels removed.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.20,
        help="Fraction of the MNIST training set reserved for validation metrics.",
    )
    parser.add_argument(
        "--representative-samples",
        type=int,
        default=256,
        help="Number of training images used for INT8 representative calibration.",
    )
    parser.add_argument(
        "--tflite-eval-samples",
        type=int,
        default=0,
        help="Number of test samples used for TFLite accuracy; 0 evaluates all test samples.",
    )
    parser.add_argument(
        "--test-digit-index",
        type=int,
        default=0,
        help="MNIST test image written to test_digit.bmp for Pi benchmark mode.",
    )
    return parser.parse_args()


# UE7: Loading now training and test data for comparison after pruning.
def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., None]
    x_test = x_test[..., None]
    return x_train, y_train, x_test, y_test


# UE8: Split once so all final models use the same train and validation data.
def split_train_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    validation_split: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split = float(np.clip(validation_split, 0.0, 0.50))
    validation_count = int(round(len(x_train) * split))
    validation_count = max(1, validation_count) if split > 0.0 else 0
    if validation_count == 0:
        return x_train, y_train, x_train[:0], y_train[:0]
    return (
        x_train[:-validation_count],
        y_train[:-validation_count],
        x_train[-validation_count:],
        y_train[-validation_count:],
    )


# UE8: Keep compile settings in one place.
def compile_for_classification(model: tf.keras.Model) -> None:
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


# UE7: Added another Dense Layer for subsequent pruning.
# We add names for the layers to access them in pruning functions.
def make_model(conv_filters: int = 16, dense_units: int = 64) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1), name="image"),
            tf.keras.layers.Conv2D(conv_filters, 3, activation="relu", name="conv"),
            tf.keras.layers.MaxPooling2D(name="pool"),
            tf.keras.layers.Flatten(name="flatten"),
            tf.keras.layers.Dense(dense_units, activation="relu", name="hidden"),
            tf.keras.layers.Dense(10, activation="softmax", name="output"),
        ]
    )
    compile_for_classification(model)
    return model


# UE8: Clone the architecture and install a provided weight list.
def clone_with_weights(base_model: tf.keras.Model, new_weights: list[np.ndarray]) -> tf.keras.Model:
    model = tf.keras.models.clone_model(base_model)
    model.set_weights(new_weights)
    compile_for_classification(model)
    return model


# UE7: Create function for unstructured pruning by evaluating individual small-magnitude weights.
def prune_synapses(base_model: tf.keras.Model, prune_ratio: float) -> tf.keras.Model:
    model = tf.keras.models.clone_model(base_model)
    model.set_weights(base_model.get_weights())
    compile_for_classification(model)

    kernels = []
    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights) >= 1:
            kernels.append(np.abs(weights[0]).reshape(-1))

    all_kernel_values = np.concatenate(kernels)
    threshold = np.percentile(all_kernel_values, prune_ratio * 100.0)

    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights) >= 1:
            kernel = weights[0]
            kernel[np.abs(kernel) <= threshold] = 0.0
            weights[0] = kernel
            layer.set_weights(weights)

    return model


# UE7: Helper function to define how many neurons or channels to keep.
def keep_count(total: int, prune_ratio: float) -> int:
    ratio = float(np.clip(prune_ratio, 0.0, 0.95))
    return max(1, min(total, int(round(total * (1.0 - ratio)))))


# UE7: Create function to prune neurons, so hidden Dense units have fewer neurons.
def prune_neurons(base_model: tf.keras.Model, prune_ratio: float) -> tf.keras.Model:
    conv = base_model.get_layer("conv")
    hidden = base_model.get_layer("hidden")
    output = base_model.get_layer("output")

    hidden_kernel, hidden_bias = hidden.get_weights()
    output_kernel, output_bias = output.get_weights()

    old_units = hidden_kernel.shape[1]
    new_units = keep_count(old_units, prune_ratio)

    scores = np.sum(np.abs(hidden_kernel), axis=0) + np.sum(np.abs(output_kernel), axis=1)
    keep = np.sort(np.argsort(scores)[-new_units:])

    model = make_model(conv_filters=conv.filters, dense_units=new_units)
    model.get_layer("conv").set_weights(conv.get_weights())
    model.get_layer("hidden").set_weights([hidden_kernel[:, keep], hidden_bias[keep]])
    model.get_layer("output").set_weights([output_kernel[keep, :], output_bias])
    return model


# UE7: Helper function for channel pruning.
def rows_for_channels(channels: np.ndarray, old_channels: int, pooled_side: int = 13) -> np.ndarray:
    rows: list[int] = []
    for y in range(pooled_side):
        for x in range(pooled_side):
            for channel in channels:
                rows.append((y * pooled_side + x) * old_channels + int(channel))
    return np.array(rows, dtype=np.int64)


# UE7: Function to prune individual channels and adapt the following dense layer.
def prune_channels(base_model: tf.keras.Model, prune_ratio: float) -> tf.keras.Model:
    conv = base_model.get_layer("conv")
    hidden = base_model.get_layer("hidden")
    output = base_model.get_layer("output")

    conv_kernel, conv_bias = conv.get_weights()
    hidden_kernel, hidden_bias = hidden.get_weights()

    old_channels = conv_kernel.shape[3]
    new_channels = keep_count(old_channels, prune_ratio)

    scores = np.sum(np.abs(conv_kernel), axis=(0, 1, 2))
    keep = np.sort(np.argsort(scores)[-new_channels:])

    model = make_model(conv_filters=new_channels, dense_units=hidden.units)
    model.get_layer("conv").set_weights([conv_kernel[:, :, :, keep], conv_bias[keep]])

    selected_rows = rows_for_channels(keep, old_channels)
    model.get_layer("hidden").set_weights([hidden_kernel[selected_rows, :], hidden_bias])
    model.get_layer("output").set_weights(output.get_weights())
    return model


# UE7: Implement finetune after pruning.
def finetune(model: tf.keras.Model, x: np.ndarray, y: np.ndarray, epochs: int, batch_size: int) -> None:
    if epochs <= 0:
        return
    model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True)


# UE8: Return the mean absolute error between original and reconstructed weights.
def mean_absolute_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    if original.size == 0:
        return 0.0
    return float(np.mean(np.abs(original.astype(np.float32) - reconstructed.astype(np.float32))))


# UE8: Show the simple INT8 formula on FP32 weights for error and compression reporting.
def linear_int8_metrics(model: tf.keras.Model) -> dict[str, float]:
    total_error = 0.0
    total_values = 0
    original_bits = 0
    compressed_bits = 0

    for values in model.get_weights():
        if not np.issubdtype(values.dtype, np.floating):
            continue

        values32 = values.astype(np.float32, copy=False)
        flat_size = int(values32.size)
        if flat_size == 0:
            continue

        qmin = -128
        qmax = 127
        rmin = float(np.min(values32))
        rmax = float(np.max(values32))

        if rmax == rmin:
            reconstructed = values32.copy()
        else:
            scale = (rmax - rmin) / float(qmax - qmin)
            zero_point = int(round(qmin - rmin / scale))
            zero_point = int(np.clip(zero_point, qmin, qmax))
            quantized = np.clip(np.round(values32 / scale + zero_point), qmin, qmax)
            reconstructed = ((quantized.astype(np.float32) - zero_point) * scale).astype(np.float32)

        total_error += mean_absolute_error(values32, reconstructed) * flat_size
        total_values += flat_size
        original_bits += flat_size * 32
        compressed_bits += flat_size * 8 + 64  # scale and zero_point per tensor

    return {
        "mae": total_error / float(max(1, total_values)),
        "compression_ratio": original_bits / float(max(1, compressed_bits)),
    }


# UE8: Representative data generator calibrates activation ranges for full INT8 TFLite conversion.
def make_representative_dataset(x: np.ndarray, sample_count: int):
    limit = max(1, min(int(sample_count), len(x)))

    def representative_dataset():
        for index in range(limit):
            yield [x[index : index + 1].astype(np.float32)]

    return representative_dataset


# UE8: Encapsulated export with optional FP16 or INT8 TFLite converter settings.
def export_tflite(
    model: tf.keras.Model,
    artifacts_dir: Path,
    name: str,
    tflite_mode: str = "fp32",
    representative_data: np.ndarray | None = None,
    representative_samples: int = 256,
) -> Path:
    keras_dir = artifacts_dir / "keras"
    saved_models_dir = artifacts_dir / "saved_models"
    keras_dir.mkdir(parents=True, exist_ok=True)
    saved_models_dir.mkdir(parents=True, exist_ok=True)

    keras_path = keras_dir / f"{name}.keras"
    saved_model_dir = saved_models_dir / name
    tflite_path = artifacts_dir / f"{name}.tflite"

    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)

    model.save(str(keras_path))
    model.export(str(saved_model_dir))

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

    # TODO insert model conversion with quanty
    if tflite_mode == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_data is None:
            raise ValueError("INT8 Conversion needs representative_data")
        converter.representative_dataset = make_representative_dataset(representative_data, representative_samples) # not sure if samples is correct bc he types fast
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_path.write_bytes(converter.convert())
    return tflite_path


# UE8: Quantize a float input tensor when the TFLite model expects INT8 input.
def quantize_tensor_for_tflite(values: np.ndarray, detail: dict[str, object]) -> np.ndarray:
    # TODO process inputs
    dtype = detail["dtype"]
    if np.issubdtype(dtype, np.floating):
        return values.astype(dtype)
    # if not floating type
    scale, zero_point = detail["quantization"]
    

    info = np.iinfo(dtype) # iinfo -> more metainfo 
    quantized = np.round(values / scale + zero_point)

    return np.clip(quantized, info.min, info.max).astype(dtype) # that's why we need iinfo

# UE8: Dequantize TFLite INT8 output back to float probabilities/logits.
def dequantize_tensor_from_tflite(values: np.ndarray, detail: dict[str, object]) -> np.ndarray:
    # TODO process outputs
    if not np.issubtype(values.dtype, np.integer):
        return values.astype(np.float32)
    # same as for_tflite
    scale, zero_point = detail["quantization"]
    scale = float(scale) if float(scale) != 0.0 else 1.0
    zero_point = int(zero_point)

    return (values.astype(np.float32) - float(zero_point)) * scale


# UE8: Evaluate exported TFLite models so the CSV includes deployable model accuracy.
def evaluate_tflite(tflite_path: Path, x: np.ndarray, y: np.ndarray, sample_limit: int = 0) -> tuple[float, float]:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    limit = len(x) if int(sample_limit) <= 0 else min(int(sample_limit), len(x))
    correct = 0
    losses: list[float] = []

    for index in range(limit):
        sample = x[index : index + 1].astype(np.float32)
        interpreter.set_tensor(input_detail["index"], quantize_tensor_for_tflite(sample, input_detail))
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])
        probabilities = dequantize_tensor_from_tflite(output, output_detail)[0]
        predicted = int(np.argmax(probabilities))
        label = int(y[index])
        correct += int(predicted == label)
        label_probability = float(np.clip(probabilities[label], 1e-7, 1.0))
        losses.append(float(-np.log(label_probability)))

    return float(np.mean(losses)), float(correct / max(1, limit))


# UE7: Helper function to evaluate sparsity.
def count_nonzero_weights(model: tf.keras.Model) -> tuple[int, int]:
    total = 0
    nonzero = 0
    for weight in model.get_weights():
        total += int(weight.size)
        nonzero += int(np.count_nonzero(weight))
    return total, nonzero


# UE7: Writes a simple uncompressed 24-bit BMP.
def save_test_digit_bmp(path: Path, image_28x28: np.ndarray, scale: int = 10) -> None:
    image = np.repeat(np.repeat(image_28x28, scale, axis=0), scale, axis=1)
    pixels = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    height, width = pixels.shape
    row_stride = (width * 3 + 3) & ~3
    pixel_data_size = row_stride * height
    file_size = 14 + 40 + pixel_data_size

    with path.open("wb") as output:
        output.write(b"BM")
        output.write(struct.pack("<IHHI", file_size, 0, 0, 14 + 40))
        output.write(struct.pack("<IiiHHIIiiII", 40, width, height, 1, 24, 0, pixel_data_size, 0, 0, 0, 0))
        padding = b"\x00" * (row_stride - width * 3)
        for y in range(height - 1, -1, -1):
            row = bytearray()
            for value in pixels[y]:
                row.extend([int(value), int(value), int(value)])  # BMP order: B, G, R
            output.write(row)
            output.write(padding)


# UE8: Evaluate validation and test metrics for every variant.
def evaluate_keras_model(
    model: tf.keras.Model,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    if len(x_validation):
        validation_loss, validation_accuracy = model.evaluate(x_validation, y_validation, verbose=0)
    else:
        validation_loss, validation_accuracy = float("nan"), float("nan")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    return {
        "validation_loss": float(validation_loss),
        "validation_accuracy": float(validation_accuracy),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
    }


# UE8: Add one final comparison row for pruning, quantization, TFLite size and estimated compression.
def add_metrics_row(
    rows: list[dict[str, object]],
    name: str,
    model: tf.keras.Model,
    keras_metrics: dict[str, float],
    tflite_metrics: tuple[float, float],
    tflite_path: Path,
    method: str,
    tflite_mode: str,
    quantization_mae: float | None,
    estimated_compression_ratio: float | None,
    baseline_tflite_bytes: int,
) -> None:
    total, nonzero = count_nonzero_weights(model)
    tflite_bytes = tflite_path.stat().st_size
    tflite_ratio = float(baseline_tflite_bytes / tflite_bytes) if tflite_bytes else 1.0
    rows.append(
        {
            "model": name,
            "method": method,
            "tflite_mode": tflite_mode,
            "tflite_file": tflite_path.name,
            "validation_accuracy": f"{keras_metrics['validation_accuracy']:.6f}",
            "test_accuracy": f"{keras_metrics['test_accuracy']:.6f}",
            "tflite_test_accuracy": f"{tflite_metrics[1]:.6f}",
            "tflite_file_size_bytes": tflite_bytes,
            "tflite_size_ratio_vs_baseline": f"{tflite_ratio:.6f}",
            "parameters_total": total,
            "parameters_nonzero": nonzero,
            "parameter_sparsity": f"{1.0 - (nonzero / total):.6f}",
            "quantization_mae": "" if quantization_mae is None else f"{quantization_mae:.9g}",
            "estimated_compression_ratio": "" if estimated_compression_ratio is None else f"{estimated_compression_ratio:.6f}",
        }
    )


# UE7: Main now trains baseline, creates pruning variants and exports every model.
# UE8: Main also creates minimal FP16, INT8 and K-Means quantization variants.
def main() -> int:
    args = parse_args()
    tf.keras.utils.set_random_seed(42)
    np.random.seed(42)

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    x_train_full, y_train_full, x_test, y_test = load_data()
    x_train, y_train, x_validation, y_validation = split_train_validation(
        x_train_full,
        y_train_full,
        args.validation_split,
    )

    # UE7: Store additional label and test image for perf.
    test_index = args.test_digit_index % len(x_test)
    save_test_digit_bmp(artifacts_dir / "test_digit.bmp", x_test[test_index, :, :, 0])
    (artifacts_dir / "test_digit_label.txt").write_text(f"{int(y_test[test_index])}\n", encoding="utf-8")

    rows: list[dict[str, object]] = []

    baseline = make_model(args.conv_filters, args.dense_units)
    baseline.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation) if len(x_validation) else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        shuffle=True,
    )

    models: list[dict[str, object]] = [
        {
            "name": "baseline",
            "method": "normal FP32 training",
            "model": baseline,
            "tflite_mode": "fp32",
            "quantization_mae": None,
            "estimated_compression_ratio": None,
        }
    ]

    synapse = prune_synapses(baseline, args.synapse_prune_ratio)
    finetune(synapse, x_train, y_train, args.finetune_epochs, args.batch_size)
    synapse = prune_synapses(synapse, args.synapse_prune_ratio)
    models.append(
        {
            "name": "synapse_pruned",
            "method": "unstructured small-weight pruning",
            "model": synapse,
            "tflite_mode": "fp32",
            "quantization_mae": None,
            "estimated_compression_ratio": None,
        }
    )

    neuron = prune_neurons(baseline, args.neuron_prune_ratio)
    finetune(neuron, x_train, y_train, args.finetune_epochs, args.batch_size)
    models.append(
        {
            "name": "neuron_pruned",
            "method": "structured hidden-neuron pruning",
            "model": neuron,
            "tflite_mode": "fp32",
            "quantization_mae": None,
            "estimated_compression_ratio": None,
        }
    )

    channel = prune_channels(baseline, args.channel_prune_ratio)
    finetune(channel, x_train, y_train, args.finetune_epochs, args.batch_size)
    models.append(
        {
            "name": "channel_pruned",
            "method": "structured convolution-channel pruning",
            "model": channel,
            "tflite_mode": "fp32",
            "quantization_mae": None,
            "estimated_compression_ratio": None,
        }
    )
    
    int8_metrics = linear_int8_metrics(baseline)
    models.append(
        {
            "name": "baseline_int8",
            "method": "linear INT8 TFLite export",
            "model": baseline,
            "tflite_mode": "int8",
            "quantization_mae": int8_metrics["mae"],
            "estimated_compression_ratio": int8_metrics["compression_ratio"],
        }
    )

    baseline_tflite_bytes = 1
    for model_info in models:
        name = str(model_info["name"])
        model = model_info["model"]
        tflite_mode = str(model_info["tflite_mode"])
        tflite_path = export_tflite(
            model,
            artifacts_dir,
            name,
            tflite_mode=tflite_mode,
            representative_data=x_train,
            representative_samples=args.representative_samples,
        )
        if name == "baseline":
            baseline_tflite_bytes = tflite_path.stat().st_size

        keras_metrics = evaluate_keras_model(model, x_validation, y_validation, x_test, y_test)
        tflite_metrics = evaluate_tflite(tflite_path, x_test, y_test, args.tflite_eval_samples)
        add_metrics_row(
            rows,
            name,
            model,
            keras_metrics,
            tflite_metrics,
            tflite_path,
            str(model_info["method"]),
            tflite_mode,
            model_info["quantization_mae"],
            model_info["estimated_compression_ratio"],
            baseline_tflite_bytes,
        )
        print(
            f"Exported {tflite_path} "
            f"keras_acc={keras_metrics['test_accuracy']:.4f} "
            f"tflite_acc={tflite_metrics[1]:.4f}"
        )

    shutil.copy2(artifacts_dir / "baseline.tflite", artifacts_dir / "model.tflite")

    metrics_path = artifacts_dir / "model_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved benchmark BMP: {artifacts_dir / 'test_digit.bmp'}")
    print(f"Saved test digit label: {int(y_test[test_index])}")
    print(f"Saved model metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
