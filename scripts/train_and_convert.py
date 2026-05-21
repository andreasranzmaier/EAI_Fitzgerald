#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Iterator

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from resampler import load_gesture_sequences

# Model input constants — must match tflite_gesture_classifier.h
kTimesteps = 50
kFeatureCount = 9
kNumClasses = 4


def require_tfmot():
    try:
        import tf_keras  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow 2.16 uses Keras 3 by default, but TFMOT 0.8.0 requires "
            "legacy Keras 2. Install tf-keras as well: "
            "pip install 'tensorflow==2.16.1' 'tf-keras==2.16.*' "
            "'tensorflow-model-optimization==0.8.0' 'numpy==1.26.4'"
        ) from exc

    try:
        from tensorflow_model_optimization.clustering import keras as clustering
    except ImportError as exc:
        raise RuntimeError(
            "Could not import TFMOT clustering. Check that "
            "tensorflow-model-optimization==0.8.0 is installed together with "
            "tensorflow==2.16.1, tf-keras==2.16.*, and numpy==1.26.4. "
            f"Original import error: {exc}"
        ) from exc
    return clustering


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a gesture CNN and export baseline, pruning and quantization variants."
    )
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--cluster-finetune-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--conv-filters", type=int, default=16)
    parser.add_argument("--dense-units", type=int, default=32)
    parser.add_argument("--synapse-prune-ratio", type=float, default=0.70)
    parser.add_argument("--neuron-prune-ratio", type=float, default=0.50)
    parser.add_argument("--channel-prune-ratio", type=float, default=0.50)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--kmeans-k", type=int, default=16)
    parser.add_argument("--representative-samples", type=int, default=256)
    parser.add_argument("--tflite-eval-samples", type=int, default=0)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    X, y = load_gesture_sequences(data_dir, num_points=kTimesteps)
    return X.astype(np.float32), y.astype(np.int32)


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    validation_split: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=validation_split * 2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def compile_for_classification(model: tf.keras.Model) -> None:
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def make_adapted_norm(norm_weights: list[np.ndarray]) -> tf.keras.layers.Layer:
    """Reconstruct a Normalization layer from stored adapt weights."""
    layer = tf.keras.layers.Normalization(axis=-1)
    layer(tf.zeros((1, kTimesteps, kFeatureCount), dtype=tf.float32))
    layer.set_weights(norm_weights)
    layer.trainable = False
    return layer


def make_model(
    norm_layer: tf.keras.layers.Layer,
    conv_filters: int = 16,
    dense_units: int = 32,
) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(kTimesteps, kFeatureCount)),
        norm_layer,
        tf.keras.layers.Conv1D(conv_filters, 3, activation="relu", name="conv"),
        tf.keras.layers.GlobalAveragePooling1D(name="gap"),
        tf.keras.layers.Dense(dense_units, activation="relu", name="dense1"),
        tf.keras.layers.Dropout(0.3, name="dropout"),
        tf.keras.layers.Dense(kNumClasses, activation="softmax", name="output"),
    ])
    compile_for_classification(model)
    return model


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def keep_count(total: int, prune_ratio: float) -> int:
    ratio = float(np.clip(prune_ratio, 0.0, 0.95))
    return max(1, min(total, int(round(total * (1.0 - ratio)))))


def prune_synapses(base_model: tf.keras.Model, prune_ratio: float) -> tf.keras.Model:
    # clone_model preserves Normalization layer stats via get_config().
    model = tf.keras.models.clone_model(base_model)
    model.set_weights(base_model.get_weights())
    compile_for_classification(model)

    # Only prune layers with trainable kernels; skip Normalization/Dropout/GAP.
    kernels = [
        np.abs(layer.kernel.numpy()).reshape(-1)
        for layer in model.layers
        if getattr(layer, "kernel", None) is not None
    ]
    if not kernels:
        return model
    threshold = np.percentile(np.concatenate(kernels), prune_ratio * 100.0)

    for layer in model.layers:
        if getattr(layer, "kernel", None) is not None:
            weights = layer.get_weights()
            weights[0][np.abs(weights[0]) <= threshold] = 0.0
            layer.set_weights(weights)
    return model


def prune_neurons(
    base_model: tf.keras.Model,
    prune_ratio: float,
    norm_weights: list[np.ndarray],
) -> tf.keras.Model:
    """Remove the least-important neurons from dense1."""
    conv = base_model.get_layer("conv")
    dense1 = base_model.get_layer("dense1")
    output = base_model.get_layer("output")

    d1_kernel, d1_bias = dense1.get_weights()
    out_kernel, out_bias = output.get_weights()

    scores = np.sum(np.abs(d1_kernel), axis=0) + np.sum(np.abs(out_kernel), axis=1)
    keep = np.sort(np.argsort(scores)[-keep_count(d1_kernel.shape[1], prune_ratio):])

    d1_kernel = d1_kernel[:, keep]
    d1_bias = d1_bias[keep]
    out_kernel = out_kernel[keep, :]

    model = make_model(make_adapted_norm(norm_weights), conv.filters, len(keep))
    model(tf.zeros((1, kTimesteps, kFeatureCount)))
    model.get_layer("conv").set_weights(conv.get_weights())
    model.get_layer("dense1").set_weights([d1_kernel, d1_bias])
    model.get_layer("output").set_weights([out_kernel, out_bias])
    return model


def prune_channels(
    base_model: tf.keras.Model,
    prune_ratio: float,
    norm_weights: list[np.ndarray],
) -> tf.keras.Model:
    """Remove the least-important Conv1D filters."""
    conv = base_model.get_layer("conv")
    dense1 = base_model.get_layer("dense1")
    output = base_model.get_layer("output")

    conv_kernel, conv_bias = conv.get_weights()  # (kernel_size, in_features, filters)
    d1_kernel, d1_bias = dense1.get_weights()    # (filters, dense_units)

    # Score each filter by combined kernel + downstream dense weight magnitude.
    conv_scores = np.sum(np.abs(conv_kernel), axis=(0, 1))  # (filters,)
    d1_scores = np.sum(np.abs(d1_kernel), axis=1)            # (filters,)
    scores = conv_scores + d1_scores

    keep = np.sort(np.argsort(scores)[-keep_count(conv_kernel.shape[2], prune_ratio):])

    conv_kernel = conv_kernel[:, :, keep]
    conv_bias = conv_bias[keep]
    d1_kernel = d1_kernel[keep, :]

    model = make_model(make_adapted_norm(norm_weights), len(keep), dense1.units)
    model(tf.zeros((1, kTimesteps, kFeatureCount)))
    model.get_layer("conv").set_weights([conv_kernel, conv_bias])
    model.get_layer("dense1").set_weights([d1_kernel, d1_bias])
    model.get_layer("output").set_weights(output.get_weights())
    return model


def finetune(model: tf.keras.Model, x: np.ndarray, y: np.ndarray, epochs: int, batch_size: int) -> None:
    if epochs > 0:
        model.fit(x, y, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True)


# ---------------------------------------------------------------------------
# Weight analysis helpers (unchanged from original)
# ---------------------------------------------------------------------------

def mean_absolute_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    if original.size == 0:
        return 0.0
    return float(np.mean(np.abs(original.astype(np.float32) - reconstructed.astype(np.float32))))


def variable_to_numpy(variable: Any) -> np.ndarray:
    if hasattr(variable, "numpy"):
        return np.asarray(variable.numpy())
    return np.asarray(variable)


def variable_key(variable: Any) -> object:
    ref = getattr(variable, "ref", None)
    if callable(ref):
        return ref()
    experimental_ref = getattr(variable, "experimental_ref", None)
    if callable(experimental_ref):
        return experimental_ref()
    handle = getattr(variable, "handle", None)
    return id(handle) if handle is not None else id(variable)


def iter_layer_weight_arrays(model: tf.keras.Model) -> Iterator[np.ndarray]:
    seen: set[object] = set()
    for layer in model.layers:
        for variable in getattr(layer, "weights", []):
            key = variable_key(variable)
            if key in seen:
                continue
            seen.add(key)
            values = variable_to_numpy(variable)
            if values.size:
                yield values


def iter_named_layer_weight_arrays(model: tf.keras.Model) -> Iterator[tuple[str, list[np.ndarray]]]:
    for layer in model.layers:
        arrays = [variable_to_numpy(variable) for variable in getattr(layer, "weights", [])]
        arrays = [values for values in arrays if values.size]
        if arrays:
            yield layer.name, arrays


def model_weight_mae(original_model: tf.keras.Model, reconstructed_model: tf.keras.Model) -> float:
    total_error = 0.0
    total_values = 0

    for layer_name, original_arrays in iter_named_layer_weight_arrays(original_model):
        try:
            reconstructed_layer = reconstructed_model.get_layer(layer_name)
        except ValueError:
            continue

        reconstructed_arrays = [
            variable_to_numpy(variable)
            for variable in getattr(reconstructed_layer, "weights", [])
        ]

        for original in original_arrays:
            if not np.issubdtype(original.dtype, np.floating):
                continue
            for index, reconstructed in enumerate(reconstructed_arrays):
                if original.shape == reconstructed.shape and np.issubdtype(reconstructed.dtype, np.floating):
                    total_error += mean_absolute_error(original, reconstructed) * int(original.size)
                    total_values += int(original.size)
                    reconstructed_arrays.pop(index)
                    break

    if total_values:
        return total_error / float(total_values)

    for original, reconstructed in zip(iter_layer_weight_arrays(original_model), iter_layer_weight_arrays(reconstructed_model)):
        if not np.issubdtype(original.dtype, np.floating):
            continue
        if original.shape != reconstructed.shape:
            continue
        total_error += mean_absolute_error(original, reconstructed) * int(original.size)
        total_values += int(original.size)

    return total_error / float(max(1, total_values))


def estimate_kmeans_compression_ratio(model: tf.keras.Model, k: int) -> float:
    original_bits = 0
    compressed_bits = 0
    clusters = max(1, int(k))
    index_bits = max(1, int(math.ceil(math.log2(max(2, clusters)))))
    for values in iter_layer_weight_arrays(model):
        if np.issubdtype(values.dtype, np.floating) and values.size:
            original_bits += int(values.size) * 32
            compressed_bits += int(values.size) * index_bits + clusters * 32
    return original_bits / float(max(1, compressed_bits))


def linear_int8_metrics(model: tf.keras.Model) -> dict[str, float]:
    total_error = 0.0
    total_values = 0
    original_bits = 0
    compressed_bits = 0

    for values in iter_layer_weight_arrays(model):
        if not np.issubdtype(values.dtype, np.floating) or values.size == 0:
            continue

        values32 = values.astype(np.float32, copy=False)
        flat_size = int(values32.size)
        rmin, rmax = float(np.min(values32)), float(np.max(values32))

        if rmax == rmin:
            reconstructed = values32.copy()
        else:
            qmin, qmax = -128, 127
            scale = (rmax - rmin) / float(qmax - qmin)
            zero_point = int(np.clip(round(qmin - rmin / scale), qmin, qmax))
            quantized = np.clip(np.round(values32 / scale + zero_point), qmin, qmax)
            reconstructed = ((quantized.astype(np.float32) - zero_point) * scale).astype(np.float32)

        total_error += mean_absolute_error(values32, reconstructed) * flat_size
        total_values += flat_size
        original_bits += flat_size * 32
        compressed_bits += flat_size * 8 + 64

    return {
        "mae": total_error / float(max(1, total_values)),
        "compression_ratio": original_bits / float(max(1, compressed_bits)),
    }


def make_kmeans_quantized_model(
    base_model: tf.keras.Model,
    k: int,
    clustering: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    finetune_epochs: int,
    batch_size: int,
) -> tuple[tf.keras.Model, dict[str, float]]:
    clustered_model = clustering.cluster_weights(
        base_model,
        number_of_clusters=max(1, int(k)),
        cluster_centroids_init=clustering.CentroidInitialization.LINEAR,
    )
    clustered_model(tf.zeros((1, kTimesteps, kFeatureCount)))
    compile_for_classification(clustered_model)
    finetune(clustered_model, x_train, y_train, finetune_epochs, batch_size)
    model = clustering.strip_clustering(clustered_model)
    compile_for_classification(model)

    return model, {
        "mae": model_weight_mae(base_model, model),
        "compression_ratio": estimate_kmeans_compression_ratio(base_model, k),
    }


# ---------------------------------------------------------------------------
# TFLite export and evaluation
# ---------------------------------------------------------------------------

def make_representative_dataset(x: np.ndarray, sample_count: int):
    limit = max(1, min(int(sample_count), len(x)))

    def representative_dataset():
        for index in range(limit):
            yield [x[index : index + 1].astype(np.float32)]

    return representative_dataset


def export_tflite(
    model: tf.keras.Model,
    artifacts_dir: Path,
    name: str,
    tflite_mode: str = "fp32",
    representative_data: np.ndarray | None = None,
    representative_samples: int = 256,
) -> Path:
    saved_models_dir = artifacts_dir / "saved_models"
    saved_models_dir.mkdir(parents=True, exist_ok=True)

    saved_model_dir = saved_models_dir / name
    tflite_path = artifacts_dir / f"{name}.tflite"

    if saved_model_dir.exists():
        shutil.rmtree(saved_model_dir)

    tf.saved_model.save(model, str(saved_model_dir))
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

    if tflite_mode == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif tflite_mode == "int8":
        if representative_data is None:
            raise ValueError("INT8 TFLite conversion needs representative_data.")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = make_representative_dataset(representative_data, representative_samples)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    tflite_path.write_bytes(converter.convert())
    return tflite_path


def quantize_tensor_for_tflite(values: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    dtype = detail["dtype"]
    if np.issubdtype(dtype, np.floating):
        return values.astype(dtype)
    scale, zero_point = detail["quantization"]
    scale = float(scale) if float(scale) != 0.0 else 1.0
    info = np.iinfo(dtype)
    return np.clip(np.round(values / scale + int(zero_point)), info.min, info.max).astype(dtype)


def dequantize_tensor_from_tflite(values: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    if not np.issubdtype(values.dtype, np.integer):
        return values.astype(np.float32)
    scale, zero_point = detail["quantization"]
    scale = float(scale) if float(scale) != 0.0 else 1.0
    return (values.astype(np.float32) - int(zero_point)) * scale


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
        probabilities = dequantize_tensor_from_tflite(interpreter.get_tensor(output_detail["index"]), output_detail)[0]
        label = int(y[index])
        correct += int(int(np.argmax(probabilities)) == label)
        losses.append(float(-np.log(float(np.clip(probabilities[label], 1e-7, 1.0)))))

    return float(np.mean(losses)), float(correct / max(1, limit))


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def count_nonzero_weights(model: tf.keras.Model) -> tuple[int, int]:
    total = 0
    nonzero = 0
    for layer in model.layers:
        for attr in ("kernel", "bias"):
            variable = getattr(layer, attr, None)
            if variable is None:
                continue
            values = variable.numpy()
            total += int(values.size)
            nonzero += int(np.count_nonzero(values))
    return total, nonzero


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


def add_metrics_row(
    rows: list[dict[str, Any]],
    name: str,
    keras_metrics: dict[str, float],
    tflite_metrics: tuple[float, float],
    tflite_path: Path,
    method: str,
    tflite_mode: str,
    quantization_mae: float | None,
    estimated_compression_ratio: float | None,
    baseline_tflite_bytes: int,
    parameter_counts: tuple[int, int],
) -> None:
    total, nonzero = parameter_counts
    tflite_bytes = tflite_path.stat().st_size
    rows.append(
        {
            "tflite_file": tflite_path.name,
            "method": method,
            "tflite_mode": tflite_mode,
            "validation_accuracy": f"{keras_metrics['validation_accuracy']:.3f}",
            "test_accuracy": f"{keras_metrics['test_accuracy']:.3f}",
            "tflite_test_accuracy": f"{tflite_metrics[1]:.3f}",
            "tflite_file_size_kilobytes": int(tflite_bytes / 1024),
            "parameters_total": total,
        }
    )


def append_model(
    models: list[dict[str, Any]],
    name: str,
    method: str,
    model: tf.keras.Model,
    tflite_mode: str = "fp32",
    quantization_mae: float | None = None,
    estimated_compression_ratio: float | None = None,
) -> None:
    models.append(
        {
            "name": name,
            "method": method,
            "model": model,
            "tflite_mode": tflite_mode,
            "quantization_mae": quantization_mae,
            "estimated_compression_ratio": estimated_compression_ratio,
        }
    )


# ---------------------------------------------------------------------------
# Test sample export
# ---------------------------------------------------------------------------

def save_test_gesture_csv(artifacts_dir: Path, data_dir: str) -> None:
    """Save one resampled gesture-A recording as the on-device test sample."""
    import glob
    import pandas as pd
    from resampler import resample_recording

    candidates = sorted(glob.glob(os.path.join(data_dir, "*_gesture_A.csv")))
    if not candidates:
        return
    df = pd.read_csv(candidates[0])
    df_resampled = resample_recording(df, kTimesteps)
    df_resampled["label"] = str(df["label"].iloc[0]).strip()
    out_path = artifacts_dir / "test_gesture.csv"
    df_resampled.to_csv(out_path, index=False)
    print(f"Saved test gesture CSV: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    clustering = require_tfmot()
    tf.keras.utils.set_random_seed(42)
    np.random.seed(42)

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Load and split gesture data.
    print("Loading gesture data…")
    X, y = load_data(args.data_dir)
    print(f"  {len(X)} samples  shape {X.shape}")
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, args.validation_split)
    print(f"  Train {len(X_train)} / Val {len(X_val)} / Test {len(X_test)}")

    save_test_gesture_csv(artifacts_dir, args.data_dir)

    # Build normalization layer adapted on raw training data.
    # Baking it into the model means C++ can pass raw sensor values directly.
    norm_layer = tf.keras.layers.Normalization(axis=-1)
    norm_layer.adapt(X_train)
    norm_layer.trainable = False
    norm_weights = norm_layer.get_weights()

    # Train baseline.
    baseline = make_model(norm_layer, args.conv_filters, args.dense_units)
    baseline.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=2,
        shuffle=True,
    )

    models: list[dict[str, Any]] = []
    append_model(models, "baseline", "normal FP32 training", baseline)

    # Synapse pruning (zero small weights, architecture-agnostic).
    synapse = prune_synapses(baseline, args.synapse_prune_ratio)
    finetune(synapse, X_train, y_train, args.finetune_epochs, args.batch_size)
    synapse = prune_synapses(synapse, args.synapse_prune_ratio)
    append_model(models, "synapse_pruned", "unstructured small-weight pruning", synapse)

    # Neuron pruning (remove least-important dense1 units).
    neuron = prune_neurons(baseline, args.neuron_prune_ratio, norm_weights)
    finetune(neuron, X_train, y_train, args.finetune_epochs, args.batch_size)
    append_model(models, "neuron_pruned", "structured dense-neuron pruning", neuron)

    # Channel pruning (remove least-important Conv1D filters).
    channel = prune_channels(baseline, args.channel_prune_ratio, norm_weights)
    finetune(channel, X_train, y_train, args.finetune_epochs, args.batch_size)
    append_model(models, "channel_pruned", "structured convolution-channel pruning", channel)

    # FP16 quantization export.
    append_model(models, "baseline_fp16", "FP16 TFLite export", baseline, "fp16", None, 2.0)

    # INT8 quantization export.
    int8_metrics = linear_int8_metrics(baseline)
    append_model(
        models,
        "baseline_int8",
        "linear INT8 TFLite export",
        baseline,
        "int8",
        int8_metrics["mae"],
        int8_metrics["compression_ratio"],
    )

    # K-Means weight clustering.
    kmeans_model, kmeans_metrics = make_kmeans_quantized_model(
        baseline, args.kmeans_k, clustering, X_train, y_train, args.cluster_finetune_epochs, args.batch_size
    )
    append_model(
        models,
        f"kmeans_k{int(args.kmeans_k)}",
        "TFMOT K-Means weight clustering",
        kmeans_model,
        "fp32",
        kmeans_metrics["mae"],
        kmeans_metrics["compression_ratio"],
    )

    # Combined pruning + K-Means + INT8.
    for source_name, source_model in [
        ("synapse_pruned", synapse),
        ("neuron_pruned", neuron),
        ("channel_pruned", channel),
    ]:
        combined_model, combined_metrics = make_kmeans_quantized_model(
            source_model, args.kmeans_k, clustering, X_train, y_train, args.cluster_finetune_epochs, args.batch_size
        )
        append_model(
            models,
            f"{source_name}_kmeans_k{int(args.kmeans_k)}_int8",
            "pruning + TFMOT K-Means weight clustering + INT8 TFLite export",
            combined_model,
            "int8",
            combined_metrics["mae"],
            combined_metrics["compression_ratio"],
        )

    # Export all model variants.
    rows: list[dict[str, Any]] = []
    baseline_tflite_bytes = 1
    for model_info in models:
        name = str(model_info["name"])
        model = model_info["model"]
        tflite_mode = str(model_info["tflite_mode"])
        weights = count_nonzero_weights(model)
        keras_metrics = evaluate_keras_model(model, X_val, y_val, X_test, y_test)
        tflite_path = export_tflite(
            model,
            artifacts_dir,
            name,
            tflite_mode=tflite_mode,
            representative_data=X_train,
            representative_samples=args.representative_samples,
        )
        if name == "baseline":
            baseline_tflite_bytes = tflite_path.stat().st_size

        tflite_metrics = evaluate_tflite(tflite_path, X_test, y_test, args.tflite_eval_samples)
        add_metrics_row(
            rows,
            name,
            keras_metrics,
            tflite_metrics,
            tflite_path,
            str(model_info["method"]),
            tflite_mode,
            model_info["quantization_mae"],
            model_info["estimated_compression_ratio"],
            baseline_tflite_bytes,
            weights,
        )
        print(
            f"Exported {tflite_path} "
            f"keras_acc={keras_metrics['test_accuracy']:.4f} "
            f"tflite_acc={tflite_metrics[1]:.4f}"
        )

    # baseline.tflite → model.tflite for backward compatibility with deploy scripts.
    shutil.copy2(artifacts_dir / "baseline.tflite", artifacts_dir / "model.tflite")

    metrics_path = artifacts_dir / "model_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved model metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
