import os
import glob
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# using accelerometer + gyroscope + magnetometer (9-axis)
features = [
    "accel_x","accel_y","accel_z",
    "gyro_x","gyro_y","gyro_z",
    "mag_x","mag_y","mag_z"
]

label_map = {"A":0, "B":1, "C":2, "X":3}


def resample_recording(df, num_points):
    """Resample a recording to a fixed number of timesteps via linear interpolation."""
    n = len(df)
    old_t = np.linspace(0, 1, n)
    new_t = np.linspace(0, 1, num_points)
    resampled = {}
    for col in features:
        f = interp1d(old_t, df[col].values, kind="linear")
        resampled[col] = f(new_t)
    return pd.DataFrame(resampled)


def load_gesture_sequences(data_dir, num_points=50):
    """Load all gesture CSVs from data_dir, resample, and return (X, y) arrays."""
    X_sequences = []
    y_labels = []
    for file_path in glob.glob(os.path.join(data_dir, "*.csv")):
        df = pd.read_csv(file_path)
        label_char = str(df["label"].iloc[0]).strip()
        if label_char not in label_map:
            continue
        label = label_map[label_char]
        df_resampled = resample_recording(df, num_points)
        X_sequences.append(df_resampled[features].values)
        y_labels.append(label)
    return np.array(X_sequences), np.array(y_labels)


if __name__ == "__main__":
    output_folder = "resampled data"
    os.makedirs(output_folder, exist_ok=True)

    for file_path in glob.glob(os.path.join("data", "*.csv")):
        df = pd.read_csv(file_path)
        label_char = str(df["label"].iloc[0]).strip()
        if label_char not in label_map:
            continue
        df_resampled = resample_recording(df, 50)
        df_resampled["label"] = label_char
        original_name = os.path.splitext(os.path.basename(file_path))[0]
        df_resampled.to_csv(os.path.join(output_folder, f"{original_name}_resampled.csv"), index=False)

    X_seq, y_seq = load_gesture_sequences("data")
    print("Shape:", X_seq.shape)
    print(f"Saved resampled files to: {output_folder}")