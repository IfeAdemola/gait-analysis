import os
import pandas as pd
from .lib.patients import Patient, PatientCollection


def construct_data(csv_files, fs, labels=None, scaling_factor=1, verbose=True, smooth=None):
    if isinstance(scaling_factor, int):
        scaling_factor = [scaling_factor] * len(csv_files)
    if isinstance(fs, int):
        fs = [fs] * len(csv_files)

    patients = []
    for i, file in enumerate(csv_files):
        # Get filename without extension.
        file_name = os.path.basename(file).split('.')[0]
        
        if verbose:
            print('Loading: {}'.format(file_name))
        
        # Load the CSV file.
        pose_estimation = pd.read_csv(file, header=0, index_col=0)
        
        # Flatten multi-index columns and convert to lowercase.
        pose_estimation = flatten_columns(pose_estimation)
        
        # Define the keypoints we need to consolidate.
        keypoints = [
            "index_finger_tip_left",
            "index_finger_tip_right",
            "middle_finger_tip_left",
            "middle_finger_tip_right",
            "left_elbow",
            "right_elbow"
        ]
        
        # Consolidate coordinate columns into single columns.
        pose_estimation = flatten_and_consolidate_keypoints(pose_estimation, keypoints)
        
        # Debug: Print first 2 rows of the processed pose_estimation
        if verbose:
            print("First 2 rows after flattening and consolidating keypoints:")
            print(pose_estimation.head(2))
        
        # Subset to only the consolidated keypoint columns.
        try:
            pose_estimation = pose_estimation[keypoints]
        except KeyError as e:
            print(f"Error: One or more expected keypoint columns are missing for {file_name}: {e}")
            continue

        # Construct a patient object.
        p = Patient(
            pose_estimation,
            fs[i],
            patient_id=file_name,
            likelihood_cutoff=0,
            label=labels[i] if labels is not None else None,
            low_cut=0,
            high_cut=None,
            clean=True,
            scaling_factor=scaling_factor[i],
            normalize=True,
            spike_threshold=10,
            interpolate_pose=True,
            smooth=smooth,
        )
        patients.append(p)

    # Construct patient collection.
    pc = PatientCollection()
    pc.add_patient_list(patients)
    
    return pc


def flatten_columns(data):
    new_cols = []
    for col in data.columns.values:
        # Only join if the column is a tuple (i.e., multi-index)
        if isinstance(col, tuple):
            new_col = '_'.join([str(x).strip() for x in col if str(x).strip() != '']).lower()
        else:
            new_col = str(col).lower()
        new_cols.append(new_col)
    data.columns = new_cols
    return data


def flatten_and_consolidate_keypoints(df, keypoints):
    """
    For each keypoint in the list, consolidate its x, y, and z coordinate columns into a single column.
    The function expects the coordinate columns to be named in the flattened format, e.g.:
        "index_finger_tip_left_x", "index_finger_tip_left_y", "index_finger_tip_left_z"
    It creates a new column with the keypoint base name (e.g. "index_finger_tip_left")
    that contains a list of the three coordinates for each row.
    """
    for kp in keypoints:
        x_col = f"{kp}_x"
        y_col = f"{kp}_y"
        z_col = f"{kp}_z"
        if {x_col, y_col, z_col}.issubset(set(df.columns)):
            # Combine the three coordinate columns into a list for each row.
            df[kp] = df[[x_col, y_col, z_col]].values.tolist()
        else:
            print(f"Warning: Missing columns for keypoint '{kp}'. Expected: {x_col}, {y_col}, {z_col}.")
    return df
