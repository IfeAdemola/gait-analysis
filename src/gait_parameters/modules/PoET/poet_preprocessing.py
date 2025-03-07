import os
import pandas as pd
from .lib.patients import Patient, PatientCollection


def construct_data(csv_files, fs, labels=None, scaling_factor=1, verbose=True, smooth=None):
    if isinstance(scaling_factor, int):
        scaling_factor = [scaling_factor] * len(csv_files)
    if isinstance(fs, int):
        fs = [fs] * len(csv_files)

    # Define keypoint versions using normalized names (without the "marker_" prefix).
    keypoint_versions = {
        "v1": [
            "index_finger_tip_left",
            "index_finger_tip_right",
            "middle_finger_tip_left",
            "middle_finger_tip_right",
            "left_elbow",
            "right_elbow"
        ],
        "v2": [
            "index_finger_tip_left",
            "index_finger_tip_right",
            "middle_finger_tip_left",
            "middle_finger_tip_right",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist"
        ]
    }
    
    patients = []
    for i, file in enumerate(csv_files):
        # Get filename without extension.
        file_name = os.path.basename(file).split('.')[0]
        
        if verbose:
            print('Loading: {}'.format(file_name))
        
        # Load the CSV file with a multi-index header.
        pose_estimation = pd.read_csv(file, header=[0, 1], index_col=0)
        
        # Enforce that the CSV has a MultiIndex header.
        if not isinstance(pose_estimation.columns, pd.MultiIndex):
            raise ValueError(f"CSV file {file_name} does not have a MultiIndex header. Please update your CSV generation process.")
        
        # Normalize the multi-index columns: lowercase all strings and remove "marker_" prefix if present.
        pose_estimation = normalize_multiindex_columns(pose_estimation)
        
        # Get available keypoints from the first level of the MultiIndex.
        available_keypoints = set(pose_estimation.columns.get_level_values(0))
        
        # Decide which version to use.
        if all(kp in available_keypoints for kp in keypoint_versions["v2"]):
            keypoints = keypoint_versions["v2"]
            if verbose:
                print(f"Using version v2 keypoints for {file_name}.")
        elif all(kp in available_keypoints for kp in keypoint_versions["v1"]):
            keypoints = keypoint_versions["v1"]
            if verbose:
                print(f"Using version v1 keypoints for {file_name}.")
        else:
            print(f"Error: Unknown keypoint version for {file_name}. Missing required keypoints.")
            continue

        # Subset to only the keypoint columns.
        pose_estimation = pose_estimation.loc[:, pose_estimation.columns.get_level_values(0).isin(keypoints)]
        
        # Debug: Print first 2 rows of the processed pose_estimation.
        if verbose:
            print("First 2 rows after normalizing and subsetting keypoints:")
            print(pose_estimation.head(2))
        
        # Construct a Patient object.
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


def normalize_multiindex_columns(data):
    """
    Enforces that data.columns is a MultiIndex. Converts all elements of the MultiIndex to lowercase strings.
    If the first-level element starts with "marker_", that prefix is removed.
    This normalization ensures that downstream processing uses the new standardized keypoint names.
    """
    if not isinstance(data.columns, pd.MultiIndex):
        raise ValueError("Expected data.columns to be a MultiIndex. Received: {}".format(type(data.columns)))
    
    new_tuples = []
    for tup in data.columns:
        # Process the first element: remove "marker_" prefix if present, then lowercase.
        first = str(tup[0]).strip().lower()
        if first.startswith("marker_"):
            first = first[len("marker_"):]
        # Process the rest of the tuple elements.
        rest = tuple(str(x).strip().lower() for x in tup[1:])
        new_tuples.append((first,) + rest)
    data.columns = pd.MultiIndex.from_tuples(new_tuples)
    return data
