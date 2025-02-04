# gait_pipeline/event_detection.py

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.decomposition import PCA
from plotting import plot_raw_pose, plot_extremas, plot_extrema_frames

def butter_lowpass_filter(data, cutoff=3, fs=30, order=4):
    """Filter signal using a lowpass Butterworth filter."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def detect_extremas(signal):
    """Find peaks and valleys in the filtered signal."""
    threshold = np.mean(signal)
    peaks, _ = find_peaks(signal, height=threshold)
    valleys, _ = find_peaks(-signal, height=-threshold)
    return peaks, valleys

def determine_gait_direction_sliding_window(pose_data, marker="sacrum", window_size=100, step_size=50):
    """
    Determine the gait direction using a sliding window approach on the specified marker.
    
    For each window, this function computes the dominant movement direction using PCA on the x and z
    coordinates, and returns the rotation angle (in radians) required to align that direction with the positive z-axis.
    
    Returns:
        List of tuples (window_center_index, rotation_angle)
    """
    angles = []
    num_frames = len(pose_data)
    for start in range(0, num_frames - window_size + 1, step_size):
        end = start + window_size
        window_data = pose_data.iloc[start:end]
        x_coords = window_data[(marker, 'x')].to_numpy()
        z_coords = window_data[(marker, 'z')].to_numpy()
        positions = np.column_stack((x_coords, z_coords))
        
        pca = PCA(n_components=1)
        pca.fit(positions)
        principal_vector = pca.components_[0]  # [v_x, v_z]
        
        # Compute the angle between the principal vector and the positive z-axis.
        # np.arctan2 returns the angle (in radians) between the vector (v_x, v_z) and the positive x-axis;
        # here we compute arctan2(v_x, v_z) so that 0 means aligned with positive z.
        angle = np.arctan2(principal_vector[0], principal_vector[1])
        # To align with the positive z-axis, we need to rotate by -angle.
        window_center = start + window_size // 2
        angles.append((window_center, -angle))
    return angles

def compute_framewise_rotation_angles(pose_data, marker="sacrum", window_size=100, step_size=50):
    """
    Computes a rotation angle for each frame by interpolating the angles computed from a sliding window approach.
    
    Returns:
        A 1D numpy array of rotation angles (in radians), one for each frame.
    """
    sliding_angles = determine_gait_direction_sliding_window(pose_data, marker, window_size, step_size)
    if not sliding_angles:
        return np.zeros(len(pose_data))
    centers, window_angles = zip(*sliding_angles)
    centers = np.array(centers)
    window_angles = np.array(window_angles)
    num_frames = len(pose_data)
    frame_indices = np.arange(num_frames)
    # Linear interpolation of the computed angles over all frames
    framewise_angles = np.interp(frame_indices, centers, window_angles)
    return framewise_angles

def rotate_pose_data_framewise(pose_data, rotation_angles):
    """
    Rotates the x and z coordinates of pose_data for each frame based on the corresponding rotation angle.
    
    Parameters:
        pose_data (pd.DataFrame): DataFrame with MultiIndex columns (marker, axis).
        rotation_angles (np.array): 1D array of rotation angles (in radians), one per frame.
    
    Returns:
        pd.DataFrame: The rotated pose data.
    """
    new_data = pose_data.copy()
    markers = pose_data.columns.get_level_values(0).unique()
    # Loop over each marker that has x and z coordinates.
    for marker in markers:
        if (marker, 'x') in pose_data.columns and (marker, 'z') in pose_data.columns:
            x_orig = pose_data[(marker, 'x')].to_numpy()
            z_orig = pose_data[(marker, 'z')].to_numpy()
            # Apply per-frame rotation.
            new_x = x_orig * np.cos(rotation_angles) - z_orig * np.sin(rotation_angles)
            new_z = x_orig * np.sin(rotation_angles) + z_orig * np.cos(rotation_angles)
            new_data[(marker, 'x')] = new_x
            new_data[(marker, 'z')] = new_z
    return new_data

class EventDetector:
    """
    Detects the heel strike (HS) and toe-off (TO) events from pose estimation data using a specified algorithm.
    
    Now incorporates dynamic rotation of the coordinate system by computing the local gait direction
    over a sliding window, then rotating the pose data accordingly.
    """
    
    def __init__(self, algorithm="zeni", frame_rate=25, window_size=100, step_size=50):
        """
        Parameters:
            algorithm (str): Which detection algorithm to use ("zeni", "dewitt", etc.).
            frame_rate (int): Frame rate of the input data.
            window_size (int): Number of frames per window for gait direction estimation.
            step_size (int): Number of frames to shift the window on each step.
        """
        self.algorithm = algorithm
        self.frame_rate = frame_rate
        self.window_size = window_size
        self.step_size = step_size
    
    def detect_heel_toe_events(self, pose_data):
        """
        Detects heel strike and toe-off events using the selected algorithm.
        
        First, it computes a framewise rotation angle using a sliding window approach on the sacrum marker,
        rotates the pose data accordingly (so that the dominant gait direction aligns with the positive z-axis),
        and then runs the event detection algorithm.
        
        Parameters:
            pose_data (pd.DataFrame): Pose estimation data.
            
        Returns:
            pd.DataFrame: DataFrame with detected event times.
        """
        # Compute framewise rotation angles based on the sacrum trajectory.
        rotation_angles = compute_framewise_rotation_angles(
            pose_data,
            marker="sacrum",
            window_size=self.window_size,
            step_size=self.step_size
        )
        # Rotate the pose data framewise.
        rotated_pose_data = rotate_pose_data_framewise(pose_data, rotation_angles)
        
        # (Optional) Plot the rotated raw pose data for debugging/visualization.
        plot_raw_pose(rotated_pose_data, self.frame_rate, output_dir="plots")
        
        # Proceed with event detection using the rotated data.
        if self.algorithm == "zeni":
            return self._detect_events_zeni(rotated_pose_data)
        elif self.algorithm == "dewitt":
            return self._detect_events_dewitt(rotated_pose_data)
        elif self.algorithm == "hreljac":
            return self._detect_events_hreljac(rotated_pose_data)
        else:
            raise ValueError("Unsupported algorithm")
    
    def _detect_events_zeni(self, pose_data):
        """Detects heel strike (HS) and toe-off (TO) events using Zeni et al.'s method."""
        # Compute sacrum as the midpoint of the left and right hips.
        for axis in ['x', 'y', 'z']:
            pose_data[('sacrum', axis)] = (pose_data[('left_hip', axis)] + pose_data[('right_hip', axis)]) / 2
        
        # Using the rotated coordinate system, assume that the forward movement is now along the z-axis.
        heel_left_forward = pose_data[('left_heel', 'z')] - pose_data[('sacrum', 'z')]
        heel_right_forward = pose_data[('right_heel', 'z')] - pose_data[('sacrum', 'z')]
        toe_left_forward = pose_data[('left_foot_index', 'z')] - pose_data[('sacrum', 'z')]
        toe_right_forward = pose_data[('right_foot_index', 'z')] - pose_data[('sacrum', 'z')]
        
        # Calculate thresholds as the mean of the forward positions.
        threshold_heel_left = np.mean(heel_left_forward)
        threshold_heel_right = np.mean(heel_right_forward)
        threshold_toe_left = np.mean(toe_left_forward)
        threshold_toe_right = np.mean(toe_right_forward)
        
        # Detect events: peaks (HS) and valleys (TO) using the thresholds.
        hs_left_idx, _ = find_peaks(heel_left_forward, height=threshold_heel_left)
        hs_right_idx, _ = find_peaks(heel_right_forward, height=threshold_heel_right)
        to_left_idx, _ = find_peaks(-toe_left_forward, height=-threshold_toe_left)
        to_right_idx, _ = find_peaks(-toe_right_forward, height=-threshold_toe_right)
        
        extremas_data = {
            "heel_left": hs_left_idx / self.frame_rate,
            "heel_right": hs_right_idx / self.frame_rate,
            "toe_left": to_left_idx / self.frame_rate,
            "toe_right": to_right_idx / self.frame_rate
        }
        
        plot_extremas(pose_data, self.frame_rate, output_dir="plots")
    
        max_length = max(len(v) for v in extremas_data.values())
        events = pd.DataFrame({
            key: pd.Series(list(values) + [np.nan] * (max_length - len(values)))
            for key, values in extremas_data.items()
        })
        return events
    
    # The _detect_events_dewitt and _detect_events_hreljac methods can be updated similarly if needed.
