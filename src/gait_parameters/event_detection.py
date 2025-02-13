# gait_pipeline/event_detection.py

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from utils import detect_extremas
from plotting import plot_raw_pose, plot_extremas, plot_extrema_frames


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
    
    # TODO: More suitable name than "forward_movement"
    def _detect_events_zeni(self, pose_data):
        """Detects heel strike (HS) and toe-off (TO) events using Zeni et al.'s method."""
        
        all_forward_movement = {}
        all_extrema_data = {}
        event_extrema_data = {}
    
        # Define foot landmarks and their corresponding event names
        foot_landmarks = {
            "heel_left": "left_heel",
            "heel_right": "right_heel",
            "toe_left": "left_foot_index",
            "toe_right": "right_foot_index"
        }

        # Detect events using the generalized approach
        for landmark_name, landmark in foot_landmarks.items():
            forward_movement = pose_data[(landmark, 'z')] - pose_data[('sacrum', 'z')]
            
            # Store forward movement in a dictionary
            all_forward_movement[landmark_name] = forward_movement

            #Detect peaks and valleys
            peaks, valleys = detect_extremas(forward_movement)

            # Store both peaks and valleys for each event
            all_extrema_data[landmark_name] = {
                "peaks": peaks / self.frame_rate,  # Convert indices to time
                "valleys": valleys / self.frame_rate
            }
            
            if "heel" in landmark_name:
                idx = peaks  # Heel Strike (HS)
            else:
                idx = valleys  # Toe-Off (TO)
            
            event_extrema_data[landmark_name] = idx / self.frame_rate  # Convert to time    
            
        plot_extremas(all_forward_movement, all_extrema_data, self.frame_rate, output_dir="plots")

        max_length = max(len(v) for v in event_extrema_data.values())
        events = pd.DataFrame({
            key: pd.Series(list(values) + [np.nan] * (max_length - len(values)))
            for key, values in event_extrema_data.items()
        })
        return events
    
    # The _detect_events_dewitt and _detect_events_hreljac methods can be updated similarly if needed.
