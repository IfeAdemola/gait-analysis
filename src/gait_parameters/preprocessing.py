import numpy as np
import pandas as pd

from scipy import signal
from scipy.signal import medfilt

from utils import log

class Preprocessor:
    def __init__(self, pose_data):
        """
        Initializes the Preprocessor with pose data.

        Parameters:
            pose_data (DataFrame): MultiIndex DataFrame containing gait coordinates (x, y, z) for various landmarks.
        """
        if not isinstance(pose_data, pd.DataFrame):
            raise ValueError("pose_data must be a pandas DataFrame")
        self.pose_data = pose_data
        log("Preprocessor initialized with pose data.", level="INFO")

    def median_filter(self, window_size=5):
        """
        Applies a median filter to positional coordinates.

        Parameters:
            window_size (int): Size of the window for the median filter. Must be an odd number.

        Returns:
            DataFrame: Pose data with missing values interpolated and coordinates filtered.
        """
        if window_size % 2 == 0:
            raise ValueError("window_size must be an odd number.")
        
        log("Applying median filter...", level="INFO")
        filtered_pose_data = self.pose_data.copy()

        # Interpolate missing values for numeric columns
        numeric_columns = filtered_pose_data.select_dtypes(include=['number']).columns
        filtered_pose_data[numeric_columns] = (
            filtered_pose_data[numeric_columns].interpolate(method='linear').ffill().bfill()
        )

        # Apply median filter to each landmark's coordinates
        for landmark in filtered_pose_data.columns.levels[0]:
            for coord in ['x', 'y', 'z']:
                if (landmark, coord) in filtered_pose_data.columns:
                    filtered_pose_data[(landmark, coord)] = medfilt(
                        filtered_pose_data[(landmark, coord)], kernel_size=window_size
                    )
        
        log("Median filtering complete.", level="INFO")
        return filtered_pose_data

    def normalize(self, window_size=5):
        """
        Normalizes gait data by the distance between specific landmarks (e.g., knee and ankle).

        Parameters:
            window_size (int): Size of the window for the median filter. Must be an odd number.

        Returns:
            DataFrame: Normalized pose data.
        """
        log("Starting normalization process...", level="INFO")
        filtered_pose_data = self.median_filter(window_size=window_size)

        # Calculate distances between left/right knee and ankle
        try:
            left_leg_length = np.sqrt(
                (filtered_pose_data[('left_knee', 'x')] - filtered_pose_data[('left_ankle', 'x')]) ** 2 +
                (filtered_pose_data[('left_knee', 'y')] - filtered_pose_data[('left_ankle', 'y')]) ** 2 +
                (filtered_pose_data[('left_knee', 'z')] - filtered_pose_data[('left_ankle', 'z')]) ** 2
            )
            right_leg_length = np.sqrt(
                (filtered_pose_data[('right_knee', 'x')] - filtered_pose_data[('right_ankle', 'x')]) ** 2 +
                (filtered_pose_data[('right_knee', 'y')] - filtered_pose_data[('right_ankle', 'y')]) ** 2 +
                (filtered_pose_data[('right_knee', 'z')] - filtered_pose_data[('right_ankle', 'z')]) ** 2
            )
        except KeyError as e:
            log(f"Missing columns for leg length calculation: {e}", level="ERROR")
            raise KeyError(f"Missing expected columns in filtered_pose_data: {e}")
        
        # Ensure leg lengths are arrays
        if np.isscalar(left_leg_length) or np.isscalar(right_leg_length):
            raise ValueError("Leg lengths should be arrays, but scalar values were found")


        # Smooth leg lengths using median filter
        left_leg_length = np.median(medfilt(left_leg_length, kernel_size=5))
        right_leg_length = np.median(medfilt(right_leg_length, kernel_size=5))

        normalized_data = filtered_pose_data.copy()

        # Normalize each landmark's coordinates
        for landmark in filtered_pose_data.columns.levels[0]:
            for coord in ['x', 'y', 'z']:
                if (landmark, coord) in filtered_pose_data.columns:
                    if 'left' in landmark:
                        normalized_data[(landmark, coord)] /= left_leg_length
                    elif 'right' in landmark:
                        normalized_data[(landmark, coord)] /= right_leg_length

        log("Normalization complete.", level="INFO")
        return normalized_data

    def preprocess(self, window_size=5):
            """
            Executes the full preprocessing pipeline: filtering and normalization.

            Parameters:
                window_size (int): Size of the window for median filtering.

            Returns:
                DataFrame: Fully preprocessed pose data.
            """
            log("Starting full preprocessing pipeline...", level="INFO")
            normalized_data = self.normalize(window_size=window_size)
            log("Preprocessing pipeline complete.", level="INFO")
            return normalized_data







def resample_signal():
    """DTW: dynamic time warping"""


