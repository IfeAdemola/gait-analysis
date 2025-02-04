import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
from plotting import plot_raw_pose, plot_extremas, plot_extrema_frames

def butter_lowpass_filter(data, cutoff=3, fs=30, order=4):
    """Filter signal"""
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

class EventDetector:
    """Detects the heel strike and toe-off events from the pose estimation using the specified algorithm
        - Heel strike (HS) is the moment the heel touches the ground
        - Toe-off (TO) is the moment the toe touches the ground
    """
    
    def __init__(self, algorithm="zeni", frame_rate=25):
        self.algorithm = algorithm
        self.frame_rate = frame_rate

    def detect_heel_toe_events(self, pose_data):
        """
        Detects heel strike and toe-off events based on the specified algorithm.
        
        Parameters:
        - pose_data: Pandas DataFrame containing pose estimation data with columns
                     for heel, and toe markers in x, y, z coordinates.
        
        Returns:
        - Dictionary with timestamp of heel strike and toe-off events for left and right foot.
            - {
                    "HS_left": [...], "TO_left": [...],
                    "HS_right": [...], "TO_right": [...]
                }
        """
        if self.algorithm == "zeni":
            return self._detect_events_zeni(pose_data)
        elif self.algorithm == "dewitt":
            return self._detect_events_dewitt(pose_data)
        elif self.algorithm == "hreljac":
            return self._detect_events_hreljac(pose_data)
        else:
            raise ValueError("Unsupported algorithm")

    #TODO: check if I can add url of paper to docstring
    def _detect_events_zeni(self, pose_data):
        """Detects heel strike (HS) and toe-off (TO) events using Zeni et al.'s method."""
        
        # Plot raw heel and toe time series
        plot_raw_pose(pose_data, self.frame_rate, output_dir="plots")

        # TODO: Could be better as: "HS_left": pd.Series([])
        events = {"HS_left": [], "TO_left": [], "HS_right": [], "TO_right": []}
        
        # Compute sacrum coordinates as the midpoint of left and right hip
        for axis in ['x', 'y', 'z']:
            pose_data[('sacrum', axis)] = (pose_data[('left_hip', axis)] + pose_data[('right_hip', axis)]) / 2
        
        sacrum_z = pose_data[('sacrum', 'z')]

        # # Compute relative X positions for left and right foot
        # heel_left_z = pose_data[('left_heel', 'z')] 
        # heel_right_z = pose_data[('right_heel', 'z')] 
        # toe_left_z = pose_data[('left_foot_index', 'z')] 
        # toe_right_z = pose_data[('right_foot_index', 'z')]


        # # Compute relative displacements
        # extremas_data = {-
        # "heel_left": heel_left_z - sacrum_z,
        # "heel_right": heel_right_z - sacrum_z,
        # "toe_left": toe_left_z - sacrum_z,
        # "toe_right": toe_right_z - sacrum_z
        # }

        # # Filter signals
        # filtered_extremas = {key: butter_lowpass_filter(value) for key, value in extremas_data.items()}
        
        # # Detect peaks and valleys
        # detected_extremas = {key: detect_extremas(value) for key, value in filtered_extremas.items()}
        
        # # Convert to time-based peaks
        # extrema_times = {key: (peaks/self.frame_rate, valleys/self.frame_rate) for key, (peaks, valleys) in detected_extremas.items()}



        # Compute relative X positions for left and right foot
        heel_left_z = pose_data[('left_heel', 'z')] - pose_data[('sacrum', 'z')]
        heel_right_z = pose_data[('right_heel', 'z')] - pose_data[('sacrum', 'z')]
        toe_left_z = pose_data[('left_foot_index', 'z')] - pose_data[('sacrum', 'z')]
        toe_right_z = pose_data[('right_foot_index', 'z')] - pose_data[('sacrum', 'z')]

        # Calculate threshold as the mean of the relative positions
        threshold_heel_left = np.mean(heel_left_z)
        threshold_heel_right = np.mean(heel_right_z)
        threshold_toe_left = np.mean(toe_left_z)
        threshold_toe_right = np.mean(toe_right_z)

        # Detect peaks (HS) and valleys (TO) using find_peaks
        hs_left_idx, _ = find_peaks(heel_left_z, height=threshold_heel_left)
        to_left_idx, _ = find_peaks(-toe_left_z, height=-threshold_toe_left)
        hs_right_idx, _ = find_peaks(heel_right_z, height=threshold_heel_right)
        to_right_idx, _ = find_peaks(-toe_right_z, height=-threshold_toe_right)


        extremas_data = {
            "heel_left":  hs_left_idx / self.frame_rate,
            "heel_right": to_left_idx / self.frame_rate,
            "toe_left": hs_right_idx / self.frame_rate,
            "toe_right": to_right_idx / self.frame_rate
        }

        plot_extremas(pose_data, self.frame_rate, output_dir="plots")

    
        # Find the maximum length of lists
        max_length = max(len(v) for v in extremas_data.values())

        # Create a DataFrame for detected events with NaN padding
        events = pd.DataFrame({
            key: pd.Series(values + [np.nan] * (max_length - len(values))) 
            for key, values in extremas_data.items()
        })


        return pd.DataFrame(events)

    # def _detect_events_dewitt(self, pose_data):
    #     """Detects toe-off (TO) events using the DeWitt et al. method."""
        
    #     events = {"HS_left": [], "TO_left": [], "HS_right": [], "TO_right": []}

    #     # Extract vertical (y-axis) data for the toe and heel markers
    #     toe_left_y = pose_data[('left_foot_index', 'y')]
    #     toe_right_y = pose_data[('right_foot_index', 'y')]
    #     heel_right_y = pose_data[('right_heel', 'y')]
    #     heel_left_y = pose_data[('left_heel', 'y')]

    #     # Calculate time between frames
    #     tint = 1 / self.frame_rate

    #     # Get Heel Strike times using Zeni method
    #     hs_to_zeni = self._detect_events_zeni(pose_data)
    #     hs_zeni_left = hs_to_zeni['HS_left']
    #     hs_zeni_right = hs_to_zeni['HS_right']

    #     # Compute acceleration and jerk for the toe marker (finite differences)
    #     toe_left_acc = np.gradient(np.gradient(toe_left_y, tint), tint)
    #     toe_right_acc = np.gradient(np.gradient(toe_right_y, tint), tint)

    #     toe_left_jerk = np.gradient(toe_left_acc, tint)
    #     toe_right_jerk = np.gradient(toe_right_acc, tint)

    #     # Find local maxima in toe acceleration (candidates for TOEY)
    #     toe_left_acc_peaks, _ = find_peaks(toe_left_acc)
    #     toe_right_acc_peaks, _ = find_peaks(toe_right_acc)

    #     # Process the detected peaks and determine toe-off times
    #     toe_off_times_left = self._process_toe_off_peaks(toe_left_acc_peaks, hs_zeni_left, toe_left_jerk)
    #     toe_off_times_right = self._process_toe_off_peaks(toe_right_acc_peaks, hs_zeni_right, toe_right_jerk)

    #     # Store events
    #     events["HS_left"] = hs_zeni_left
    #     events["TO_left"] = toe_off_times_left
    #     events["HS_right"] = hs_zeni_right
    #     events["TO_right"] = toe_off_times_right

    #     return pd.DataFrame(events)

    # def _process_toe_off_peaks(self, toe_acc_peaks, hs_times, toe_jerk):
    #     """Process peaks in toe acceleration to find toe-off times."""
    #     toe_off_times = []
    #     for peak_idx in toe_acc_peaks:
    #         # Ensure the peak lies between the previous HS and next HS
    #         prev_hs = max([i for i, hs in enumerate(hs_times) if hs < peak_idx])
    #         next_hs = min([i for i, hs in enumerate(hs_times) if hs > peak_idx])

    #         # Find jerk transition through zero (TOEY criterion)
    #         t1, t2 = self._find_jerk_transition(toe_jerk, peak_idx, prev_hs, next_hs)

    #         if t1 is not None and t2 is not None:
    #             # Interpolate to find the exact time of jerk = 0
    #             J_t1 = toe_jerk[t1]
    #             J_t2 = toe_jerk[t2]
    #             time_toey = t1 + (J_t1 / (J_t1 - J_t2)) * (1 / self.frame_rate)
    #             toe_off_times.append(time_toey)
    #     return toe_off_times

    # def _find_jerk_transition(self, jerk, peak_idx, prev_hs, next_hs):
    #     """Find the time where jerk transitions from positive to negative."""
    #     t1, t2 = None, None
    #     for i in range(peak_idx, next_hs):
    #         if jerk[i] > 0 and jerk[i + 1] < 0:
    #             t1, t2 = i, i + 1
    #             break
    #     return t1, t2

    # def _detect_events_hreljac(self, pose_data):
    #     """Detects events using the Hreljac algorithm."""
        
    #     events = {"HS_left": [], "TO_left": [], "HS_right": [], "TO_right": []}
        
    #     # Frame interval
    #     frame_interval = 1 / self.frame_rate
        
    #     # Iterate over both sides (left and right foot)
    #     for side in ["left", "right"]:
    #         # Extract relevant columns for the current foot
    #         heel_y = pose_data[(f'{side}_heel', 'y')]
    #         toe_x = pose_data[(f'{side}_foot_index', 'x')]

    #         # Compute derivatives for heel vertical position (HS detection)
    #         heel_accel = heel_y.diff().diff() / frame_interval**2  # 2nd derivative
    #         heel_jerk = heel_accel.diff() / frame_interval         # 3rd derivative

    #         # Compute derivatives for toe horizontal position (TO detection)
    #         toe_accel = toe_x.diff().diff() / frame_interval**2    # 2nd derivative
    #         toe_jerk = toe_accel.diff() / frame_interval           # 3rd derivative

    #         # Detect HS (local maxima in heel acceleration where jerk crosses zero)
    #         hs_indices = [i for i in range(1, len(heel_jerk) - 1) if heel_jerk[i - 1] > 0 and heel_jerk[i + 1] < 0]
    #         hs_times = [i * frame_interval + (heel_jerk[i] / (heel_jerk[i] - heel_jerk[i + 1])) * frame_interval for i in hs_indices]

    #         # Detect TO (local maxima in toe acceleration where jerk crosses zero)
    #         to_indices = [i for i in range(1, len(toe_jerk) - 1) if toe_jerk[i - 1] > 0 and toe_jerk[i + 1] < 0]
    #         to_times = [i * frame_interval + (toe_jerk[i] / (toe_jerk[i] - toe_jerk[i + 1])) * frame_interval for i in to_indices]

    #         # Save results for the current side
    #         events[f"HS_{side}"] = hs_times
    #         events[f"TO_{side}"] = to_times

        # return pd.DataFrame(events)
