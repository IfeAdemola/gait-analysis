import numpy as np
import pandas as pd

from utils.gait_parameters import prepare_gait_dataframe
from utils.helpers import save_csv

class GaitParameters:
    @staticmethod
    def compute_parameters(events, save_path=None):
        """
        Computes gait parameters for left and right sides and returns a DataFrame with results.

        Args:
            events (dict or pd.DataFrame): Contains heel strike ('HS_side') and toe-off ('TO_side') events.
            save_path (str): Path where the resulting CSV file should be saved.

        Returns:
            pd.DataFrame: DataFrame with gait parameters.
        """
        if isinstance(events, dict):
            events = pd.DataFrame(events)  # Convert dict to DataFrame if needed

        # Create empty DataFrame
        gait_df = prepare_gait_dataframe()

        for side in ['left', 'right']:
            other_side = 'left' if side == 'right' else 'right'

            # Compute parameters
            gait_df[(side, 'stride_duration')] = events[f'HS_{side}'].shift(-1) - events[f'HS_{side}']
            gait_df[(side, 'step_duration')] = events[f'HS_{side}'].shift(-1) - events[f'HS_{other_side}']
            gait_df[(side, 'cadence')] = 60 / gait_df[(side, 'step_duration')].replace(0, np.nan)
            gait_df[(side, 'initial_double_support')] = events[f'TO_{other_side}'] - events[f'HS_{side}']
            gait_df[(side, 'terminal_double_support')] = events[f'TO_{side}'] - events[f'HS_{side}'].shift(-1)
            gait_df[(side, 'single_limb_support')] = events[f'HS_{other_side}'] - events[f'TO_{side}']
            gait_df[(side, 'stance')] = events[f'TO_{side}'] - events[f'HS_{side}']
            gait_df[(side, 'swing')] = events[f'HS_{side}'].shift(-1) - events[f'TO_{side}']

            # Compute step and stride lengths
            gait_df[(side, 'step_length')] = GaitParameters.compute_step_length(events, side, other_side)
            gait_df[(side, 'stride_length')] = gait_df[(side, 'step_length')].shift(-1) + gait_df[(side, 'step_length')]
            gait_df[(side, 'gait_speed')] = gait_df[(side, 'stride_length')] / gait_df[(side, 'stride_duration')]

        # Asymmetry calculations
        for feature in ['stride_duration', 'step_duration', 'stance', 'swing',
                        'terminal_double_support', 'initial_double_support', 
                        'double_support', 'single_limb_support']:
            gait_df[('asymmetry', feature)] = (
                abs(gait_df[('left', feature)] - gait_df[('right', feature)])
                / gait_df[[('left', feature), ('right', feature)]].max(axis=1) * 100
            )

        return gait_df

    

    @staticmethod
    def compute_step_length(events, side, other_side):
        """
        Compute the step length for a given side. This requires spatial data for the markers.

        Args:
            events (dict): Dictionary containing heel strike and toe-off events.
            side (str): 'left' or 'right'.
            other_side (str): 'left' or 'right' (opposite side of the foot).
        
        Returns:
            pd.Series: Step lengths for the given side.
        """
        # Assuming the spatial data contains x, y, z coordinates for the heel and toe markers for each foot
        # Calculate step length by using the Euclidean distance between heel and toe markers.
        # For simplicity, using just x and y coordinates (can add z for 3D spatial data).
        
        # Example spatial data (replace with actual spatial data)
        left_heel = events[f'HS_left']
        right_heel = events[f'HS_right']
        left_toe = events[f'HS_left']
        right_toe = events[f'HS_right']
        
        # Compute step length as distance between the heel and toe markers
        # For this example, let's use a simple 2D distance (x and y)
        step_length = np.sqrt(
            (left_heel - left_toe)**2 + (right_heel - right_toe)**2
        )
        
        return step_length
