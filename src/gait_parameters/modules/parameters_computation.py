import os
import json
import numpy as np
import pandas as pd

from my_utils.gait_parameters import prepare_gait_dataframe
from my_utils.helpers import save_csv

class GaitParameters:
    @staticmethod
    def compute_parameters(events, save_path=None):
        """
        Computes gait parameters for left and right sides and returns a DataFrame with results.
        If a save_path is provided or determined from the configuration, the results are also saved as CSV.

        Args:
            events (dict or pd.DataFrame): Contains heel strike ('HS_side') and toe-off ('TO_side') events.
            save_path (str, optional): Path where the resulting CSV file should be saved.

        Returns:
            pd.DataFrame: DataFrame with gait parameters.
        """
        if isinstance(events, dict):
            events = pd.DataFrame(events)  # Convert dict to DataFrame if needed

        # If no save_path is provided, try to load the default from config.json
        if save_path is None:
            config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../config.json"))
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                default_dir = config.get("gait_parameters", {}).get("save_path", "../../output/gait_parameters")
                # Ensure the default directory exists.
                default_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), default_dir))
                os.makedirs(default_dir, exist_ok=True)
                # Use a default file name; you can customize this as needed.
                save_path = os.path.join(default_dir, "gait_parameters.csv")
            except Exception as e:
                # If loading the config fails, simply do not save.
                print(f"Warning: Could not load config for gait_parameters output path: {e}")
                save_path = None

        # Create empty DataFrame
        gait_df = prepare_gait_dataframe()

        for side in ['left', 'right']:
            other_side = 'left' if side == 'right' else 'right'

            # Compute required intermediate parameters:
            gait_df[(side, 'stride_duration')] = events[f'HS_{side}'].shift(-1) - events[f'HS_{side}']
            gait_df[(side, 'step_duration')] = events[f'HS_{side}'].shift(-1) - events[f'HS_{other_side}']

            # Top 5 parameters:
            gait_df[(side, 'cadence')] = 60 / gait_df[(side, 'step_duration')].replace(0, np.nan)
            gait_df[(side, 'step_length')] = GaitParameters.compute_step_length(events, side, other_side)
            gait_df[(side, 'stride_length')] = gait_df[(side, 'step_length')].shift(-1) + gait_df[(side, 'step_length')]
            gait_df[(side, 'gait_speed')] = gait_df[(side, 'stride_length')] / gait_df[(side, 'stride_duration')]
            gait_df[(side, 'swing')] = events[f'HS_{side}'].shift(-1) - events[f'TO_{side}']
            gait_df[(side, 'initial_double_support')] = events[f'TO_{other_side}'] - events[f'HS_{side}']

            # --- Other parameters computed but commented out for PD analysis ---
            # gait_df[(side, 'terminal_double_support')] = events[f'TO_{side}'] - events[f'HS_{side}'].shift(-1)
            # gait_df[(side, 'single_limb_support')] = events[f'HS_{other_side}'] - events[f'TO_{side}']
            # gait_df[(side, 'stance')] = events[f'TO_{side}'] - events[f'HS_{side}']

        # --- Asymmetry Calculations ---
        # For PD analysis, asymmetry in gait can be informative.
        # Here, we compute asymmetry for the selected top features.
        for feature in ['stride_duration', 'cadence', 'swing', 'initial_double_support']:
            gait_df[('asymmetry', feature)] = (
                abs(gait_df[('left', feature)] - gait_df[('right', feature)])
                / gait_df[[('left', feature), ('right', feature)]].max(axis=1) * 100
            )

        # Save CSV if a valid save_path is determined
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_csv(gait_df, save_path)

        return gait_df

    @staticmethod
    def compute_step_length(events, side, other_side):
        """
        Compute the step length for a given side. This requires spatial data for the markers.

        Args:
            events (dict or pd.DataFrame): Contains heel strike and toe-off events.
            side (str): 'left' or 'right'.
            other_side (str): The opposite side ('left' or 'right').
        
        Returns:
            pd.Series: Step lengths for the given side.
        """
        # NOTE: This implementation is a placeholder.
        # It assumes the events contain spatial data for the markers.
        # Replace this with your actual calculation if spatial marker data is available.
        left_heel = events.get(f'HS_left') if isinstance(events, dict) else events[f'HS_left']
        right_heel = events.get(f'HS_right') if isinstance(events, dict) else events[f'HS_right']
        left_toe = events.get(f'HS_left') if isinstance(events, dict) else events[f'HS_left']
        right_toe = events.get(f'HS_right') if isinstance(events, dict) else events[f'HS_right']

        step_length = np.sqrt(
            (left_heel - left_toe) ** 2 + (right_heel - right_toe) ** 2
        )

        return step_length
