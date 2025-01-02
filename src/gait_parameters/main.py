import argparse
import os
from gait_pipeline import GaitPipeline
import pandas as pd

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run Gait Analysis Pipeline")

    # Mutually exclusive group for input_path and input_dir
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('input_path', type=str, nargs='?', help="Path to the input file (CSV or video)")
    group.add_argument('--input_dir', type=str, help="Directory containing input files")

    parser.add_argument('--config', type=str, default='config.json', help="Path to the configuration JSON file")
    parser.add_argument('--output_dir', type=str, default='./gait_parameters', help="Directory to save the output CSV file(s)")
    return parser.parse_args()

def load_config(config_path):
    """
    Load the configuration file.

    Args:
        config_path (str): Path to the config file (JSON).

    Returns:
        dict: Parsed configuration.
    """
    import json
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config

def get_save_gait_parameters_path(args):
    """
    Save the gait parameters to a CSV file.

    Args:

    Returns: 
        output_path (str): Path to save the gait parameters CSV file.
    """
    file_name = os.path.splitext(os.path.basename(args.input_path))[0]
    base_folder = os.path.dirname(args.input_path)

    # Ensure output_dir is set and create the directory if it doesn't exist
    if not args.output_dir:
        args.output_dir = os.path.join(base_folder, "gait_parameters")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    # Define the path where gait parameters df is to be saved
    output_path = os.path.join(args.output_dir, f"{file_name}_gait_parameters.csv")
    return output_path

# TODO: take care of batch processing for a main dir. Current implementation is for a file.
def main():
    # Parse command-line arguments
    args = parse_args()

    # Load the configuration
    config = load_config(args.config)

    # Get the output path for gait parameters
    save_parameters_path = get_save_gait_parameters_path(args)

    # Initialize the GaitPipeline with the input path and config
    pipeline = GaitPipeline(input_path=args.input_path, 
                            config=config, 
                            save_parameters_path=save_parameters_path)

    # Load the input data (either video or CSV)
    pose_data = pipeline.load_input()
    print(f"Loaded input data from {args.input_path}")

    # Preprocess the pose data
    pose_data = pipeline.preprocess()
    print("Pose data preprocessed.")

    # Detect events (heel strikes and toe-offs)
    events = pipeline.detect_events()
    print("Detected gait events.")

    # Compute gait parameters based on events
    gait_parameters = pipeline.compute_gait_parameters()
    print("Gait parameters computed. Gait parameters saved to", save_parameters_path)

if __name__ == "__main__":
    main()
