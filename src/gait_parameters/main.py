import argparse
import os
import glob
import pandas as pd

from gait_pipeline import GaitPipeline
from utils.helpers import save_csv

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        Namespace: Command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run Gait Analysis Pipeline")

    # Mutually exclusive group for input_path and input_dir
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input_path', type=str, nargs='?', help="Path to the input file (CSV or video)")
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

def get_save_gait_parameters_path(input_file, output_dir='./gait_parameters'):
    """
    Save the gait parameters to a CSV file.

    Args:
actually saved the gait parameters"
    Returns: 
        output_path (str): Path to save the gait parameters CSV file.
    """
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    base_folder = os.path.dirname(input_file)

    # Ensure output_dir is set and create the directory if it doesn't exist
    if not output_dir:
        output_dir = os.path.join(base_folder, "gait_parameters")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Define the path where gait parameters df is to be saved
    output_path = os.path.join(output_dir, f"{file_name}_gait_parameters.csv")
    return output_path

def process_single_file(input_file, output_dir, config):
    """Process a single file with the GaitPipeline."""

    pipeline = GaitPipeline(input_path=input_file, 
                            config=config, 
                            save_parameters_path=save_parameters_path)

    pose_data = pipeline.load_input()
    if pose_data is None:
        print(f"Skipping {input_file} due to loading issues.")
        return

    pose_data = pipeline.preprocess()
    events = pipeline.detect_events()
    gait_parameters = pipeline.compute_gait_parameters()
    
    save_parameters_path = get_save_gait_parameters_path(input_file, output_dir)
    save_csv(gait_parameters, save_parameters_path)

    print(f"Processed {input_file}, gait parameters saved to {save_parameters_path}")



# TODO: take care of batch processing for a main dir. Current implementation is for a file.
def main():
    # Parse command-line arguments
    args = parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir

    if args.input_path:  # Process a single file
        process_single_file(args.input_path, args.output_dir, config)

    elif args.input_dir:  # Process all files in a directory
        # Search for CSV and video files in the given directory
        input_files = glob.glob(os.path.join(args.input_dir, "*.csv")) + \
                      glob.glob(os.path.join(args.input_dir, "*.mp4")) + \
                      glob.glob(os.path.join(args.input_dir, "*.mov"))

        if not input_files:
            print(f"No valid input files found in directory: {args.input_dir}")
            return

        print(f"Found {len(input_files)} files to process in {args.input_dir}")

        for input_file in input_files:
            process_single_file(input_file, output_dir, config)


if __name__ == "__main__":
    main()
