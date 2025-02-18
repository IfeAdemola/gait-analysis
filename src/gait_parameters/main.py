import os
import glob
import json
import pandas as pd
import logging

from gait_pipeline import GaitPipeline
from utils.helpers import save_csv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def get_project_root():
    """
    Returns the absolute path two levels above this file.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    # Get the project root (two levels above main.py)
    project_root = get_project_root()

    # Define absolute paths relative to project_root
    data_dir = os.path.join(project_root, "data")
    output_dir = os.path.join(project_root, "output")

    # Load config.json (if you still want to keep other settings in it)
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = load_config(config_path)

    # Override folder paths in config with our absolute paths:
    config["data_dir"] = data_dir
    config["gait_parameters"]["save_path"] = os.path.join(output_dir, "gait_parameters")
    config["pose_estimator"]["tracked_csv_dir"] = os.path.join(output_dir, "tracked_csv")
    config["pose_estimator"]["tracked_video_dir"] = os.path.join(output_dir, "tracked_videos")
    config["event_detection"]["plots_dir"] = os.path.join(output_dir, "plots")

    # Ensure that the needed directories exist
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(config["gait_parameters"]["save_path"], exist_ok=True)
    os.makedirs(config["pose_estimator"]["tracked_csv_dir"], exist_ok=True)
    os.makedirs(config["pose_estimator"]["tracked_video_dir"], exist_ok=True)
    os.makedirs(config["event_detection"]["plots_dir"], exist_ok=True)

    # Gather input files from data_dir (CSV, MP4, MOV, etc.)
    input_files = []
    for ext in ("*.csv", "*.mp4", "*.MP4", "*.mov", "*.MOV"):
        input_files.extend(glob.glob(os.path.join(data_dir, ext)))

    if not input_files:
        logger.error("No valid input files found in %s", data_dir)
        return

    logger.info("Found %d files to process in %s", len(input_files), data_dir)

    # Process each file
    for input_file in input_files:
        process_single_file(input_file, config["gait_parameters"]["save_path"], config)


def get_save_gait_parameters_path(input_file, output_dir):
    """
    Generate a valid file path for saving the gait parameters CSV file.
    """
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info("Created output directory: %s", output_dir)

    output_path = os.path.join(output_dir, f"{file_name}_gait_parameters.csv")

    # If the output path exists and is a directory, handle the conflict.
    if os.path.exists(output_path) and os.path.isdir(output_path):
        logger.warning("Output path %s is a directory, expected a file path.", output_path)
        try:
            if not os.listdir(output_path):
                os.rmdir(output_path)
                logger.info("Removed empty directory at %s", output_path)
            else:
                base, ext = os.path.splitext(output_path)
                i = 1
                new_output_path = f"{base}_{i}{ext}"
                while os.path.exists(new_output_path):
                    i += 1
                    new_output_path = f"{base}_{i}{ext}"
                logger.info("Renaming output file path to avoid conflict: %s", new_output_path)
                output_path = new_output_path
        except Exception as e:
            logger.exception("Error handling existing directory at %s: %s", output_path, e)
            raise

    return output_path


def process_single_file(input_file, output_dir, config):
    """
    Process a single file using GaitPipeline and save the resulting CSV.
    """
    save_parameters_path = get_save_gait_parameters_path(input_file, output_dir)

    # Extra safeguard: remove directory if it still exists at save_parameters_path
    if os.path.exists(save_parameters_path) and os.path.isdir(save_parameters_path):
        logger.warning("Final save path %s is still a directory. Removing it.", save_parameters_path)
        os.rmdir(save_parameters_path)

    pipeline = GaitPipeline(
        input_path=input_file,
        config=config,
        save_parameters_path=save_parameters_path
    )

    # Run the pipeline
    pose_data = pipeline.load_input()
    if pose_data is None:
        logger.error("Skipping %s due to loading issues.", input_file)
        return

    pipeline.preprocess()
    pipeline.detect_events()
    gait_parameters = pipeline.compute_gait_parameters()

    # Save computed gait parameters
    save_csv(gait_parameters, save_parameters_path)
    logger.info("Processed %s, gait parameters saved to %s", input_file, save_parameters_path)


if __name__ == "__main__":
    main()
