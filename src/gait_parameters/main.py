import os
import glob
import json
import pandas as pd
import logging

from gait_pipeline import GaitPipeline
from my_utils.helpers import save_csv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def get_project_root():
    """
    Returns the absolute path two levels above this file.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_external_folder(external_name, project_root, fallback_relative):
    """
    Check for an external folder (one level above the project) named external_name.
    If found, return its absolute path; otherwise, return the fallback path,
    which is relative to the project_root.
    """
    parent_dir = os.path.abspath(os.path.join(project_root, ".."))
    external_path = os.path.join(parent_dir, external_name)
    if os.path.isdir(external_path):
        logger.info("Using external %s folder: %s", external_name, external_path)
        return external_path
    else:
        fallback = os.path.join(project_root, fallback_relative)
        logger.info("External %s folder not found. Using internal folder: %s", external_name, fallback)
        return fallback


def main():
    # Get the project root (two levels above main.py)
    project_root = get_project_root()

    # Load config.json
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    config = load_config(config_path)

    # Resolve the data and output directories:
    data_dir = get_external_folder("Data", project_root, config["data_dir"])
    output_dir = get_external_folder("Output", project_root, config["output_dir"])

    # Update paths for output subdirectories from config using the resolved output_dir
    # (Note: individual CSVs will no longer be saved.)
    config["gait_parameters"]["save_path"] = os.path.join(output_dir, config["gait_parameters"]["save_path"])
    config["pose_estimator"]["tracked_csv_dir"] = os.path.join(output_dir, config["pose_estimator"]["tracked_csv_dir"])
    config["pose_estimator"]["tracked_video_dir"] = os.path.join(output_dir, config["pose_estimator"]["tracked_video_dir"])
    config["event_detection"]["plots_dir"] = os.path.join(output_dir, config["event_detection"]["plots_dir"])

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
    print(input_files)
    if not input_files:
        logger.error("No valid input files found in %s", data_dir)
        return

    logger.info("Found %d files to process in %s", len(input_files), data_dir)
    
    # List to store summary DataFrames (with medians only) for all videos
    all_summaries = []

    # Process each file
    for input_file in input_files:
        summary_df = process_single_file(input_file, config["gait_parameters"]["save_path"], config)
        if summary_df is not None:
            all_summaries.append(summary_df)
    
    # After processing all files, combine summaries and save master summary CSV.
    if all_summaries:
        master_summary = pd.concat(all_summaries, ignore_index=True)
        master_summary_csv_path = os.path.join(config["gait_parameters"]["save_path"], "all_gait_summary.csv")
        save_csv(master_summary, master_summary_csv_path)
        logger.info("Master summary (medians only) saved to %s", master_summary_csv_path)
    else:
        logger.info("No summaries were generated.")


def get_save_gait_parameters_path(input_file, output_dir):
    """
    Generate a valid file path for saving the gait parameters CSV file.
    (This function is no longer used since we are not saving individual CSV files.)
    """
    file_name = os.path.splitext(os.path.basename(input_file))[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info("Created output directory: %s", output_dir)

    output_path = os.path.join(output_dir, f"{file_name}_gait_parameters.csv")
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
    Process a single file using the GaitPipeline.
    This version no longer saves an individual detailed CSV file,
    and computes a summary containing only the median values.
    """
    # If the input file is a video, crop it first.
    if input_file.endswith((".mp4", ".MP4", ".mov", ".MOV")):
        from modules.yolo_cropper import YOLOCropper
        cropper = YOLOCropper(confidence_threshold=config.get("yolo_confidence_threshold", 0.5))
        base, ext = os.path.splitext(input_file)
        cropped_video_path = f"{base}_cropped{ext}"
        input_file, cropped_size = cropper.crop_video(
            input_video_path=input_file,
            output_video_path=cropped_video_path
        )
        # Update the configuration with cropped dimensions for accurate pose estimation.
        config['pose_estimator']['image_dimensions'] = cropped_size

    # Initialize the gait pipeline.
    # Since we no longer save individual CSVs, we can pass None (or simply ignore this parameter).
    pipeline = GaitPipeline(
        input_path=input_file,
        config=config,
        save_parameters_path=None
    )

    # Run the pipeline.
    pose_data = pipeline.load_input()
    if pose_data is None:
        logger.info("Skipping %s due to loading issues.", input_file)
        return None

    pipeline.preprocess()
    pipeline.detect_events()
    gait_parameters = pipeline.compute_gait_parameters()

    # Do NOT save the individual gait parameters CSV file.
    # Instead, compute a summary containing only the median values.
    video_name = os.path.splitext(os.path.basename(input_file))[0]
    median_summary = gait_parameters.median(numeric_only=True)
    median_summary["video_name"] = video_name
    # Convert the summary Series into a DataFrame with a single row.
    summary_df = pd.DataFrame(median_summary).T

    logger.info("Processed %s; median gait parameters computed.", input_file)
    return summary_df


if __name__ == "__main__":
    main()
