#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated poet_integration.py
Created on Mon Mar  3 21:58:39 2025
Author: Lange_L

This script runs the full PoET pipeline (tracking → preprocessing → kinematics → feature extraction)
on a single video file and returns tremor metrics.

Note: The preprocessing step now preserves the multi-index keypoint format 
(e.g. columns with (keypoint, coordinate) tuples) so that the downstream feature extraction code works as expected.
This version only supports the MultiIndex CSV format.
"""

import os
import logging
import numpy as np
import pandas as pd

# Import the robust FPS extraction helper from your gait module's helpers.
from my_utils.helpers import get_robust_fps

logger = logging.getLogger(__name__)

def run_poet_analysis(video_path, config):
    """
    Runs the PoET pipeline on a single video file and returns a DataFrame of tremor metrics.
    This version only supports CSV tracking data with a MultiIndex header.
    """
    # Determine output folder for tracking results
    output_folder = config.get("poet_output_folder", "./tracking/")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    # Construct the expected CSV path for tracking data.
    video_name = os.path.basename(video_path).split('.')[0]
    csv_path = os.path.join(output_folder, video_name + '_MPtracked.csv')
    
    # Check if the tracking CSV already exists and is valid.
    track_data = None
    if os.path.exists(csv_path):
        try:
            track_data = pd.read_csv(csv_path, header=[0, 1], index_col=0)
            if not track_data.empty:
                logger.info("Existing tracking CSV found for %s, skipping tracking.", video_path)
            else:
                logger.info("Tracking CSV exists for %s but is empty; will re-run tracking.", video_path)
                track_data = None
        except Exception as e:
            logger.warning("Error reading tracking CSV for %s: %s; will re-run tracking.", video_path, e)
            track_data = None

    # If tracking CSV is not available or invalid, run the tracking step.
    if track_data is None:
        logger.info("### ENTERING TRACKING")
        from .PoET.poet_tracking import load_models, track_video
        hands, pose = load_models(
            min_hand_detection_confidence=config.get("min_hand_detection_confidence", 0.5),
            min_tracking_confidence=config.get("min_tracking_confidence", 0.7)
        )
        logger.info("Tracking video: %s", video_path)
        track_video(
            video=video_path,
            pose=pose,
            hands=hands,
            output_folder=output_folder,
            make_csv=True,
            make_video=True,
            plot=True,
            world_coords=True
        )
    
    # After tracking (or if skipped), load the tracking CSV.
    logger.info("### ENTERING PREPROCESSING")
    try:
        track_data = pd.read_csv(csv_path, header=[0, 1], index_col=0)
        # Enforce that the CSV is in the expected MultiIndex format.
        if not isinstance(track_data.columns, pd.MultiIndex):
            logger.error("Tracking CSV for %s is not in the expected MultiIndex format. Please update your tracking data.", video_path)
            return None
    except Exception as e:
        logger.error("Failed to read tracking CSV for %s: %s", video_path, str(e))
        return None

    if track_data.empty:
        logger.error("Tracking failed or returned empty data for %s", video_path)
        return None

    # 2) Frame Rate Extraction: use the robust FPS helper.
    logger.info("### ENTERING FRAME RATE EXTRACTION")
    try:
        frame_rate = get_robust_fps(video_path, tolerance=0.1)
        logger.info("Extracted frame rate: %s FPS", frame_rate)
    except Exception as e:
        logger.warning("FPS extraction failed for %s: %s", video_path, str(e))
        frame_rate = config.get("frame_rate", 30.0)
        logger.info("Using fallback frame rate: %s FPS", frame_rate)

    # 3) Preprocessing: construct a PatientCollection using the tracking CSV.
    logger.info("### STARTING PREPROCESSING")
    from .PoET.poet_preprocessing import construct_data
    pc = construct_data(
        csv_files=[csv_path],
        fs=[frame_rate],
        labels=[None],
        scaling_factor=config.get("scaling_factor", 1),
        verbose=True
    )
    
    if pc is None:
        logger.error("Preprocessing failed for %s", video_path)
        return None
    logger.info("Preprocessing complete.")

    # 4) Kinematics: extract tremor-related signals.
    logger.info("### ENTERING KINEMATIC ANALYSIS")
    from .PoET.poet_kinematics import extract_tremor
    pc = extract_tremor(pc)
    if pc is None:
        logger.error("Kinematics extraction failed for %s", video_path)
        return None
    logger.info("Kinematic analysis complete.")

    # 5) Postprocessing: assign hand time periods.
    logger.info("### ENTERING POSTPROCESSING")
    from .PoET.poet_features import assign_hand_time_periods
    pc = assign_hand_time_periods(pc)
    logger.info("Postprocessing complete.")

    # 6) Feature extraction: compute tremor features for proximal arm, distal arm, and fingers.
    logger.info("### ENTERING FEATURE EXTRACTION")
    from .PoET.poet_features import (
        extract_proximal_arm_tremor_features,
        extract_distal_arm_tremor_features,
        extract_fingers_tremor_features
    )

    proximal_arm_features = extract_proximal_arm_tremor_features(pc, plot=False, save_plots=False)
    distal_arm_features = extract_distal_arm_tremor_features(pc, plot=False, save_plots=False)
    fingers_features = extract_fingers_tremor_features(pc, plot=False, save_plots=False)

    # Combine the features into a single DataFrame.
    tremor_features = pd.concat([proximal_arm_features, distal_arm_features, fingers_features], axis=1)

    # Optionally, include the detected frame rate in the output.
    if isinstance(tremor_features, dict):
        tremor_features['frame_rate'] = frame_rate
    elif isinstance(tremor_features, pd.DataFrame):
        tremor_features['frame_rate'] = frame_rate

    logger.info("### PIPELINE COMPLETE")
    return tremor_features

if __name__ == "__main__":
    # Example usage: Run analysis on a provided video file.
    config = {
        "poet_output_folder": "./tracking/",
        "min_hand_detection_confidence": 0.5,
        "min_tracking_confidence": 0.7,
        "scaling_factor": 1,
        "frame_rate": 30.0
    }
    video_path = "path/to/your/video.mp4"
    features = run_poet_analysis(video_path, config)
    if features is not None:
        print("Extracted features:")
        print(features)
    else:
        print("PoET analysis failed.")
