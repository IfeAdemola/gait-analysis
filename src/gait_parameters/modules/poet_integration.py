#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 21:58:39 2025

@author: Lange_L
"""
# poet_integration.py
import os
import logging
import numpy as np
import pandas as pd

# Import the robust FPS extraction helper from your gait module's helpers.
from my_utils.helpers import get_robust_fps

logger = logging.getLogger(__name__)

def run_poet_analysis(video_path, config):
    """
    Runs the PoET pipeline (tracking → preprocessing → kinematics → feature extraction)
    on a single video file and returns a dictionary (or DataFrame) of tremor metrics.
    Mimics the full workflow of the original (dinosaur) PoET script.
    """
    # Determine output folder for tracking results
    output_folder = config.get("poet_output_folder", "./tracking/")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    # 1) Tracking: extract landmarks from each frame using track_video
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
        make_video=False,
        plot=False,
        world_coords=True
    )
    
    # After tracking, load the CSV with tracking data.
    video_name = os.path.basename(video_path).split('.')[0]
    csv_path = os.path.join(output_folder, video_name + '_MPtracked.csv')
    try:
        track_data = pd.read_csv(csv_path, header=[0, 1], index_col=0)
    except Exception as e:
        logger.error("Failed to read tracking CSV for %s: %s", video_path, str(e))
        return None

    if track_data is None or track_data.empty:
        logger.error("Tracking failed or returned empty data for %s", video_path)
        return None

    # 2) Frame Rate Extraction: use the robust FPS helper from the gait module.
    try:
        frame_rate = get_robust_fps(video_path, tolerance=0.1)
        logger.info("Extracted frame rate: %s FPS", frame_rate)
    except Exception as e:
        logger.warning("FPS extraction failed for %s: %s", video_path, str(e))
        frame_rate = config.get("frame_rate", 30.0)
        logger.info("Using fallback frame rate: %s FPS", frame_rate)

    # 3) Preprocessing: construct a PatientCollection using the tracking CSV.
    # Note: construct_data expects a list of CSV file paths, sampling frequency, labels, etc.
    from .PoET.poet_preprocessing import construct_data
    pc = construct_data(
        csv_files=[csv_path],
        fs=[frame_rate],
        labels=[None],
        scaling_factor=config.get("scaling_factor", 1),
        verbose=False
    )
    
    if pc is None:
        logger.error("Preprocessing failed for %s", video_path)
        return None

    # 4) Kinematics: extract tremor-related signals from the preprocessed data.
    from .PoET.poet_kinematics import extract_tremor
    pc = extract_tremor(pc)
    if pc is None:
        logger.error("Kinematics extraction failed for %s", video_path)
        return None

    # 5) Postprocessing: assign hand time periods.
    from .PoET.poet_features import assign_hand_time_periods, extract_tremor_features
    pc = assign_hand_time_periods(pc)
    
    # 6) Feature extraction: compute tremor features.
    tremor_features = extract_tremor_features(
        pc,
        tremor_type=config.get("tremor_type", "postural")
    )
    if tremor_features is None:
        logger.error("Feature extraction failed for %s", video_path)
        return None

    # 7) Optionally, include the detected frame_rate in the output.
    if isinstance(tremor_features, dict):
        tremor_features['frame_rate'] = frame_rate
    elif isinstance(tremor_features, pd.DataFrame):
        tremor_features['frame_rate'] = frame_rate

    return tremor_features
