#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 21:58:39 2025

@author: Lange_L
"""

# modules/poet_integration.py
import logging
import numpy as np
import pandas as pd

# Import the PoET modules you copied over
from . import tracking
from . import preprocessing
from . import features

logger = logging.getLogger(__name__)

def run_poet_analysis(video_path, config):
    """
    Runs the PoET pipeline (tracking → preprocessing → features)
    on a single video file and returns a dictionary of tremor metrics.
    """

    # 1) Tracking: extract landmarks from each frame
    # (You can pass any tracking config you like here. E.g. MediaPipe vs. DeepLabCut.)
    track_data = tracking.run_tracking(
        video_path=video_path,
        method='mediapipe',   # or 'dlc' if you want to use a DLC model
        # Additional parameters if needed...
    )
    if track_data is None or len(track_data) == 0:
        logger.error("Tracking failed or returned empty data for %s", video_path)
        return None

    # 2) Preprocessing: filter & interpolate the raw landmark signals
    # The PoET code typically returns a dictionary or DataFrame of x,y coords over time.
    clean_data = preprocessing.preprocess_track_data(
        track_data,
        # e.g. specify the bandpass range, or fill config from your JSON
        low_freq=1.0,
        high_freq=10.0
    )
    if clean_data is None or len(clean_data) == 0:
        logger.error("Preprocessing failed or returned empty data for %s", video_path)
        return None

    # 3) Feature extraction: compute frequency, amplitude, etc.
    tremor_features = features.compute_tremor_features(
        clean_data,
        # e.g. pass frame_rate or other parameters from config
        frame_rate=30.0
    )
    if tremor_features is None:
        logger.error("Feature extraction failed for %s", video_path)
        return None

    return tremor_features
