#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 21:47:33 2025

@author: Lange_L
"""

# modules/postural_tremor_analysis.py

import cv2
import numpy as np
import mediapipe as mp
import logging
import pandas as pd
from scipy.signal import butter, filtfilt, welch

logger = logging.getLogger(__name__)

class PosturalTremorAnalyzer:
    def __init__(self, config):
        self.config = config
        # Setup MediaPipe Hands for pose tracking.
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def analyze_video(self, video_path):
        """
        Analyzes a video file to compute postural tremor features.
        
        Returns:
            A dictionary with tremor features including dominant frequency and amplitude.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("Error opening video file: %s", video_path)
            return None

        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        time_series = []  # To store (x, y) coordinates from the index finger tip.
        timestamps = []

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert frame to RGB as required by MediaPipe.
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image)
            if results.multi_hand_landmarks:
                # For simplicity, choose the first detected hand and its index finger tip.
                hand_landmarks = results.multi_hand_landmarks[0]
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                time_series.append((index_tip.x, index_tip.y))
            else:
                # Append NaNs for frames where the hand is not detected.
                time_series.append((np.nan, np.nan))
            timestamps.append(frame_idx / frame_rate)
            frame_idx += 1

        cap.release()
        time_series = np.array(time_series)  # Shape: (n_frames, 2)
        timestamps = np.array(timestamps)

        # Interpolate missing values for both x and y.
        for i in range(2):
            col = time_series[:, i]
            valid = ~np.isnan(col)
            if valid.sum() < 2:
                continue
            time_series[:, i] = np.interp(timestamps, timestamps[valid], col[valid])
        
        # For tremor analysis, we can use one dimension (e.g., x-axis) as the tremor signal.
        signal = time_series[:, 0]

        # Bandpass filter parameters (e.g., 1-10 Hz for postural tremor).
        lowcut = 1.0
        highcut = 10.0
        nyq = 0.5 * frame_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(2, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal)

        # Compute power spectral density using Welch's method.
        freqs, psd = welch(filtered_signal, fs=frame_rate, nperseg=256)
        # Focus on the frequency band of interest.
        mask = (freqs >= lowcut) & (freqs <= highcut)
        if np.sum(mask) == 0:
            logger.error("No frequency components found in the expected tremor band.")
            dominant_freq = float('nan')
            tremor_amplitude = float('nan')
        else:
            idx = np.argmax(psd[mask])
            dominant_freq = freqs[mask][idx]
            # Tremor amplitude is taken as the square root of power at the dominant frequency.
            tremor_amplitude = np.sqrt(psd[mask][idx])

        features = {
            'frame_rate': frame_rate,
            'dominant_tremor_frequency': dominant_freq,
            'tremor_amplitude': tremor_amplitude,
            'n_frames': len(time_series)
        }
        return features
