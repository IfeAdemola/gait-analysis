import pandas as pd

from modules.pose_estimation import PoseEstimator
from modules.preprocessing import Preprocessor
from modules.event_detection import EventDetector
from modules.parameters_computation import GaitParameters

from utils.helpers import load_csv

class GaitPipeline:
    def __init__(self, input_path, config, save_parameters_path):
        self.input_path = input_path
        self.config = config
        self.save_parameters_path = save_parameters_path
        self.pose_data = None
        self.events = None
        self.gait_params = None
    
    def load_input(self):
        if self.input_path.endswith(".csv"):
            # Load a CSV directly.
            self.pose_data = load_csv(file_path=self.input_path)

        elif self.input_path.endswith((".mp4", ".MP4", ".mov", ".MOV")):
            # Pass self.config so PoseEstimator uses the same folder settings.
            pose_estimator = PoseEstimator(config=self.config)
            self.pose_data = pose_estimator.process_video(video_path=self.input_path)

            # If PoseEstimator returned a DataFrame, ensure columns are numeric.
            if self.pose_data is not None:
                self.pose_data = self.pose_data.apply(pd.to_numeric, errors='coerce')

        else:
            raise ValueError(
                "Unsupported input format. Use .mp4/.mov for videos or .csv for spreadsheets."
            )

        return self.pose_data

    def preprocess(self):
        preprocessor = Preprocessor(pose_data=self.pose_data)
        # Example: use the median_filter_window from your config
        self.pose_data = preprocessor.preprocess(
            window_size=self.config['preprocessing']['median_filter_window']
        )
        return self.pose_data

    def detect_events(self):
        # Pass the event_detection settings via **kwargs
        detector = EventDetector(**self.config['event_detection'])
        self.events = detector.detect_heel_toe_events(self.pose_data)
        return self.events

    def compute_gait_parameters(self):
        gait_params = GaitParameters()
        # Pass save_parameters_path to store final CSV
        self.gait_params = gait_params.compute_parameters(
            self.events, save_path=self.save_parameters_path
        )
        return self.gait_params
