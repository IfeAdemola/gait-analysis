import pandas as pd
import logging

from modules.pose_estimation import PoseEstimator
from modules.preprocessing import Preprocessor
from modules.gait_event_detection import EventDetector
from modules.gait_parameters_computation import GaitParameters


from my_utils.helpers import load_csv, get_frame_rate

class GaitPipeline:
    def __init__(self, input_path, config, save_parameters_path):
        self.input_path = input_path
        self.config = config
        self.save_parameters_path = save_parameters_path
        self.pose_data = None
        self.frame_rate = None
        self.events = None
        self.gait_params = None
        self.fog_events = None  # Freezing of Gait events
        
        # Initialize a logger for this class
        self.logger = logging.getLogger(__name__)
        # Optionally, set the logging level and add handlers if needed
        self.logger.setLevel(logging.DEBUG)
        if not self.logger.hasHandlers():
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

    
    def load_input(self):
        if self.input_path.endswith(".csv"):
            # Load a CSV directly.
            self.pose_data = load_csv(file_path=self.input_path)
            self.frame_rate = get_frame_rate(file_path=self.input_path) 
        elif self.input_path.endswith((".mp4", ".MP4", ".mov", ".MOV")):
            # Pass self.config so PoseEstimator uses the same folder settings.
            pose_estimator = PoseEstimator(config=self.config)
            self.pose_data, self.frame_rate = pose_estimator.process_video(video_path=self.input_path)
            if self.pose_data is not None:
                self.pose_data = self.pose_data.apply(pd.to_numeric, errors='coerce')
        else:
            raise ValueError("Unsupported input format. Use .mp4/.mov for videos or .csv for spreadsheets.")
        return self.pose_data

    def preprocess(self):
        preprocessor = Preprocessor(pose_data=self.pose_data)
        self.pose_data = preprocessor.preprocess(window_size=self.config['preprocessing']['median_filter_window'])
        return self.pose_data

    def detect_events(self):
        detector = EventDetector(**self.config['event_detection'], input_path=self.input_path, frame_rate=self.frame_rate)
        self.events = detector.detect_heel_toe_events(self.pose_data)
        return self.events

    def compute_gait_parameters(self):
        gait_params = GaitParameters()
        # Compute all gait parameters (including step lengths) as usual.
        self.gait_params = gait_params.compute_parameters(
            self.events, self.pose_data, self.frame_rate, save_path=self.save_parameters_path
        )
        # Now, explicitly recompute step lengths with the correct pairing:
        left_step_length = GaitParameters.compute_step_length(
            self.events, self.pose_data, self.frame_rate, side="left", other_side="right"
        )
        right_step_length = GaitParameters.compute_step_length(
            self.events, self.pose_data, self.frame_rate, side="right", other_side="left"
        )
        # Override the step_length columns with these values.
        self.gait_params[("left", "step_length")] = left_step_length
        self.gait_params[("right", "step_length")] = right_step_length
    
        # Log a few values to verify correct computation.
        self.logger.debug("Recomputed left step length (first few values): %s", left_step_length.head())
        self.logger.debug("Recomputed right step length (first few values): %s", right_step_length.head())
        return self.gait_params
    


    def compute_forward_displacement(self):
        """
        Computes the forward displacement signal based on the toe markers relative to the sacrum.
        Assumes that the pose_data DataFrame has a MultiIndex with markers like:
        ('left_foot_index', 'z'), ('right_foot_index', 'z'), and ('sacrum', 'z').
        """
        try:
            left_toe = self.pose_data[('left_foot_index', 'z')]
            right_toe = self.pose_data[('right_foot_index', 'z')]
            sacrum = self.pose_data[('sacrum', 'z')]
        except KeyError as e:
            raise ValueError("Required markers not found in pose data. Ensure pose_data includes 'left_foot_index', 'right_foot_index', and 'sacrum'.") from e

        left_forward = left_toe - sacrum
        right_forward = right_toe - sacrum
        # Average the forward displacements from both sides.
        forward_disp = (left_forward + right_forward) / 2.0
        return forward_disp.to_numpy()

    def detect_freezes(self):
        """
        Uses the FreezingDetector module to detect FoG events.
        It computes the forward displacement signal, then runs the freezing detector,
        storing the detected freeze events in self.fog_events.
        """
        from modules.freezing_detector import FreezingDetector  # Import here to avoid circular dependencies
        forward_disp = self.compute_forward_displacement()
        
        # Retrieve freezing detection parameters from config (with defaults)
        freezing_config = self.config.get('freezing', {})
        velocity_threshold = freezing_config.get('velocity_threshold', 0.05)
        fi_threshold = freezing_config.get('fi_threshold', 2.0)
        window_size_sec = freezing_config.get('window_size_sec', 2.0)
        step_size_sec = freezing_config.get('step_size_sec', 0.5)
        
        fd = FreezingDetector(
            frame_rate=self.frame_rate,
            window_size_sec=window_size_sec,
            step_size_sec=step_size_sec,
            velocity_threshold=velocity_threshold,
            fi_threshold=fi_threshold
        )
        self.fog_events = fd.detect_freezes(forward_disp)
        return self.fog_events
