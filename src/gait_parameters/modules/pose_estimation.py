import numpy as np
import skvideo.io
import mediapipe as mp
import os
import glob
import json
import logging

import config 

from pathlib import Path
from tqdm import tqdm
from typing import Optional, Any, Tuple, Path

from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

from my_utils.mediapipe_landmarks import prepare_empty_dataframe
from my_utils.helpers import set_ffmpeg_path, get_output_dir, get_robust_fps  # Updated import for robust FPS extraction


class PoseEstimator:
    def __init__(self, make_video: bool = True, make_csv: bool = True, plot: bool = False, tracked_csv_dir: Optional[str] = None, tracked_video_dir: Optional[str] = None):
        
        self.make_video = make_video
        self.make_csv = make_csv
        self.plot = plot
        
        # Set default paths if none are provided
        self.tracked_csv_dir = get_output_dir(tracked_csv_dir, config.PROJECT_ROOT / "output" / "tracked_csv")
        self.tracked_video_dir = get_output_dir(tracked_video_dir, config.PROJECT_ROOT / "output" / "tracked_videos")

        self.hand_model_path = config.MAIN_ROOT / "models" / "hand_landmarker.task"
        self.pose_model_path = config.MAIN_ROOT / "models" / "pose_landmarker_heavy.task"

        self.logger = self._setup_logger()

        self.initialize_mediapipe_models()

    @staticmethod
    def _setup_logger() -> logging.Logger:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        return logging.getLogger("PoseEstimator")
    
    def initialize_mediapipe_models(self):
        self.hands = self._load_hand_model()
        self.pose = self._load_pose_model()
        self.logger.debug("MediaPipe models have been initialized.")
    
    def draw_pose_landmarks_on_image(self, image: np.ndarray, detection_result: Any) -> np.ndarray:
        annotated_image = np.copy(image)
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in detection_result.pose_landmarks[0]
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )
        return annotated_image
    
    def draw_hand_landmarks_on_image(self, rgb_image: np.ndarray, detection_result: Any) -> np.ndarray:
        hand_landmarks_list = detection_result.hand_landmarks
        annotated_image = np.copy(rgb_image)
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style()
            )
        return annotated_image
    
    def draw_face_landmarks_on_image(self, rgb_image: np.ndarray, detection_result: Any) -> np.ndarray:
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                for landmark in face_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
            )
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
            )
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style()
            )
        return annotated_image
    
    def _load_pose_model(self) -> Any:
        base_options = mp.tasks.BaseOptions(model_asset_path=self.pose_model_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO
        )
        return mp.tasks.vision.PoseLandmarker.create_from_options(options)
    
    def _load_hand_model(self) -> Any:
        base_options = mp.tasks.BaseOptions(model_asset_path=self.hand_model_path)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            running_mode=mp.tasks.vision.RunningMode.VIDEO
        )
        return mp.tasks.vision.HandLandmarker.create_from_options(options)
    
    def process_video(self, video_path: str) -> Optional[Any]:
        """
        Processes the given video and stores results in the appropriate directories.

        Args:
            video_path (str): Path to the input video file.

        Returns:
            Optional[Any]: Processed video output (implementation-specific).
        """
        set_ffmpeg_path()

        tracked_csv_path, tracked_video_path = self.prepare_file_paths(video_path)
        
        # TODO: I don't see the necessity of this
        # If tracked CSV already exists, load it and return the data.
        if self.make_csv and os.path.isfile(tracked_csv_path):
            self.logger.info(f"CSV already exists for {video_path}. Loading tracked data.")
            import pandas as pd  # Import here if not already imported at the top
            try:
                marker_df = pd.read_csv(tracked_csv_path, header=[0,1])
            except Exception as e:
                self.logger.error(f"Error loading CSV with multi-index: {e}. Loading without multi-index.")
                marker_df = pd.read_csv(tracked_csv_path)
            fs = self.get_fps_from_metadata(tracked_csv_path)
            return marker_df, fs

        if self.make_video and os.path.isfile(tracked_video_path):
            self.logger.info(f"Tracked Video already exists for {video_path}. (Using CSV if available)")

        videogen = list(skvideo.io.vreader(video_path))
        fs = get_robust_fps(video_path)
        self.logger.info(f"Video loaded. Frame rate: {fs} fps.")
        writer = skvideo.io.FFmpegWriter(
                    tracked_video_path, 
                    outputdict={
                        "-r": str(fs), 
                        "-vcodec": "libx264", 
                        "-acodec": "aac",      
                        "-strict": "-2",       
                        "-pix_fmt": "yuv420p"  
                    }
                ) if self.make_video else None

        marker_df, marker_mapping = prepare_empty_dataframe(hands='both', pose=True)
        # Log the marker mapping and verify that the expected keys are present.
        # TODO: Remove after all updating
        self.logger.debug("Initial empty pose DataFrame columns: {}".format(marker_df.columns))
        self.logger.debug("Marker mapping: {}".format(marker_mapping))
        if "left_foot_index" not in marker_mapping:
            self.logger.error("Marker mapping does not contain key 'left_foot_index'")
        if "right_foot_index" not in marker_mapping:
            self.logger.error("Marker mapping does not contain key 'right_foot_index'")

        for i, image in enumerate(tqdm(videogen, desc=f"Processing {os.path.basename(video_path)}", total=len(videogen))):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            frame_ms = int(fs * i)

            results_hands = self.hands.detect_for_video(mp_image, frame_ms)
            results_pose = self.pose.detect_for_video(mp_image, frame_ms)

            annotated_image = np.copy(image)

            if results_pose.pose_world_landmarks:
                annotated_image = self.draw_pose_landmarks_on_image(annotated_image, results_pose)
                for l, landmark in enumerate(results_pose.pose_world_landmarks[0]):
                    marker = marker_mapping['pose'][l]
                    marker_df.loc[i, (marker, 'x')] = landmark.x
                    marker_df.loc[i, (marker, 'y')] = landmark.y
                    marker_df.loc[i, (marker, 'z')] = landmark.z
                    marker_df.loc[i, (marker, 'visibility')] = landmark.visibility
                    marker_df.loc[i, (marker, 'presence')] = landmark.presence

            if results_hands.hand_landmarks:
                annotated_image = self.draw_hand_landmarks_on_image(annotated_image, results_hands)
                for h, hand in enumerate(results_hands.hand_world_landmarks):
                    handedness = results_hands.handedness[h][0].display_name
                    handedness = 'Right' if handedness == 'Left' else 'Left'
                    for l, landmark in enumerate(hand):
                        marker = marker_mapping[f"{handedness}_hand"][l]
                        marker_df.loc[i, (marker, 'x')] = landmark.x
                        marker_df.loc[i, (marker, 'y')] = landmark.y
                        marker_df.loc[i, (marker, 'z')] = landmark.z
                        marker_df.loc[i, (marker, 'visibility')] = landmark.visibility
                        marker_df.loc[i, (marker, 'presence')] = landmark.presence

            if self.make_video:
                writer.writeFrame(annotated_image)

        if self.make_csv:
            marker_df.to_csv(tracked_csv_path, index=False)
            self.logger.info(f"Saved pose estimation CSV to {tracked_csv_path}")
            metadata_path = tracked_csv_path.with_name(tracked_csv_path.stem + "_metadata.json")
            metadata_json = {"fps": fs}
            with metadata_path.open("w") as f:  # Open the JSON file for writing
                json.dump(metadata_json, f)

        if self.make_video:
            writer.close()
            self.logger.info(f"Saved annotated video to {tracked_video_path}")

        return marker_df, fs
    
    def batch_video_processing(self, input_directory) -> Any:
        pass
    
    def get_fps_from_metadata(self, tracked_csv_path: str, default_fps: int = 25) -> int:
        """
        Extracts FPS from metadata JSON file. If the file is missing or invalid, returns a default FPS.
        """
        metadata_path = Path(tracked_csv_path).with_name(Path(tracked_csv_path).stem + "_metadata.json")  # ✅ Fix applied

        if metadata_path.is_file():
            try:
                with metadata_path.open("r") as f:
                    metadata = json.load(f)
                return int(metadata.get("fps", default_fps))  # Ensure FPS is an integer
            except (json.JSONDecodeError, ValueError):
                self.logger.warning(f"Warning: Metadata file {metadata_path} is corrupted. Using default FPS: {default_fps}")

        return default_fps  # If file does not exist or is invalid, return default FPS

    def prepare_file_paths(self, video_path: str) -> Tuple[Path, Path]:
        """
        Prepares file paths for tracked CSV and tracked video, ensuring directories exist.

        Args:
            video_path (str): Path to the input video file.

        Returns:
            Tuple[Path, Path]: Paths for the tracked CSV file and tracked video file.
        """
        file_name = Path(video_path).stem  # Extract filename without extension

        tracked_csv_path = self.tracked_csv_dir / f"{file_name}_MPtracked.csv"
        tracked_video_path = self.tracked_video_dir / f"{file_name}_MPtracked.mp4"

        return tracked_csv_path, tracked_video_path
    