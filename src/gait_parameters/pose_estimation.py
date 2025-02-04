import numpy as np
import skvideo
skvideo.setFFmpegPath('c:/Users/ifeol/ffmpeg-master-latest-win64-gpl/bin')
import skvideo.io
import mediapipe as mp
from typing import Optional, Any, Tuple

import os
import glob

import logging

from tqdm import tqdm

from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

from mediapipe_landmarks import prepare_empty_dataframe


class PoseEstimator:
    def __init__(self, make_video: bool = True, make_csv: bool = True, plot: bool = False):
        """
        Initialize the PoseEstimator with required paths and configurations.

        Args:
            make_video (bool): Whether to generate annotated videos. Default is False.
            make_csv (bool): Whether to generate CSVs of landmarks. Default is True.
            plot (bool): Whether to visualize the results (not implemented). Default is False.
        """
        self.make_video = make_video
        self.make_csv = make_csv
        self.plot = plot
        self.hand_model_path = './models/hand_landmarker.task'
        self.pose_model_path = './models/pose_landmarker_heavy.task'
        self.logger = self._setup_logger()

        # Initialize MediaPipe models
        self.initialize_mediapipe_models()

    @staticmethod
    def _setup_logger() -> logging.Logger:
        """
        Set up the logger for logging messages.

        Returns:
            logging.Logger: Configured logger instance.
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
        )
        return logging.getLogger("PoseEstimator")
    
    def initialize_mediapipe_models(self):
        """
        (Re)Initializes MediaPipe models for hands and pose detection.
        """
        self.hands = self._load_hand_model()
        self.pose = self._load_pose_model()
        self.logger.debug("MediaPipe models have been initialized.")
    
    def draw_pose_landmarks_on_image(self, image: np.ndarray, detection_result: Any) -> np.ndarray:
        """
        Annotates the image with pose landmarks.

        Args:
            image (np.ndarray): The input image to be annotated.
            detection_result (Any): The result of pose detection containing landmarks.

        Returns:
            np.ndarray: The annotated image with pose landmarks.
        """
        annotated_image = np.copy(image)
        
        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in detection_result.pose_landmarks[0]
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style()
        )
        
        return annotated_image
    
    def draw_hand_landmarks_on_image(self, rgb_image: np.ndarray, detection_result: Any) -> np.ndarray:
        """
        Annotates the image with hand landmarks.

        Args:
            rgb_image (np.ndarray): The input RGB image to be annotated.
            detection_result (Any): The result of hand detection containing landmarks.

        Returns:
            np.ndarray: The annotated image with hand landmarks.
        """
        hand_landmarks_list = detection_result.hand_landmarks
        annotated_image = np.copy(rgb_image)
        
        # Loop through the detected poses to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            
            # Draw the pose landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style())
            
        return annotated_image
    
    def draw_face_landmarks_on_image(self, rgb_image: np.ndarray, detection_result: Any) -> np.ndarray:
        """
        Annotates the image with face landmarks.

        Args:
            rgb_image (np.ndarray): The input RGB image to be annotated.
            detection_result (Any): The result of face detection containing landmarks.

        Returns:
            np.ndarray: The annotated image with face landmarks.
        """
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)
        
        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]
            
            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])
            
            solutions.drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks_proto,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_tesselation_style())
            
            solutions.drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks_proto,
                    connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_contours_style())
            
            solutions.drawing_utils.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks_proto,
                    connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_iris_connections_style())
        
        return annotated_image
    
    def _load_pose_model(self) -> Any:
        """
        Load the pose estimation model.

        Returns:
            Any: Mediapipe pose landmarker object.
        """
        base_options = mp.tasks.BaseOptions(model_asset_path=self.pose_model_path)
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO
        )
        return mp.tasks.vision.PoseLandmarker.create_from_options(options)
    
    def _load_hand_model(self) -> Any:
        """
        Load the hand landmark detection model.

        Returns:
            Any: Mediapipe hand landmarker object.
        """
        base_options = mp.tasks.BaseOptions(model_asset_path=self.hand_model_path)
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            running_mode=mp.tasks.vision.RunningMode.VIDEO
        )
        return mp.tasks.vision.HandLandmarker.create_from_options(options)
    
    def process_video(self, video_path:str, tracked_csv_dir:Optional[str]=None, tracked_video_dir: Optional[str] = None) -> Optional[Any]:
        """
        Processes a single video file, extracting landmarks, saving annotated video and/or CSV.

        Args:
            video_path (str): Path to the input video file.
            csv_path (str, optional): Path to save the output CSV. Default is None.
            video_output_path (str, optional): Path to save the annotated video. Default is None.

        Returns:
            Optional[Any]: DataFrame containing landmark pose data if CSV generation is enabled, otherwise None. 
                            Video with tracking if video generation is enabled.
        """
        self.initialize_mediapipe_models()

        tracked_csv_path, tracked_video_path = self.prepare_file_paths(video_path, tracked_csv_dir, tracked_video_dir)

        # Check if the CSV already exists to skip processing
        if self.make_csv and os.path.isfile(tracked_csv_path):
            self.logger.info(f"CSV already exists for {video_path}. Skipping.")
            return
        if self.make_video and os.path.isfile(tracked_video_path):
            self.logger.info(f"Tracked Video already exists for {video_path}. Skipping.")
            return

        videogen = list(skvideo.io.vreader(video_path))
        metadata = skvideo.io.ffprobe(video_path)
        fs = int(metadata['video']['@r_frame_rate'].split('/')[0])
        writer = skvideo.io.FFmpegWriter(tracked_video_path) if self.make_video else None

        marker_df, marker_mapping = prepare_empty_dataframe(hands='both', pose=True)

        for i, image in enumerate(tqdm(videogen, desc=f"Processing {os.path.basename(video_path)}", total=len(videogen))):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            frame_ms = fs * i

            # Run Mediapipe models
            results_hands = self.hands.detect_for_video(mp_image, frame_ms)
            results_pose = self.pose.detect_for_video(mp_image, frame_ms)

            annotated_image = np.copy(image)

            # Annotate and extract pose landmarks
            if results_pose.pose_world_landmarks:
                annotated_image = self.draw_pose_landmarks_on_image(annotated_image, results_pose)
                for l, landmark in enumerate(results_pose.pose_world_landmarks[0]):
                    marker = marker_mapping['pose'][l]
                    marker_df.loc[i, (marker, 'x')] = landmark.x
                    marker_df.loc[i, (marker, 'y')] = landmark.y
                    marker_df.loc[i, (marker, 'z')] = landmark.z
                    marker_df.loc[i, (marker, 'visibility')] = landmark.visibility
                    marker_df.loc[i, (marker, 'presence')] = landmark.presence

            # Annotate and extract hand landmarks
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

            # Write frame to video
            if self.make_video:
                writer.writeFrame(annotated_image)

        # Save the CSV
        if self.make_csv:
            marker_df.to_csv(tracked_csv_path, index=False)
            self.logger.info(f"Saved pose estimation CSV to {tracked_csv_path}")


        if self.make_video:
            writer.close()
            self.logger.info(f"Saved annotated video to {tracked_video_path}")

        return marker_df
    
    def batch_video_processing(self, input_directory) -> Any:
        """
        Process all video files in the input directory, including those in subdirectories.

        Returns:
            None
        """
        # Get all subdirectories in the input directory
        folders = [x[0] for x in os.walk(input_directory)][1:]
        folders = [f for f in folders if 'tracked' not in f]

        if not folders:
            self.logger.warning(f"No subdirectories found in {input_directory}.")
            return

        self.logger.info(f"Found {len(folders)} folders to process.")

        for folder in folders:
            # Create output subfolders for processed videos and CSVs
            tracked_video_folder = os.path.join(folder, 'tracked_videos')
            csv_folder = os.path.join(folder, 'tracked_csv')

            if not os.path.exists(tracked_video_folder):
                os.makedirs(tracked_video_folder)
                self.logger.info(f"Created directory: {tracked_video_folder}")
            else:
                self.logger.debug(f"Tracked video folder already exists: {tracked_video_folder}")

            if not os.path.exists(csv_folder):
                os.makedirs(csv_folder)
                self.logger.info(f"Created directory: {csv_folder}")
            else:
                self.logger.debug(f"CSV folder already exists: {csv_folder}")

            # Find video files in the current folder
            video_files = glob.glob(os.path.join(folder, "*.mp4")) + glob.glob(os.path.join(folder, "*.mov"))
            if not video_files:
                self.logger.warning(f"No video files found in {folder}")
                continue

            self.logger.info(f"Found {len(video_files)} video files in {folder} to process.")

            for video_file in video_files:
                self.process_video(video_file, csv_folder, tracked_video_folder) 

    @staticmethod
    def prepare_file_paths(video_path: str, csv_dir: Optional[str], video_dir: Optional[str]) -> Tuple[str, str]:
        """
        Prepares file paths for tracked CSV and tracked video, ensuring directories exist.

        Args:
            video_path (str): Path to the input video file.
            csv_dir (str, optional): Directory for saving the CSV file. Default is None.
            video_dir (str, optional): Directory for saving the video file. Default is None.

        Returns:
            Tuple[str, str]: Paths for the tracked CSV file and tracked video file.
        """
        file_name = os.path.splitext(os.path.basename(video_path))[0]
        base_folder = os.path.dirname(video_path)

        # Prepare the tracked CSV directory and path
        if csv_dir is None:
            csv_dir = os.path.join(base_folder, "tracked_csv")
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir, exist_ok=True)
        tracked_csv_path = os.path.join(csv_dir, f"{file_name}_MPtracked.csv")

        # Prepare the tracked video directory and path
        if video_dir is None:
            video_dir = os.path.join(base_folder, "tracked_videos")
        if not os.path.exists(video_dir):
            os.makedirs(video_dir, exist_ok=True)
        tracked_video_path = os.path.join(video_dir, f"{file_name}_MPtracked.mp4")
        
        return tracked_csv_path, tracked_video_path


# def archive():
    # # Handle `tracked_csv_dir`
        # if tracked_csv_dir is None:
        #     # Default to creating `tracked_csv` folder in the base folder
        #     tracked_csv_dir = os.path.join(base_folder, "tracked_csv")
        #     if not os.path.exists(tracked_csv_dir):
        #         os.makedirs(tracked_csv_dir)
        #         self.logger.info(f"Created tracked CSV directory: {tracked_csv_dir}")
        #     tracked_csv_path = os.path.join(tracked_csv_dir, f"{file_name}_MPtracked.csv")
        # elif os.path.isdir(tracked_csv_dir):
        #     if not os.path.exists(tracked_csv_dir):
        #         os.makedirs(tracked_csv_dir)
        #         self.logger.info(f"Created tracked CSV directory: {tracked_csv_dir}")
        #     # If a directory is provided, append the file name
        #     tracked_csv_path = os.path.join(tracked_csv_dir, f"{file_name}_MPtracked.csv")
        # else:
        #     # If `tracked_csv_dir` is not a directory, raise an error
        #     raise ValueError(f"Invalid `tracked_csv_dir`: {tracked_csv_dir} is not a directory.")

        # # Handle `tracked_video_dir`
        # if tracked_video_dir is None:
        #     # Default to creating `tracked_video` folder in the base folder
        #     tracked_video_dir = os.path.join(base_folder, "tracked_videos")
        #     if not os.path.exists(tracked_video_dir):
        #         os.makedirs(tracked_video_dir)
        #         self.logger.info(f"Created tracked video directory: {tracked_video_dir}")
        #     tracked_video_path = os.path.join(tracked_video_dir, f"{file_name}_MPtracked.mp4")
        # elif os.path.isdir(tracked_video_dir):
        #     if not os.path.exists(tracked_video_dir):
        #         os.makedirs(tracked_video_dir)
        #         self.logger.info(f"Created tracked video directory: {tracked_video_dir}")
        #     # If a directory is provided, append the file name
        #     tracked_video_path = os.path.join(tracked_video_dir, f"{file_name}_MPtracked.mp4")
        # else:
        #     # If `tracked_video_dir` is not a directory, raise an error
        #     raise ValueError(f"Invalid `tracked_video_dir`: {tracked_video_dir} is not a directory.")

if __name__ == "__main__":
    pose_estimator = PoseEstimator()
    pose_estimator.process_video("Input_videos\ghadir\IMG_2601.mp4")
    # pose_estimator.batch_video_processing("./directory/")