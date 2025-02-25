import cv2
import torch
import os

class YOLOCropper:
    def __init__(self, model_name="yolov5s", confidence_threshold=0.5):
        # Load the pre-trained YOLOv5 model from Ultralytics via torch.hub
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.confidence_threshold = confidence_threshold
        # This variable will store the user-selected person index (0-indexed) once chosen.
        self.selected_person_index = None

    def detect_persons(self, frame):
        """
        Run YOLO on the given frame and return a list of bounding boxes for detected persons.
        Each bounding box is a tuple: (x1, y1, x2, y2).
        """
        results = self.model(frame)
        detections = results.xyxy[0].cpu().numpy()
        person_detections = [det for det in detections if int(det[5]) == 0 and det[4] >= self.confidence_threshold]
        boxes = []
        for det in person_detections:
            x1, y1, x2, y2 = map(int, det[:4])
            boxes.append((x1, y1, x2, y2))
        return boxes

    def prompt_user_for_selection(self, frame, boxes):
        """
        Draw bounding boxes with labels on the frame and display a window prompting the user
        to select one of the detected persons by pressing a number key.
        Returns the selected index (0-indexed).
        """
        display_frame = frame.copy()
        for idx, (x1, y1, x2, y2) in enumerate(boxes):
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"{idx+1}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        window_name = "Multiple persons detected - Press the corresponding number key"
        cv2.imshow(window_name, display_frame)
        print("Multiple persons detected. Please press the number key corresponding to the person to crop (e.g., '1' for first person).")
        
        key = cv2.waitKey(0)
        cv2.destroyWindow(window_name)
        cv2.waitKey(1)  # allow the window to close properly
        
        try:
            selected_index = int(chr(key)) - 1  # Convert pressed key to 0-indexed integer.
            if selected_index < 0 or selected_index >= len(boxes):
                print("Invalid selection. Defaulting to first detected person.")
                return 0
            return selected_index
        except Exception:
            print("Error in selection. Defaulting to first detected person.")
            return 0


    def crop_video(self, input_video_path, output_video_path, margin=50, max_fail_frames=20, smoothing_factor=0.9):
        """
        Process the input video frame by frame. For each frame, detect persons.
        - If no person is detected, the full frame is used (or the last known box).
        - If one person is detected, that bounding box is used.
        - If multiple persons are detected:
             * Prompt for selection (if not already done) and use the chosen box.
        Additionally, if the intended person is not detected for more than max_fail_frames consecutive frames,
        the processing stops and the video is cut at that timestamp.
        
        The selected bounding box is expanded by the given margin, then smoothed temporally using an
        exponential moving average with smoothing_factor, and used to crop the frame.
        
        Returns:
            output_video_path (str): Path to the saved cropped video.
            cropped_frame_size (tuple): (width, height) of the cropped frames.
        """
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {input_video_path}")
    
        fps = cap.get(cv2.CAP_PROP_FPS)
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = None
        cropped_frame_size = None
    
        self.last_selected_box = None  # last known detection
        self.smoothed_box = None       # smoothed bounding box
        consecutive_fail_count = 0
    
        while True:
            ret, frame = cap.read()
            if not ret:
                break
    
            boxes = self.detect_persons(frame)
            selected_box = None
    
            if not boxes:
                # No detection; if we have a last known box, use it.
                if self.last_selected_box is not None:
                    selected_box = self.last_selected_box
                else:
                    selected_box = (0, 0, orig_width, orig_height)
                consecutive_fail_count += 1
            elif len(boxes) == 1:
                selected_box = boxes[0]
                self.last_selected_box = selected_box
                consecutive_fail_count = 0
            else:
                # Multiple detections.
                if self.selected_person_index is None:
                    self.selected_person_index = self.prompt_user_for_selection(frame, boxes)
                if self.selected_person_index < len(boxes):
                    selected_box = boxes[self.selected_person_index]
                    self.last_selected_box = selected_box
                    consecutive_fail_count = 0
                else:
                    if self.last_selected_box is not None:
                        selected_box = self.last_selected_box
                    else:
                        selected_box = boxes[0]
                        self.last_selected_box = selected_box
                    consecutive_fail_count += 1
    
            if consecutive_fail_count >= max_fail_frames:
                print(f"Person not detected for {max_fail_frames} consecutive frames. Cutting video here.")
                break
    
            # Apply temporal smoothing using exponential moving average.
            if self.smoothed_box is None:
                self.smoothed_box = selected_box
            else:
                self.smoothed_box = tuple(
                    int(smoothing_factor * prev + (1 - smoothing_factor) * curr)
                    for prev, curr in zip(self.smoothed_box, selected_box)
                )
    
            # Expand the smoothed bounding box by the margin.
            x1, y1, x2, y2 = self.smoothed_box
            x1 = max(x1 - margin, 0)
            y1 = max(y1 - margin, 0)
            x2 = min(x2 + margin, orig_width)
            y2 = min(y2 + margin, orig_height)
            cropped_frame = frame[y1:y2, x1:x2]
    
            if out_writer is None:
                cropped_frame_size = (cropped_frame.shape[1], cropped_frame.shape[0])
                out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, cropped_frame_size)
            else:
                if (cropped_frame.shape[1], cropped_frame.shape[0]) != cropped_frame_size:
                    cropped_frame = cv2.resize(cropped_frame, cropped_frame_size)
            out_writer.write(cropped_frame)
    
        cap.release()
        if out_writer:
            out_writer.release()
        print(f"Cropped video saved to: {output_video_path}")
        return output_video_path, cropped_frame_size
