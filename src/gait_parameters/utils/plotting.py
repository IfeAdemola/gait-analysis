import math
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

def butter_lowpass_filter(data, cutoff=3, fs=30, order=4):
    """Apply a Butterworth low-pass filter to smooth the signal."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def detect_extremas(signal):
    """Find peaks and valleys in the filtered signal."""
    threshold = np.mean(signal)
    peaks, _ = find_peaks(signal, height=threshold)
    valleys, _ = find_peaks(-signal, height=-threshold)
    return peaks, valleys 

def plot_raw_pose(landmarks, frame_rate, output_dir="plots"):
    """
    Plot raw vertical displacement signals for heel and toe landmarks.
    
    Left subplot: Heel vertical positions.
    Right subplot: Toe vertical positions.
    
    The y-axis shows the vertical position (z coordinate) for each landmark.
    Saves the figure as 'raw_pose.png' in the output directory.
    """
    heel_left_z = landmarks[("left_heel", "z")]
    heel_right_z = landmarks[("right_heel", "z")]
    toe_left_z = landmarks[("left_foot_index", "z")]  # Toe
    toe_right_z = landmarks[("right_foot_index", "z")]  # Toe
    
    time = np.array([t / frame_rate for t in range(len(heel_left_z))])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot heel displacement
    axes[0].plot(time, heel_left_z, label="Left Heel (z coordinate)", color='blue', linewidth=2)
    axes[0].plot(time, heel_right_z, label="Right Heel (z coordinate)", color='orange', linewidth=2)
    axes[0].set_xlabel("Time (s)", fontsize=14)
    axes[0].set_ylabel("Vertical Position (z coordinate)", fontsize=14)
    axes[0].set_title("Heel Vertical Position", fontsize=16)
    axes[0].legend(loc='best', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Plot toe displacement
    axes[1].plot(time, toe_left_z, label="Left Toe (z coordinate)", color='blue', linewidth=2)
    axes[1].plot(time, toe_right_z, label="Right Toe (z coordinate)", color='orange', linewidth=2)
    axes[1].set_xlabel("Time (s)", fontsize=14)
    axes[1].set_ylabel("Vertical Position (z coordinate)", fontsize=14)
    axes[1].set_title("Toe Vertical Position", fontsize=16)
    axes[1].legend(loc='best', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "raw_pose.png"), dpi=300)
    plt.close(fig)

def plot_extremas(all_forward_movement, all_extrema_data, frame_rate, input_path, output_dir="plots"):
    """
    Plot the forward displacement signals with detected peaks and valleys.
    
    For each event (e.g., HS_left, TO_right), the plot shows:
      - A line plot of the forward displacement signal, calculated as:
           (landmark z-coordinate - sacrum z-coordinate)
      - Red scatter points marking detected peaks (e.g. heel strikes).
      - Blue scatter points marking detected valleys (e.g. toe-offs).
      
    The x-axis is time (s) and the y-axis is the forward displacement.
    The title for each subplot includes the landmark type (Heel or Toe) and side (Left/Right)
    along with a note that the displacement is the difference between the landmark and sacrum.
    
    The composite figure is saved with a filename derived from the input file.
    """
    # Create time axis
    time = np.arange(len(next(iter(all_forward_movement.values())))) / frame_rate

    # Create a 2x2 grid (or adjust if you have a different number of events)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, (event_name, forward_movement) in zip(axes, all_forward_movement.items()):
        peaks_time = all_extrema_data[event_name]["peaks"]
        valleys_time = all_extrema_data[event_name]["valleys"]
        
        # Plot the forward displacement signal (z landmark - z sacrum)
        ax.plot(time, forward_movement, label="Forward Displacement", 
                color='black', linewidth=2)
        
        # Convert time (in seconds) to indices for extracting y-values from the signal
        indices_peaks = (peaks_time * frame_rate).astype(int)
        indices_valleys = (valleys_time * frame_rate).astype(int)
        
        # Scatter plot peaks and valleys
        ax.scatter(peaks_time, forward_movement[indices_peaks], color='red', s=50, label="Peaks")
        ax.scatter(valleys_time, forward_movement[indices_valleys], color='blue', s=50, label="Valleys")
        
        # Create a descriptive title for the subplot:
        # Determine landmark type and side from event name.
        if "hs" in event_name.lower():
            landmark_type = "Heel"
        elif "to" in event_name.lower():
            landmark_type = "Toe"
        else:
            landmark_type = event_name.title()
        side = "Left" if "left" in event_name.lower() else "Right"
        title_text = f"{side} {landmark_type} Forward Displacement\n(z landmark - z sacrum)"
        ax.set_title(title_text, fontsize=14)
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Forward Displacement\n(z coordinate difference)", fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Optionally, add annotations for a few extrema points
        if len(peaks_time) < 10:
            for t, y in zip(peaks_time, forward_movement[indices_peaks]):
                ax.annotate("Peak", xy=(t, y), xytext=(t, y + 0.05),
                            arrowprops=dict(arrowstyle="->", color='red'), fontsize=8)
        if len(valleys_time) < 10:
            for t, y in zip(valleys_time, forward_movement[indices_valleys]):
                ax.annotate("Valley", xy=(t, y), xytext=(t, y - 0.05),
                            arrowprops=dict(arrowstyle="->", color='blue'), fontsize=8)

    filename = os.path.basename(input_path).split('.')[0]
    fig.suptitle(f"Forward Displacement Signals for {filename}\n(Computed as landmark z - sacrum z)", fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{filename}_extremas.png"), dpi=300)
    plt.close(fig)

def plot_extrema_frames(extremas_dict, output_dir, frames_dir):
    """
    Plot and save video frames corresponding to the detected extrema.
    
    For each key in extremas_dict, two plots are created (one for peaks and one for valleys)
    that display the frames (images) associated with the detected extrema. The images are
    arranged in a grid and saved in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)

    for key, (peaks, valleys) in extremas_dict.items():
        extrema_types = {'peaks': peaks, 'valleys': valleys}

        for extrema_type, indices in extrema_types.items():
            if len(indices) == 0:
                continue  # Skip if no extrema points found
        
            img_width, img_height = 10, 7
            cols = 4
            rows = math.ceil(len(indices) / cols)
            figsize = (cols * img_width, rows * img_height)

            fig, axes = plt.subplots(rows, cols, figsize=figsize)
            axes = np.array(axes).flatten()  # Ensure axes is an array for iteration

            for i, idx in enumerate(indices):
                frame_filename = os.path.join(frames_dir, f"frame_{int(idx)}.png")
                frame = cv2.imread(frame_filename)

                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    axes[i].imshow(frame_rgb)
                    axes[i].axis('off')
                    axes[i].set_title(f"Frame {int(idx)}", fontsize=16)
                else:
                    axes[i].axis('off')  # Hide empty subplot

            # Hide any remaining empty subplots
            for j in range(i + 1, len(axes)):
                axes[j].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{key}_{extrema_type}.png"), dpi=300)
            plt.close(fig)

def extract_frames(video_path, output_dir):
    """Extract all frames from a video and save them as individual images."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for idx in range(total_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_filename = os.path.join(output_dir, f"frame_{idx}.png")
            cv2.imwrite(frame_filename, frame)
    
    cap.release()
