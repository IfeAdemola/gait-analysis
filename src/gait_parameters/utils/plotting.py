import math
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

def butter_lowpass_filter(data, cutoff=3, fs=30, order=4):
    """
    Apply a Butterworth low-pass filter to smooth the signal.
    
    Parameters:
      - data: The input signal.
      - cutoff: The cutoff frequency.
      - fs: Sampling frequency.
      - order: Order of the filter.
      
    Returns:
      - The filtered signal.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def detect_extremas(signal):
    """
    Find peaks and valleys in the filtered signal.
    
    Uses the mean value as a threshold.
    
    Parameters:
      - signal: 1D numpy array.
      
    Returns:
      - peaks: Indices of peaks.
      - valleys: Indices of valleys.
    """
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
    
    Parameters:
      - landmarks: A DataFrame or dict with keys like ("left_heel", "z").
      - frame_rate: Frames per second.
      - output_dir: Directory to save the plot.
    """
    heel_left_z = landmarks[("left_heel", "z")]
    heel_right_z = landmarks[("right_heel", "z")]
    toe_left_z = landmarks[("left_foot_index", "z")]  # Toe marker
    toe_right_z = landmarks[("right_foot_index", "z")]  # Toe marker
    
    time = np.array([t / frame_rate for t in range(len(heel_left_z))])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot heel vertical positions
    axes[0].plot(time, heel_left_z, label="Left Heel (z coordinate)", color='blue', linewidth=2)
    axes[0].plot(time, heel_right_z, label="Right Heel (z coordinate)", color='orange', linewidth=2)
    axes[0].set_xlabel("Time (s)", fontsize=14)
    axes[0].set_ylabel("Vertical Position (z coordinate)", fontsize=14)
    axes[0].set_title("Heel Vertical Position", fontsize=16)
    axes[0].legend(loc='best', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # Plot toe vertical positions
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
      - A line plot of the forward displacement signal computed as:
            (landmark z-coordinate - sacrum z-coordinate)
      - Red scatter points marking detected peaks (e.g. heel strikes).
      - Blue scatter points marking detected valleys (e.g. toe-offs).
      
    The x-axis is time (s) and the y-axis is the forward displacement.
    The title for each subplot includes the landmark type (Heel or Toe) and side (Left/Right)
    along with a note on the displacement computation.
    
    The composite figure is saved with a filename derived from the input file.
    
    Parameters:
      - all_forward_movement: Dict mapping event names to displacement signals.
      - all_extrema_data: Dict mapping event names to extrema data (peaks, valleys).
      - frame_rate: Frames per second.
      - input_path: Path to the input file (used to name the output file).
      - output_dir: Directory to save the plot.
    """
    # Create time axis
    time = np.arange(len(next(iter(all_forward_movement.values())))) / frame_rate

    # Create a 2x2 grid for the subplots (adjust if number of events differs)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, (event_name, forward_movement) in zip(axes, all_forward_movement.items()):
        peaks_time = all_extrema_data[event_name]["peaks"]
        valleys_time = all_extrema_data[event_name]["valleys"]
        
        # Plot the forward displacement signal (landmark z - sacrum z)
        ax.plot(time, forward_movement, label="Forward Displacement", 
                color='black', linewidth=2)
        
        # Convert time (in seconds) to indices for y-values extraction
        indices_peaks = (peaks_time * frame_rate).astype(int)
        indices_valleys = (valleys_time * frame_rate).astype(int)
        
        # Scatter plot for peaks and valleys
        ax.scatter(peaks_time, forward_movement[indices_peaks], color='red', s=50, label="Peaks")
        ax.scatter(valleys_time, forward_movement[indices_valleys], color='blue', s=50, label="Valleys")
        
        # Build a descriptive title based on event name
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
        
        # Optionally add annotations when there are only a few extrema points
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
    
    Parameters:
      - extremas_dict: Dict mapping keys to a tuple (peaks, valleys).
      - output_dir: Directory to save the plots.
      - frames_dir: Directory containing the extracted frame images.
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
            axes = np.array(axes).flatten()  # Flatten axes array for easy iteration

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
    """
    Extract all frames from a video and save them as individual images.
    
    Parameters:
      - video_path: Path to the video file.
      - output_dir: Directory where frames will be saved.
    """
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

def plot_combined_toe(all_forward_movement, all_extrema_data, frame_rate, input_path, output_dir="plots"):
    """
    Plot combined toe events in two ways:
      1. A time series plot that overlays Toe Left and Toe Right forward displacement signals.
         (Forward displacement is computed as toe z-coordinate minus sacrum z-coordinate.)
      2. A phase plot of Toe Left displacement versus Toe Right displacement.
    
    This helps visualize both the individual signal patterns (top plot) and how the two signals cycle
    relative to one another (bottom plot).
    
    Parameters:
      - all_forward_movement: Dict mapping event names to displacement signals.
      - all_extrema_data: Dict mapping event names to extrema data (peaks, valleys) [unused here].
      - frame_rate: Frames per second.
      - input_path: Path to the input file (used for naming the output file).
      - output_dir: Directory to save the plot.
    """
    # Retrieve toe signals
    toe_left_signal = all_forward_movement.get("TO_left")
    toe_right_signal = all_forward_movement.get("TO_right")

    # Create time axis (in seconds)
    time = np.arange(len(toe_left_signal)) / frame_rate

    # Create a figure with two subplots: top (time series) and bottom (phase plot)
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    
    # --- Top Plot: Time Series ---
    ax = axes[0]
    ax.plot(time, toe_left_signal, label="Toe Left", color="blue", linewidth=2)
    ax.plot(time, toe_right_signal, label="Toe Right", color="orange", linewidth=2)
    
    ax.set_title("Combined Toe Forward Displacement Time Series\n(Toe z - Sacrum z)", fontsize=16)
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Forward Displacement (z difference)", fontsize=14)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # --- Bottom Plot: Phase Plot ---
    ax2 = axes[1]
    # The phase plot shows Toe Left displacement on the x-axis and Toe Right displacement on the y-axis.
    # This visualizes how the two signals vary relative to each other over the gait cycle.
    ax2.plot(toe_left_signal, toe_right_signal, color='black', linewidth=2)
    ax2.set_title("Phase Plot: Toe Left vs. Toe Right", fontsize=16)
    ax2.set_xlabel("Toe Left Displacement (z difference)", fontsize=14)
    ax2.set_ylabel("Toe Right Displacement (z difference)", fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Save the combined toe plot
    filename = os.path.basename(input_path).split('.')[0]
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}_combined_toe.png"), dpi=300)
    plt.close(fig)

