import math
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

def butter_lowpass_filter(data, cutoff=3, fs=30, order=4):
    """Filter signal"""
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
    """Plot raw signals for heel and toe landmarks and optionally save the plot."""
    heel_left_z = landmarks[("left_heel", "z")]
    heel_right_z = landmarks[("right_heel", "z")]
    toe_left_z = landmarks[("left_foot_index", "z")]  # Toe
    toe_right_z = landmarks[("right_foot_index", "z")]  # Toe
    
    time = [t / frame_rate for t in range(len(heel_left_z))]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot heels
    axes[0].plot(time, heel_left_z, label="Heel Left (z)", color='blue')
    axes[0].plot(time, heel_right_z, label="Heel Right (z)", color='orange')
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Vertical Position (z)")
    axes[0].set_title("Heel Vertical Displacement")
    axes[0].legend()
    axes[0].grid()
    
    # Plot toes
    axes[1].plot(time, toe_left_z, label="Toe Left (z)", color='blue')
    axes[1].plot(time, toe_right_z, label="Toe Right (z)", color='orange')
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Vertical Position (z)")
    axes[1].set_title("Toe Vertical Displacement")
    axes[1].legend()
    axes[1].grid()
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "raw_pose.png"))
    
    # plt.show()

def plot_extremas(all_forward_movement, all_extrema_data, frame_rate, output_dir="plots"):
    """
    Plots filtered heel and toe signals with detected peaks and valleys.
    
    Parameters:
      - all_forward_movement: Dictionary containing the forward movement data for each event.
      - all_extrema_data: Dictionary containing peaks and valleys for each event.
      - frame_rate: Frame rate of the data for time conversion.
      - output_dir: Directory where the plot will be saved if make_plot is True.
      - make_plot: If True, saves the plot.
    """
    # Prepare time array based on the length of the forward movement data.
    time = np.arange(len(next(iter(all_forward_movement.values())))) / frame_rate

    # Set up plot layout
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Loop through each event and plot its data
    for ax, (event_name, forward_movement) in zip(axes, all_forward_movement.items()):
        peaks = all_extrema_data[event_name]["peaks"]
        valleys = all_extrema_data[event_name]["valleys"]

        # Plot the forward movement (displacement signal)
        ax.plot(time, forward_movement, label=f"{event_name.replace('_', ' ').title()}")
        
        # Scatter plot peaks and valleys
        ax.scatter(peaks, forward_movement[(peaks * frame_rate).astype(int)], color='red', label="Peaks")
        ax.scatter(valleys, forward_movement[(valleys * frame_rate).astype(int)], color='blue', label="Valleys")

        ax.set_title(f"{event_name.replace('_', ' ').title()} Displacement")
        ax.legend()
        ax.grid()

    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "extremas.png"), dpi=300)
    
    # plt.show()

def plot_extrema_frames(extremas_dict, output_dir, frames_dir):
    """Plot and save extrema frames dynamically from dictionary."""
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
                frame_filename = os.path.join(frames_dir, f"frame_{idx}.png")
                frame = cv2.imread(frame_filename)

                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    axes[i].imshow(frame_rgb)
                    axes[i].axis('off')
                    axes[i].set_title(f"Frame {idx}", fontsize=22)
                else:
                    axes[i].axis('off')  # Hide empty subplot

            # Hide any remaining empty subplots
            for j in range(i + 1, len(axes)):
                axes[j].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{key}_{extrema_type}.png"), dpi=300)
            # plt.show()

def extract_frames(video_path, output_dir):
    """Extract all frames from a video and save them."""
    
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
