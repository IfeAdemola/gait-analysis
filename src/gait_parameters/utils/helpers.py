import os
import pandas as pd
import numpy as np
import json
import shutil
import skvideo

from scipy.signal import find_peaks
from scipy.signal import butter, lfilter, filtfilt

# --- Set FFmpeg Path ---
def set_ffmpeg_path():
    try:
        # Check if ffmpeg is available in the system PATH
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            print(f"FFmpeg is available. Path: {ffmpeg_path}")
            skvideo.setFFmpegPath(os.path.dirname(ffmpeg_path))
    except FileNotFoundError:
        print("FFmpeg is not found on the system.")
    return

# --- Get Video Frame Rate ---  
def get_metadata_path(file_path):
    """Get the metadata file path corresponding to the input CSV file

    Args:
        input_file (str): Path to the input CSV file of the tracked pose estimation

    Returns:
        str: Path to the metadata JSON file
    """
    # Get the file name without extension and directory
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Construct the metadata file name
    metadata_file = f"{base_name}_metadata.json"
    
    # Get the directory of the input file
    directory = os.path.dirname(file_path)
    
    # Combine directory and metadata file name
    return os.path.join(directory, metadata_file)

def get_frame_rate(file_path):
    """Get the frame rate of a video from its corresponding metadata file

    Args:
        file_path (str): .json file with "metadata" in file name

    Returns:
        int: frame rate of video file
    """
    metadata_path = get_metadata_path(file_path)

    try:
        with open(metadata_path, 'r') as file:
            data = json.load(file)
            fps =  data.get('fps')
            return int(fps) if fps is not None else None
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print('Error reading or parsing file:', e)
        return None
    
 
# --- File Handling Utilities ---
def validate_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return True

def load_csv(file_path, header=[0,1]):
    """Load a CSV file into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.
        header (list, optional): List of column names. Defaults to None.

    Returns:
        pandas.DataFrame: DataFrame containing the CSV data.
    """
    validate_file_exists(file_path)
    try:
        df = pd.read_csv(file_path, header=header)
        return df
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def save_csv(data, file_path):
    """
    Saves the provided data to a CSV file.

    Args:
        data (dict or pd.DataFrame): Data to be saved.
        file_path (str): Path to the CSV file where the data will be saved.
    """
    # Ensure the directory for the file exists
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    
    # If data is already a DataFrame, use it directly
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)


# --- Gait Analysis Utilities ---
def detect_extremas(signal):
    """Find peaks and valleys in the filtered signal."""
    threshold = np.mean(signal)
    peaks, _ = find_peaks(signal, height=threshold)
    valleys, _ = find_peaks(-signal, height=-threshold)
    return peaks, valleys

def detect_peaks(signal, threshold=0.5):
    """
    Detects peaks in a signal above a certain threshold.
    :param signal: numpy array, input signal
    :param threshold: float, minimum peak height
    :return: indices of peaks
    """
    peaks, _ = find_peaks(signal, height=threshold)
    return peaks


# --- Filtering Utilities ---
def butter_lowpass_filter(df, columns=None, cutoff=3, fs=30, order=4):
    """Apply a lowpass Butterworth filter to specified columns of a DataFrame."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    # Automatically exclude columns with these keywords in their name
    exclude_keywords = ["marker", "subindex", "visibility", "presence"]
    
    def should_exclude(column):
        """Check if any level of a MultiIndex column contains an exclusion keyword."""
        return any(keyword in str(level) for level in column for keyword in exclude_keywords)

    # Select numeric columns that do NOT contain excluded keywords
    filtered_columns = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if not should_exclude(col)
    ]

    # Apply the filter to selected columns
    filtered_df = df.copy()
    filtered_df[filtered_columns] = filtered_df[filtered_columns].apply(lambda col: filtfilt(b, a, col.values))

    return filtered_df

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Design a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply a Butterworth bandpass filter to the data."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# --- Logging Utilities ---
def log(message, level="INFO"):
    """
    Simple logging function.
    :param message: str, message to log
    :param level: str, log level (e.g., 'INFO', 'DEBUG', 'WARNING')
    """
    print(f"[{level}] {message}")


def compute_and_save_summary(gait_df, video_name, output_dir):
    """
    Computes summary statistics (mean and median) for each column in the gait DataFrame and saves them as a CSV file.
    Only columns that contain values will be included in the summary.
    The resulting CSV has an explicit 'statistic' column (with values "mean" and "median") and flattened headers.
    The video name is included as the first column in the CSV.
    
    Args:
        gait_df (pd.DataFrame): DataFrame containing per-stride gait parameters.
        video_name (str): Identifier for the video, used in naming the output file.
        output_dir (str): Directory where the summary CSV should be saved.
    
    Returns:
        pd.DataFrame: The summary statistics DataFrame.
    """
    # Compute summary statistics (mean and median for each column)
    summary_stats = gait_df.agg(['mean', 'median'])
    
    # Drop columns that are completely empty (both mean and median are NaN)
    summary_stats = summary_stats.dropna(axis=1, how='all')
    
    # Flatten multi-index columns (e.g., ('left', 'step_length') becomes 'left_step_length')
    summary_stats.columns = [
        "{}_{}".format(col[0], col[1]) if isinstance(col, tuple) else col 
        for col in summary_stats.columns
    ]
    
    # Reset index to turn the row labels ("mean", "median") into a column and rename it "statistic"
    summary_stats = summary_stats.reset_index().rename(columns={'index': 'statistic'})
    
    # Insert video name as the first column
    summary_stats.insert(0, 'video', video_name)
    
    # Build the save path for the individual summary CSV.
    summary_csv_path = os.path.join(output_dir, f"{video_name}_gait_summary.csv")
    
    # Save using the existing helper
    save_csv(summary_stats, summary_csv_path)
    
    # Return the summary DataFrame so it can be aggregated later.
    return summary_stats

