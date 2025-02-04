import os
import pandas as pd
import csv

from scipy.signal import find_peaks
from scipy import signal
from scipy.signal import hilbert, butter, lfilter, medfilt


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
        data (dict): Dictionary containing the data to be saved.
        file_path (str): Path to the CSV file where the data will be saved.
    """
    # Ensure save_path exists
    os.makedirs(file_path, exist_ok=True)
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

# def save_csv(data, file_path, header=None, delimiter=','):
#     """
#     Saves a NumPy array to a CSV file.
#     :param data: numpy array, the data to save
#     :param file_path: str, path to save the CSV file
#     :param header: list or None, column names for the CSV
#     :param delimiter: str, the delimiter to use in the CSV file
#     """
#     try:
#         with open(file_path, mode='w', newline='') as file:
#             writer = csv.writer(file, delimiter=delimiter)
#             if header:
#                 writer.writerow(header)
#             writer.writerows(data)
#         log(f"Data successfully saved to {file_path}", level="INFO")
#     except Exception as e:
#         raise ValueError(f"Error saving CSV file: {e}")

# --- Gait Analysis Utilities ---
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