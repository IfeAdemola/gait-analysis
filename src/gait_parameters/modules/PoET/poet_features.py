import os
import pandas as pd
import numpy as np
from tqdm import tqdm

from scipy import signal
from scipy import ndimage
from scipy.signal import medfilt, hilbert, butter, lfilter
import matplotlib.pyplot as plt

# Import local utilities (assumed to work with flat column names)
from .poet_utils import identify_active_time_period, check_hand_, meanfilt
from skimage.restoration import denoise_tv_chambolle

# Define a global output directory for plots.
PLOTS_DIR = "./plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Global debug flag
DEBUG = True

# ---------------------------
# Data Loading and Time Period Assignment
# ---------------------------
def assign_hand_time_periods(pc):
    """
    Loop over patients and assign active time periods based on the flat column structure.
    Assumes that p.structural_features already has preformatted column names.
    """
    print('Extracting active time periods of hands ... ')
    for p in tqdm(pc.patients, total=len(pc.patients)):
        fs = p.sampling_frequency
        if DEBUG:
            if p.structural_features is None or p.structural_features.empty:
                print(f"Debug: Structural features for patient {p.patient_id} not found (empty DataFrame).")
            else:
                print(f"Debug: Structural features for patient {p.patient_id} loaded. Columns: {p.structural_features.columns.tolist()}")
                print(f"Debug: First two rows for patient {p.patient_id}:\n{p.structural_features.head(2)}")
        hands = check_hand_(p)
        structural_features = p.structural_features
        if len(hands) > 1:
            time_periods = identify_active_time_period(structural_features, fs)
        else:
            time_periods = None
        p.hand_time_periods = time_periods
    return pc

# ---------------------------
# Signal Processing Functions (unchanged)
# ---------------------------
def tv1d_denoise(signal_array, fs):
    """
    Perform 1D total variation denoising.
    """
    weight = fs / 1000.0
    signal_2d = signal_array.reshape(-1, 1)
    denoised = denoise_tv_chambolle(signal_2d, weight=weight)
    return denoised.flatten()

def spectrogram(x, fs):
    n = len(x)
    x = np.pad(x, (int(fs/2), int(fs/2)), mode='symmetric')
    amplitudes = []
    frequencies = []
    for i in range(n):
        y = x[i:i+int(fs)]
        fft_result = np.fft.fft(y)
        frequency = np.fft.fftfreq(y.size, d=1/fs)
        amplitude = 2 * np.abs(fft_result) / len(y)
        amplitudes.append(amplitude)
        frequencies.append(frequency)
    amplitude = np.vstack(amplitudes)
    amplitude = amplitude[:, frequency >= 0]
    frequency = frequency[frequency >= 0]
    return frequency, amplitude

def spectrum(y, fs):
    fft_result = np.fft.fft(y)
    frequency = np.fft.fftfreq(y.size, d=1/fs)
    amplitude = 2 * np.abs(fft_result) / len(y)
    amplitude = amplitude[frequency >= 0]
    frequency = frequency[frequency >= 0]
    return frequency, amplitude

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#####################################
# PCA Analysis Helper Functions
#####################################
def pca_main_component(data):
    """
    Perform PCA on a 2D array (n_samples x 3) and return the first principal component
    and the projection of the data onto that component.
    """
    data_centered = data - np.mean(data, axis=0)
    U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
    principal_component = Vt[0]
    projection = data_centered.dot(principal_component)
    return principal_component, projection

def pca_tremor_analysis(signal_3d, fs):
    """
    Given a 3D time series (n_samples x 3), perform PCA to find the main tremor axis,
    then compute tremor features using the projected (1D) signal.
    """
    principal_component, projection = pca_main_component(signal_3d)
    analytic_signal = hilbert(projection)
    inst_amplitude = np.abs(analytic_signal)
    max_hilbert_amp = inst_amplitude.max()
    mean_hilbert_amp = inst_amplitude.mean()
    
    f, P = spectrum(projection, fs)
    f_spec, S = spectrogram(projection, fs)
    mean_freq = f_spec[S.mean(axis=0).argmax()]
    max_freq = f_spec[S.max(axis=0).argmax()]
    
    dom_f_idx = P.argmax()
    dominant_frequency = f[dom_f_idx]
    power_spectral_max_amp = P[dom_f_idx]
    
    features = {
        'pca_hilbert_max_amplitude': 2 * max_hilbert_amp,
        'pca_hilbert_mean_amplitude': 2 * mean_hilbert_amp,
        'pca_spectrogram_mean_frequency': mean_freq,
        'pca_spectrogram_max_frequency': max_freq,
        'pca_power_spectral_dominant_frequency': dominant_frequency,
        'pca_power_spectral_max_amplitude': 2 * power_spectral_max_amp
    }
    return features, projection, principal_component

###############################################
# New Tremor Feature Extraction Functions (using flat column names)
###############################################
def extract_proximal_arm_tremor_features(pc, plot=False, save_plots=False, debug=DEBUG):
    """
    Extract proximal arm tremor features (shoulder and elbow markers) using PCA.
    Uses flat, preformatted column names.
    Expected columns (for right hand):
        marker_right_shoulder_x, marker_right_shoulder_y, marker_right_shoulder_z,
        marker_right_elbow_x, marker_right_elbow_y, marker_right_elbow_z
    and similarly for left hand.
    """
    features_df = pd.DataFrame(index=pc.get_patient_ids())
    print('Extracting proximal arm tremor features using PCA ... ')
    
    for p in tqdm(pc.patients, total=len(pc.patients)):
        fs = p.sampling_frequency
        hands = check_hand_(p)
        structural_features = p.structural_features
        
        if debug:
            if structural_features is None or structural_features.empty:
                print(f"Debug: No structural features found for patient {p.patient_id}.")
            else:
                print(f"Debug: Structural features for patient {p.patient_id} loaded. Columns: {structural_features.columns.tolist()}")
                print(f"Debug: First two rows for patient {p.patient_id}:\n{structural_features.head(2)}")
        
        if len(hands) > 1:
            if not p.hand_time_periods:
                time_periods = identify_active_time_period(structural_features, fs)
            else:
                time_periods = p.hand_time_periods
        else:
            time_periods = None
        
        for hand in hands:
            if hand == 'right':
                shoulder_cols = ['marker_right_shoulder_x', 'marker_right_shoulder_y', 'marker_right_shoulder_z']
                elbow_cols    = ['marker_right_elbow_x', 'marker_right_elbow_y', 'marker_right_elbow_z']
            else:
                shoulder_cols = ['marker_left_shoulder_x', 'marker_left_shoulder_y', 'marker_left_shoulder_z']
                elbow_cols    = ['marker_left_elbow_x', 'marker_left_elbow_y', 'marker_left_elbow_z']
            
            if debug:
                print(f"Debug: For patient {p.patient_id}, processing {hand} hand.")
                print(f"Debug: Expected shoulder columns: {shoulder_cols}")
                print(f"Debug: Expected elbow columns: {elbow_cols}")
            
            # For time window calculation, use one representative column per marker
            if time_periods and hand in time_periods and len(time_periods[hand]) > 0:
                key_names = [shoulder_cols[0], elbow_cols[0]]
                start_frame = min([time_periods[hand].get(key, [0, 0])[0] for key in key_names])
                end_frame   = max([time_periods[hand].get(key, [0, structural_features.shape[0]])[1] for key in key_names])
            else:
                start_frame = 0
                end_frame = structural_features.shape[0]
            
            if debug:
                print(f"Debug: Frame window for patient {p.patient_id}, {hand} hand: {start_frame} to {end_frame}")
            
            try:
                data_shoulder = structural_features.loc[:, shoulder_cols].interpolate().iloc[start_frame:end_frame].dropna()
            except KeyError as e:
                print(f"Debug: KeyError when accessing shoulder columns for patient {p.patient_id} ({hand} hand).")
                print(f"Debug: Available columns: {structural_features.columns.tolist()}")
                continue
            
            try:
                data_elbow = structural_features.loc[:, elbow_cols].interpolate().iloc[start_frame:end_frame].dropna()
            except KeyError as e:
                print(f"Debug: KeyError when accessing elbow columns for patient {p.patient_id} ({hand} hand).")
                print(f"Debug: Available columns: {structural_features.columns.tolist()}")
                continue
            
            common_index = data_shoulder.index.intersection(data_elbow.index)
            data_shoulder = data_shoulder.loc[common_index]
            data_elbow = data_elbow.loc[common_index]
            
            if data_shoulder.empty or data_elbow.empty:
                print(f"Proximal arm tremor for {hand} hand not found (no overlapping data).")
                continue
            
            centroid = (data_shoulder.values + data_elbow.values) / 2.0
            centroid_detrended = np.apply_along_axis(signal.detrend, 0, centroid)
            
            filtered = np.zeros_like(centroid_detrended)
            for i in range(3):
                filtered[:, i] = butter_bandpass_filter(centroid_detrended[:, i], 3, 12, fs, order=5)
            
            features, projection, principal_component = pca_tremor_analysis(filtered, fs)
            
            if plot:
                plt.figure()
                plt.plot(projection)
                plt.title(f"{p.patient_id}_{hand}_proximal_arm")
                if save_plots:
                    plt.savefig(os.path.join(PLOTS_DIR, f'pca_proximal_arm_{p.patient_id}_{hand}.svg'))
            
            features_df.loc[p.patient_id, f'pca_hilbert_max_amplitude_proximal_arm_{hand}'] = features['pca_hilbert_max_amplitude']
            features_df.loc[p.patient_id, f'pca_hilbert_mean_amplitude_proximal_arm_{hand}'] = features['pca_hilbert_mean_amplitude']
            features_df.loc[p.patient_id, f'pca_spectrogram_mean_frequency_proximal_arm_{hand}'] = features['pca_spectrogram_mean_frequency']
            features_df.loc[p.patient_id, f'pca_spectrogram_max_frequency_proximal_arm_{hand}'] = features['pca_spectrogram_max_frequency']
            features_df.loc[p.patient_id, f'pca_power_spectral_dominant_frequency_proximal_arm_{hand}'] = features['pca_power_spectral_dominant_frequency']
            features_df.loc[p.patient_id, f'pca_power_spectral_max_amplitude_proximal_arm_{hand}'] = features['pca_power_spectral_max_amplitude']
    return features_df

def extract_distal_arm_tremor_features(pc, plot=False, save_plots=False, debug=DEBUG):
    """
    Extract distal arm tremor features (elbow and wrist markers) using PCA.
    Uses flat, preformatted column names.
    Expected columns for right hand:
        marker_right_elbow_x, marker_right_elbow_y, marker_right_elbow_z,
        marker_right_wrist_x, marker_right_wrist_y, marker_right_wrist_z
    and similarly for left hand.
    """
    features_df = pd.DataFrame(index=pc.get_patient_ids())
    print('Extracting distal arm tremor features using PCA ... ')
    
    for p in tqdm(pc.patients, total=len(pc.patients)):
        fs = p.sampling_frequency
        hands = check_hand_(p)
        structural_features = p.structural_features
        
        if debug:
            if structural_features is None or structural_features.empty:
                print(f"Debug: No structural features found for patient {p.patient_id}.")
            else:
                print(f"Debug: Structural features for patient {p.patient_id} loaded. Columns: {structural_features.columns.tolist()}")
                print(f"Debug: First two rows for patient {p.patient_id}:\n{structural_features.head(2)}")
        
        if len(hands) > 1:
            if not p.hand_time_periods:
                time_periods = identify_active_time_period(structural_features, fs)
            else:
                time_periods = p.hand_time_periods
        else:
            time_periods = None
        
        for hand in hands:
            if hand == 'right':
                elbow_cols = ['marker_right_elbow_x', 'marker_right_elbow_y', 'marker_right_elbow_z']
                wrist_cols = ['marker_right_wrist_x', 'marker_right_wrist_y', 'marker_right_wrist_z']
            else:
                elbow_cols = ['marker_left_elbow_x', 'marker_left_elbow_y', 'marker_left_elbow_z']
                wrist_cols = ['marker_left_wrist_x', 'marker_left_wrist_y', 'marker_left_wrist_z']
            
            if debug:
                print(f"Debug: For patient {p.patient_id}, processing {hand} hand.")
                print(f"Debug: Expected elbow columns: {elbow_cols}")
                print(f"Debug: Expected wrist columns: {wrist_cols}")
            
            # Use a representative column from each marker for the time window
            if time_periods and hand in time_periods and len(time_periods[hand]) > 0:
                key_names = [elbow_cols[0], wrist_cols[0]]
                start_frame = min([time_periods[hand].get(key, [0, 0])[0] for key in key_names])
                end_frame   = max([time_periods[hand].get(key, [0, structural_features.shape[0]])[1] for key in key_names])
            else:
                start_frame = 0
                end_frame = structural_features.shape[0]
            
            if debug:
                print(f"Debug: Frame window for patient {p.patient_id}, {hand} hand: {start_frame} to {end_frame}")
            
            try:
                data_elbow = structural_features.loc[:, elbow_cols].interpolate().iloc[start_frame:end_frame].dropna()
            except KeyError as e:
                print(f"Debug: KeyError when accessing elbow columns for patient {p.patient_id} ({hand} hand).")
                print(f"Debug: Available columns: {structural_features.columns.tolist()}")
                continue
            
            try:
                data_wrist = structural_features.loc[:, wrist_cols].interpolate().iloc[start_frame:end_frame].dropna()
            except KeyError as e:
                print(f"Debug: KeyError when accessing wrist columns for patient {p.patient_id} ({hand} hand).")
                print(f"Debug: Available columns: {structural_features.columns.tolist()}")
                continue
            
            common_index = data_elbow.index.intersection(data_wrist.index)
            data_elbow = data_elbow.loc[common_index]
            data_wrist = data_wrist.loc[common_index]
            
            if data_elbow.empty or data_wrist.empty:
                print(f"Distal arm tremor for {hand} hand not found (no overlapping data).")
                continue
            
            centroid = (data_elbow.values + data_wrist.values) / 2.0
            centroid_detrended = np.apply_along_axis(signal.detrend, 0, centroid)
            
            filtered = np.zeros_like(centroid_detrended)
            for i in range(3):
                filtered[:, i] = butter_bandpass_filter(centroid_detrended[:, i], 3, 12, fs, order=5)
            
            features, projection, principal_component = pca_tremor_analysis(filtered, fs)
            
            if plot:
                plt.figure()
                plt.plot(projection)
                plt.title(f"{p.patient_id}_{hand}_distal_arm")
                if save_plots:
                    plt.savefig(os.path.join(PLOTS_DIR, f'pca_distal_arm_{p.patient_id}_{hand}.svg'))
            
            features_df.loc[p.patient_id, f'pca_hilbert_max_amplitude_distal_arm_{hand}'] = features['pca_hilbert_max_amplitude']
            features_df.loc[p.patient_id, f'pca_hilbert_mean_amplitude_distal_arm_{hand}'] = features['pca_hilbert_mean_amplitude']
            features_df.loc[p.patient_id, f'pca_spectrogram_mean_frequency_distal_arm_{hand}'] = features['pca_spectrogram_mean_frequency']
            features_df.loc[p.patient_id, f'pca_spectrogram_max_frequency_distal_arm_{hand}'] = features['pca_spectrogram_max_frequency']
            features_df.loc[p.patient_id, f'pca_power_spectral_dominant_frequency_distal_arm_{hand}'] = features['pca_power_spectral_dominant_frequency']
            features_df.loc[p.patient_id, f'pca_power_spectral_max_amplitude_distal_arm_{hand}'] = features['pca_power_spectral_max_amplitude']
    return features_df

def extract_fingers_tremor_features(pc, plot=False, save_plots=False, debug=DEBUG):
    """
    Extract fingers (index and middle finger) tremor features via PCA.
    Uses flat, preformatted column names.
    Expected columns for right hand:
        marker_index_finger_tip_right_x, marker_index_finger_tip_right_y, marker_index_finger_tip_right_z,
        marker_middle_finger_tip_right_x, marker_middle_finger_tip_right_y, marker_middle_finger_tip_right_z
    and similarly for left hand.
    """
    features_df = pd.DataFrame(index=pc.get_patient_ids())
    print('Extracting fingers/hands tremor features using PCA ... ')
    
    for p in tqdm(pc.patients, total=len(pc.patients)):
        fs = p.sampling_frequency
        hands = check_hand_(p)
        structural_features = p.structural_features
        
        if debug:
            if structural_features is None or structural_features.empty:
                print(f"Debug: No structural features found for patient {p.patient_id}.")
            else:
                print(f"Debug: Structural features for patient {p.patient_id} loaded. Columns: {structural_features.columns.tolist()}")
                print(f"Debug: First two rows for patient {p.patient_id}:\n{structural_features.head(2)}")
        
        if len(hands) > 1:
            if not p.hand_time_periods:
                time_periods = identify_active_time_period(structural_features, fs)
            else:
                time_periods = p.hand_time_periods
        else:
            time_periods = None
        
        for hand in hands:
            if hand == 'right':
                index_cols = ['marker_index_finger_tip_right_x', 'marker_index_finger_tip_right_y', 'marker_index_finger_tip_right_z']
                middle_cols = ['marker_middle_finger_tip_right_x', 'marker_middle_finger_tip_right_y', 'marker_middle_finger_tip_right_z']
            else:
                index_cols = ['marker_index_finger_tip_left_x', 'marker_index_finger_tip_left_y', 'marker_index_finger_tip_left_z']
                middle_cols = ['marker_middle_finger_tip_left_x', 'marker_middle_finger_tip_left_y', 'marker_middle_finger_tip_left_z']
            
            if debug:
                print(f"Debug: For patient {p.patient_id}, processing {hand} hand.")
                print(f"Debug: Expected index columns: {index_cols}")
                print(f"Debug: Expected middle finger columns: {middle_cols}")
            
            # Use a representative column for time window calculation
            if time_periods and hand in time_periods and len(time_periods[hand]) > 0:
                key_names = [index_cols[0], middle_cols[0]]
                start_frame = min([time_periods[hand].get(key, [0, 0])[0] for key in key_names])
                end_frame   = max([time_periods[hand].get(key, [0, structural_features.shape[0]])[1] for key in key_names])
            else:
                start_frame = 0
                end_frame = structural_features.shape[0]
            
            if debug:
                print(f"Debug: Frame window for patient {p.patient_id}, {hand} hand: {start_frame} to {end_frame}")
            
            try:
                data_index = structural_features.loc[:, index_cols].interpolate().iloc[start_frame:end_frame].dropna()
            except KeyError as e:
                print(f"Debug: KeyError when accessing index finger columns for patient {p.patient_id} ({hand} hand).")
                print(f"Debug: Available columns: {structural_features.columns.tolist()}")
                continue
            
            try:
                data_middle = structural_features.loc[:, middle_cols].interpolate().iloc[start_frame:end_frame].dropna()
            except KeyError as e:
                print(f"Debug: KeyError when accessing middle finger columns for patient {p.patient_id} ({hand} hand).")
                print(f"Debug: Available columns: {structural_features.columns.tolist()}")
                continue
            
            common_index = data_index.index.intersection(data_middle.index)
            data_index = data_index.loc[common_index]
            data_middle = data_middle.loc[common_index]
            
            if data_index.empty or data_middle.empty:
                print(f"Fingers/hands tremor for {hand} hand not found (no overlapping data).")
                continue
            
            centroid = (data_index.values + data_middle.values) / 2.0
            centroid_detrended = np.apply_along_axis(signal.detrend, 0, centroid)
            
            filtered = np.zeros_like(centroid_detrended)
            for i in range(3):
                filtered[:, i] = butter_bandpass_filter(centroid_detrended[:, i], 3, 12, fs, order=5)
            
            features, projection, principal_component = pca_tremor_analysis(filtered, fs)
            
            if plot:
                plt.figure()
                plt.plot(projection)
                plt.title(f"{p.patient_id}_{hand}_fingers")
                if save_plots:
                    plt.savefig(os.path.join(PLOTS_DIR, f'pca_fingers_{p.patient_id}_{hand}.svg'))
            
            features_df.loc[p.patient_id, f'pca_hilbert_max_amplitude_fingers_{hand}'] = features['pca_hilbert_max_amplitude']
            features_df.loc[p.patient_id, f'pca_hilbert_mean_amplitude_fingers_{hand}'] = features['pca_hilbert_mean_amplitude']
            features_df.loc[p.patient_id, f'pca_spectrogram_mean_frequency_fingers_{hand}'] = features['pca_spectrogram_mean_frequency']
            features_df.loc[p.patient_id, f'pca_spectrogram_max_frequency_fingers_{hand}'] = features['pca_spectrogram_max_frequency']
            features_df.loc[p.patient_id, f'pca_power_spectral_dominant_frequency_fingers_{hand}'] = features['pca_power_spectral_dominant_frequency']
            features_df.loc[p.patient_id, f'pca_power_spectral_max_amplitude_fingers_{hand}'] = features['pca_power_spectral_max_amplitude']
    return features_df
