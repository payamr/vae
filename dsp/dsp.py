import numpy as np
import scipy.signal as sig

window_types = [
    'boxcar',
    'bartlett',
    'hann',
    'blackman',
    'blackmanharris'
]
window_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

def compute_cola_window(config):
    """
    Using 'window_type' and 'window_size' values from 'config', computes 
    desired window and corresponding hop size to achieve constant-overlap-add 
    (COLA) property. Note that with some window - window size combinations the
    quality of the COLA reconstruction is not perfect.
    """
    window_type = config['window_type']
    window_size = config['window_size']
    assert(window_size >= 64 and (window_size & (window_size - 1)) == 0)
    if window_type == 'boxcar':
        window = np.ones(window_size)
        return window, window_size
    window_size = window_size - 1
    if window_type == 'bartlett':
        window = sig.windows.bartlett(window_size)
        overlap_factor = 0.5
    elif window_type == 'hann':
        window = sig.windows.hann(window_size)
        overlap_factor = 0.5
    elif window_type == 'blackman':
        window = sig.windows.blackman(window_size)
        overlap_factor = 2.0 / 3.0
    elif window_type == 'blackmanharris':
        window = sig.windows.blackmanharris(window_size)
        overlap_factor = 0.75
    else:
        raise Exception(f"Unsupported window type {window_type}. Must be one of {', '.join(window_types)}")
    
    assert(overlap_factor >= 0.0 and overlap_factor <= 1.0)
    
    hop_factor = 1.0 - overlap_factor
    initial_hop_size = int(np.round(hop_factor * window_size))

    # COLA check
    best_hop_size = initial_hop_size
    best_gain_factor = 1.0
    best_error = np.Infinity
    for hop_size in [initial_hop_size, initial_hop_size - 1]:
        num_windows = int(2.0 / hop_factor)
        cola_check = np.zeros(num_windows * hop_size + (window_size - hop_size))
        for i in range(num_windows):
            n_start = i * hop_size
            n_end = n_start + window_size
            cola_check[n_start : n_end] += window
        gain_factor = 1.0 / np.max(cola_check)
        cola_check *= gain_factor

        cola_start = window_size - hop_size // 2
        cola_end = cola_start + window_size - hop_size
        cola_check = cola_check[cola_start : cola_end]
        cola_check /= np.max(cola_check)

        error = np.sum(np.ones_like(cola_check) - cola_check)
        if error < best_error:
            best_error = error
            best_hop_size = hop_size
            best_gain_factor = gain_factor
    
    hop_size = best_hop_size
    gain_factor = best_gain_factor
    window = np.append(window * gain_factor, np.zeros(1))

    return window, hop_size

def get_windowed_frame_size(config):
    window_size = config['window_size']
    padding_factor = config['padding_factor']
    frame_size = window_size * pow(2, padding_factor)
    return frame_size

def windowed_frame_analysis(signal, config):
    window = config['window']
    window_size = config['window_size']
    assert(len(window) == window_size)
    hop_size = config['hop_size']
    assert(hop_size <= window_size)
    padding_factor = config['padding_factor']
    assert(int(padding_factor) == padding_factor)
    
    zero_phase = config['zero_phase']
    sqrt_window = config['sqrt_window']

    N = len(signal)
    H = hop_size
    num_hops = (N - window_size) // H + 1
    M = num_hops * H + window_size
    if M > N:
        signal = np.append(signal, np.zeros(M - N))
    assert(len(signal) == M)
    
    num_hops = (M - window_size) // H + 1
    
    if sqrt_window and np.all(window > 0):
        window = np.sqrt(window)

    frame_indexer = np.arange(window_size).reshape(1, -1) + hop_size * np.arange(num_hops).reshape(-1, 1)
    
    windowed_frames = np.copy(signal[frame_indexer]) * window

    frame_size = get_windowed_frame_size(config)
    padding_size = frame_size - window_size
    if padding_size > 0:
        windowed_frames = np.append(windowed_frames, np.zeros((num_hops, padding_size)), axis=1)

    if zero_phase:
        zero_phase_shift = 1 - window_size // 2
        windowed_frames = np.roll(windowed_frames, zero_phase_shift, axis=1)
    
    windowed_frame_data = {
        'padded_duration_samp': M,
        'frame_size': frame_size,
        'frames': windowed_frames
    }
        
    if config['return_padded_signal']:
        windowed_frame_data['padded_signal'] = signal
    return windowed_frame_data

def windowed_frame_synthesis(windowed_frame_data, config):
    windowed_frames = windowed_frame_data['frames']
    M = windowed_frame_data['padded_duration_samp']
    
    window = config['window']
    window_size = config['window_size']
    assert(len(window) == window_size)
    hop_size = config['hop_size']
    assert(hop_size <= window_size)
    padding_factor = config['padding_factor']
    assert(int(padding_factor) == padding_factor)
    
    zero_phase = config['zero_phase']
    sqrt_window = config['sqrt_window']
    
    signal = np.zeros(M)
    
    if zero_phase:
        zero_phase_shift = 1 - window_size // 2
        windowed_frames = np.roll(windowed_frames, -zero_phase_shift, axis=1)
    
    frame_size = get_windowed_frame_size(config)
    assert(frame_size == windowed_frame_data['frame_size'])
    padding_size = frame_size - window_size
    if padding_size > 0:
        windowed_frames = windowed_frames[:, :-padding_size]
        
    if sqrt_window and np.all(window > 0):
        window = np.sqrt(window)
        windowed_frames[:, : window_size] *= window

    num_hops = windowed_frames.shape[0]    
    for i in range(num_hops):
        n_start = i * hop_size
        n_end = n_start + window_size
        signal[n_start : n_end] += windowed_frames[i]
    return signal

def get_fft_bin_phase_frequencies(num_bins, hop_size, sample_rate):
    bin_frequencies = np.linspace(0, sample_rate // 2, num=num_bins, endpoint=True)
    bin_phases = np.linspace(0, np.pi * hop_size, num=num_bins, endpoint=True)
    return bin_frequencies, bin_phases

def get_feature_diffs(feature_frames):
    feature_diff_frames = np.vstack((feature_frames[0], np.diff(feature_frames, axis=0)))
    assert(np.allclose(feature_frames.shape, feature_diff_frames.shape))
    return feature_diff_frames

def get_feature_cumsum(feature_diff_frames):
    feature_frames = np.cumsum(feature_diff_frames, axis=0)
    assert(np.allclose(feature_frames.shape, feature_diff_frames.shape))
    return feature_frames

def get_normalized_phase(phase_frames):
    phase_norm_frames = phase_frames / np.pi
    assert(np.all(phase_norm_frames >= -1.0))
    assert(np.all(phase_norm_frames <= 1.0))
    return phase_norm_frames

def get_denormalized_phase(phase_norm_frames):
    phase_frames = phase_norm_frames * np.pi
    assert(np.all(phase_frames >= -np.pi))
    assert(np.all(phase_frames <= np.pi))
    return phase_frames

def magnitude_to_db(magnitude_frames, silence_db):
    magnitude_db_frames = np.ones_like(magnitude_frames) * silence_db
    silence_lin = pow(10.0, silence_db / 20.0)
    valid_indices = np.where(magnitude_frames > silence_lin)
    magnitude_db_frames[valid_indices] = 20.0 * np.log10(magnitude_frames[valid_indices])
    return magnitude_db_frames

def db_to_magnitude(magnitude_db_frames, silence_db):
    magnitude_frames = np.zeros_like(magnitude_db_frames)
    valid_indices = np.where(magnitude_db_frames > silence_db)
    magnitude_frames[valid_indices] = pow(10.0, magnitude_db_frames[valid_indices] / 20.0)
    return magnitude_frames

def stft_analysis(windowed_frames, config):
    spectral_frames = np.fft.rfft(windowed_frames, axis=1)
    magnitude_frames = np.abs(spectral_frames)
    phase_frames = np.angle(spectral_frames)
    
    feature_names = config['features']
    feature_frames = {}
    
    if 'magnitude' in feature_names:
        feature_frames['magnitude'] = magnitude_frames
    if 'magnitude_diff' in feature_names:
        feature_frames['magnitude_diff'] = get_feature_diffs(magnitude_frames)
    if 'phase_norm' in feature_names:
        feature_frames['phase_norm'] = get_normalized_phase(phase_frames)
    if 'phase_norm_diff' in feature_names:
        feature_frames['phase_norm_diff'] = get_feature_diffs(get_normalized_phase(phase_frames))
    return feature_frames

def stft_synthesis(feature_frames, config):
    feature_names = config['features']
    
    spectral_frames = None
    magnitude_frames = None
    phase_frames = None
    
    if 'magnitude' in feature_names:
        magnitude_frames = feature_frames['magnitude']
    elif 'magnitude_diff' in feature_names:
        magnitude_frames = get_feature_cumsum(feature_frames['magnitude_diff'])
    else:
        raise ValueError('No magnitude data for synthesis!')
    if 'phase_norm' in feature_names:
        phase_frames = get_denormalized_phase(feature_frames['phase_norm'])
    elif 'phase_norm_diff' in feature_names:
        phase_frames = get_denormalized_phase(get_feature_cumsum(feature_frames['phase_norm_diff']))
    else:
        raise ValueError('No phase data for synthesis!')
        
    spectral_frames = magnitude_frames * np.exp(phase_frames * 1j)
    windowed_frames = np.fft.irfft(spectral_frames, axis=1)
    
    return windowed_frames
    