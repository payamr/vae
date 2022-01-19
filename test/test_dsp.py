import unittest
import numpy as np
import dsp.dsp as dsp

class TestComputeColaWindow(unittest.TestCase):
    def test_cola_windows(self):
        for window_type in dsp.window_types:
            for window_size in dsp.window_sizes:
                config = {
                    'window_type': window_type,
                    'window_size': window_size
                }
                window, hop_size = dsp.compute_cola_window(config)
                center_idx = (window_size - 1) // 2
                if window_type == 'boxcar':
                    self.assertEqual(np.sum(window), window_size)
                    self.assertEqual(hop_size, window_size)
                else:
                    # hop size should be greater than 0 and less than window size
                    self.assertGreater(hop_size, 0)
                    self.assertLess(hop_size, window_size)

                    self.assertEqual(window[center_idx], max(window))
                    self.assertEqual(window[-1], 0.0)
                    
                    # ignore trailing zero
                    window = window[:-1]

                    # windows should be symmetric
                    for i in range(center_idx):
                        self.assertAlmostEqual(window[i], window[-i - 1])

class TestWindowedFrameAnalysisSynthesis(unittest.TestCase):
    def setUp(self):
        self.signal = 2.0 * (np.random.random(12000) - 0.5)
        self.tolerances = {
            'boxcar': 1e-7,
            'bartlett': 1e-7,
            'hann': 1e-7,
            'blackman': 1e-3,
            'blackmanharris': 1e-3
        }
        self.configs = []
        for padding_factor in [0, 1]:
            for zero_phase in [False, True]:
                for window_type in dsp.window_types:
                    for window_size in dsp.window_sizes:
                        self.configs.append({
                            'window_type': window_type,
                            'window_size': window_size,
                            'padding_factor': padding_factor,
                            'zero_phase': zero_phase,
                            'sqrt_window': False,
                            'return_padded_signal': True
                        })

    def test_windowed_frames(self):
        for config in self.configs:
            window_type = config['window_type']
            window_size = config['window_size']
            window, hop_size = dsp.compute_cola_window(config)
            config['window'] = window
            config['hop_size'] = hop_size
            windowed_frame_data = dsp.windowed_frame_analysis(self.signal, config)

            self.assertIn('padded_signal', windowed_frame_data)
            padded_signal = windowed_frame_data['padded_signal']
            self.assertIn('padded_duration_samp', windowed_frame_data)
            padded_duration_samp = windowed_frame_data['padded_duration_samp']
            self.assertIn('frame_size', windowed_frame_data)
            
            self.assertGreaterEqual(len(padded_signal), len(self.signal))
            self.assertEqual(len(padded_signal), padded_duration_samp)

            reconstructed_signal = dsp.windowed_frame_synthesis(windowed_frame_data, config)
            self.assertEqual(len(reconstructed_signal), len(padded_signal))

            offset = 0 if window_type == 'boxcar' else window_size
            padded_signal = padded_signal[offset : padded_duration_samp - offset]
            reconstructed_signal = reconstructed_signal[offset : padded_duration_samp - offset]
            success = np.allclose(padded_signal, reconstructed_signal, atol=self.tolerances[window_type])
            self.assertTrue(success)

class TestFeatureDiffCumsum(unittest.TestCase):
    def setUp(self):
        frames = 10
        features = 21
        self.feature_frames = np.random.random(frames * features).reshape((frames, features))

    def test_feature_diff_cumsum(self):
        feature_diff_frames = dsp.get_feature_diffs(self.feature_frames)
        self.assertEqual(self.feature_frames.shape, feature_diff_frames.shape)

        feature_cumsum_frames = dsp.get_feature_cumsum(feature_diff_frames)
        self.assertEqual(self.feature_frames.shape, feature_cumsum_frames.shape)

        success = np.allclose(self.feature_frames, feature_cumsum_frames)
        self.assertTrue(success)

class TestStftFeatures(unittest.TestCase):
    def setUp(self):
        signal = 2.0 * (np.random.random(2400) - 0.5)
        windowed_frame_config = {
            'window_type': 'hann',
            'window_size': 128,
            'padding_factor': 1,
            'zero_phase': True,
            'sqrt_window': False,
            'return_padded_signal': False
        }
        window, hop_size = dsp.compute_cola_window(windowed_frame_config)
        windowed_frame_config['window'] = window
        windowed_frame_config['hop_size'] = hop_size
        windowed_frame_data = dsp.windowed_frame_analysis(signal, windowed_frame_config)
        self.windowed_frames = windowed_frame_data['frames']
        self.expected_num_bins = windowed_frame_data['frame_size'] // 2 + 1
        sample_rate = 8000
        self.configs = []
        for features in [('spectrum_norm'), ('magnitude_norm', 'phase_norm'), ('magnitude_norm_diff', 'phase_norm_diff')]:
            self.configs.append({
                'sample_rate': sample_rate,
                'hop_size': hop_size,
                'features': features
            })

    def test_stft_features(self):
        for config in self.configs: 
            all_feature_frames = dsp.stft_analysis(self.windowed_frames, config)
            for feature_name in all_feature_frames:
                feature_frames = all_feature_frames[feature_name]
                self.assertEqual(feature_frames.shape[0], self.windowed_frames.shape[0])
                self.assertEqual(feature_frames.shape[1], self.expected_num_bins)
            
            reconstructed_windowed_frames = dsp.stft_synthesis(all_feature_frames, config)
            self.assertEqual(self.windowed_frames.shape, reconstructed_windowed_frames.shape)

            success = np.allclose(self.windowed_frames, reconstructed_windowed_frames)
            self.assertTrue(success)
    
if __name__ == '__main__':
    unittest.main()
