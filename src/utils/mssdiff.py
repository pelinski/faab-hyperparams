import numpy as np


class MultiScaleSpectralDiff:
    """
    Simple multi-scale spectral difference calculator.
    Creates difference spectrograms for each STFT scale.
    """

    def __init__(self, sample_rate=44100, scales=None):
        self.sample_rate = sample_rate

        # Default scales for 1024-sample blocks
        if scales is None:
            self.scales = [
                {'window_size': 128, 'hop_length': 32},
                {'window_size': 256, 'hop_length': 64},
                {'window_size': 512, 'hop_length': 128},
                {'window_size': 1024, 'hop_length': 256}
            ]
        else:
            self.scales = scales

    def process_block(self, audio1_block, audio2_block):
        """Process audio block and return difference spectrograms for each scale."""
        audio1, audio2 = np.array(audio1_block, dtype=np.float32), np.array(
            audio2_block, dtype=np.float32)

        block_size = len(audio1)

        results = {}
        for i, scale in enumerate(self.scales):
            window_size = scale['window_size']
            hop_length = scale['hop_length']
            scale_key = f'scale_{i}_{window_size}'

            # Skip if window larger than block
            if window_size > block_size:
                continue

            # Calculate frames and create difference spectrogram
            diff_spec = self._compute_difference_spectrogram(
                audio1, audio2, window_size, hop_length
            )

            results[scale_key] = diff_spec

        return results

    def _compute_difference_spectrogram(self, audio1, audio2, window_size, hop_length):
        """Compute difference spectrogram for a single scale."""
        block_size = len(audio1)

        # Pre-compute window and frequency bins
        window = np.hanning(window_size).astype(np.float32)
        freqs = np.fft.rfftfreq(window_size, 1/self.sample_rate)

        # Calculate number of frames
        n_frames = (block_size - window_size) // hop_length + 1

        if n_frames <= 0:
            return None

        # Pre-allocate spectrogram array
        diff_spectrogram = np.zeros((n_frames, len(freqs)), dtype=np.float32)

        # Compute STFT difference for each frame
        for frame_idx in range(n_frames):
            start_idx = frame_idx * hop_length
            end_idx = start_idx + window_size

            # Window the audio segments
            windowed_1 = audio1[start_idx:end_idx] * window
            windowed_2 = audio2[start_idx:end_idx] * window

            # Compute STFT and magnitude difference
            stft_1 = np.fft.rfft(windowed_1)
            stft_2 = np.fft.rfft(windowed_2)

            mag_diff = np.abs(stft_1) - np.abs(stft_2)
            diff_spectrogram[frame_idx] = mag_diff

            # FIXME should be absolute value?

        return {
            'spectrogram': diff_spectrogram,  # Shape: (n_frames, freq_bins)
            'frequencies': freqs,
            'window_size': window_size,
            'hop_length': hop_length,
        }
