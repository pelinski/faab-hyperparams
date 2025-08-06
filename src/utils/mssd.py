# with claude
import numpy as np
import queue


class MultiScaleSpectralDiff:
    """
    Simple multi-scale spectral difference calculator with sonification.
    Creates difference spectrograms for each STFT scale and can sonify them.
    """

    def __init__(self, sample_rate=44100, scales=None, enable_sonification=False, target_block_size=1024):
        self.sample_rate = sample_rate
        self.enable_sonification = enable_sonification
        self.target_block_size = target_block_size
        self.scales = scales if scales else [
            {'window_size': 128, 'hop_length': 32},
            {'window_size': 256, 'hop_length': 64},
            {'window_size': 512, 'hop_length': 128},
            {'window_size': 1024, 'hop_length': 1024}
        ]

        if self.enable_sonification:
            self.sonifiers = [BlockSonifier(sample_rate=sample_rate, target_block_size=target_block_size)
                              for _ in range(len(self.scales))]

        self.mssd_queue = queue.Queue(maxsize=5)
        self.mssd_results = queue.Queue(maxsize=5)
        self.latest_mssd_audio = np.zeros(target_block_size, dtype=np.float32)

    def _mssd_worker(self):
        """Worker thread to process audio blocks and compute MSSD."""
        while True:
            try:
                in_audio, out_audio = self.mssd_queue.get(timeout=1)
                mssd_per_scale = self.process_block(in_audio, out_audio)
                mixed_audio = self.get_audio(mssd_per_scale)

                self.latest_mssd_audio = mixed_audio if mixed_audio is not None else np.zeros(
                    self.target_block_size, dtype=np.float32)

                try:
                    self.mssd_results.put_nowait((mssd_per_scale, mixed_audio))
                except queue.Full:
                    pass  # Silent drop
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in MSSD worker: {e}")
                continue

    def process_block(self, audio1_block, audio2_block):
        """Process audio block and return difference spectrograms for each scale."""
        audio1, audio2 = np.array(audio1_block, dtype=np.float32), np.array(
            audio2_block, dtype=np.float32)

        assert len(audio1) == len(
            audio2), "Audio blocks must be the same length"

        mssd = {}
        for i, scale in enumerate(self.scales):
            window_size = scale['window_size']
            hop_length = scale['hop_length']
            scale_key = f'scale_{i}_{window_size}'

            mssd[scale_key] = self._compute_difference_spectrogram(
                audio1, audio2, window_size, hop_length)

            if self.enable_sonification and mssd[scale_key] is not None:
                mssd[scale_key]['audio'] = self.sonifiers[i].sonify_block(
                    mssd[scale_key])

        return mssd

    def _compute_difference_spectrogram(self, audio1, audio2, window_size, hop_length):
        """Compute difference spectrogram for a single scale."""
        block_size = len(audio1)
        window = np.hanning(window_size).astype(np.float32)
        freqs = np.fft.rfftfreq(window_size, 1/self.sample_rate)
        n_frames = (block_size - window_size) // hop_length + 1

        if n_frames <= 0:
            return None

        diff_spectrogram = np.zeros((n_frames, len(freqs)), dtype=np.float32)

        for frame_idx in range(n_frames):
            start_idx = frame_idx * hop_length
            end_idx = start_idx + window_size

            windowed_1 = audio1[start_idx:end_idx] * window
            windowed_2 = audio2[start_idx:end_idx] * window

            stft_1 = np.fft.rfft(windowed_1)
            stft_2 = np.fft.rfft(windowed_2)

            mag_diff = np.abs(stft_1) - np.abs(stft_2)
            mag_diff = np.abs(mag_diff)  # Use absolute value for difference
            diff_spectrogram[frame_idx] = mag_diff

        return {
            'spectrogram': diff_spectrogram,
            'frequencies': freqs,
            'window_size': window_size,
            'hop_length': hop_length,
        }

    def get_audio(self, results):
        if not self.enable_sonification:
            return None

        mixed_audio = np.zeros(self.target_block_size, dtype=np.float32)

        for _, scale_data in results.items():
            if scale_data is not None and 'audio' in scale_data and scale_data['audio'] is not None:
                audio = scale_data['audio']
                mixed_audio += audio * 0.25  # Scale down each layer to prevent clipping

        # Normalize mixed audio to prevent clipping
        max_val = np.max(np.abs(mixed_audio))
        if max_val > 0:
            mixed_audio = mixed_audio / max_val * 0.5  # Conservative scaling

        return mixed_audio


class BlockSonifier:
    """Block-based sonification that outputs exactly 1024 samples per block."""

    def __init__(self, sample_rate, target_block_size=1024):
        self.sample_rate = sample_rate
        self.target_block_size = target_block_size

        # Phase accumulators for coherent synthesis
        self.phase_accumulators = None
        self.frequencies = None
        self.phase_increments = None

    def sonify_block(self, diff_spec_data, amplitude_scale=0.01):  # Reduced amplitude scale
        """
        Convert entire spectrogram block to exactly target_block_size samples.
        Fixed version that reduces noise and artifacts.
        """
        if diff_spec_data is None:
            return np.zeros(self.target_block_size, dtype=np.float32)

        spectrogram = diff_spec_data['spectrogram']
        frequencies = diff_spec_data['frequencies']

        n_frames, n_freq_bins = spectrogram.shape

        # Initialize phase accumulators if needed
        if self.phase_accumulators is None:
            self.frequencies = frequencies
            # Use fewer frequency bins to reduce noise
            # Much fewer frequencies
            max_freq_idx = min(len(frequencies) // 4, 64)
            self.phase_accumulators = np.random.rand(max_freq_idx) * 2 * np.pi
            self.phase_increments = 2 * np.pi * \
                frequencies[:max_freq_idx] / self.sample_rate

        max_freq_idx = len(self.phase_accumulators)
        output_audio = np.zeros(self.target_block_size, dtype=np.float32)

        # Calculate time positions for each frame
        hop_length = diff_spec_data.get('hop_length', 128)
        time_per_frame = hop_length / self.sample_rate

        # For each frequency bin, create a continuous time-varying amplitude envelope
        for freq_idx in range(max_freq_idx):
            if freq_idx >= n_freq_bins:
                continue

            # Get amplitude envelope for this frequency across all frames
            amplitude_envelope = spectrogram[:, freq_idx] * amplitude_scale

            # Skip if all amplitudes are very small
            if np.max(amplitude_envelope) < 1e-4:
                continue

            # Create interpolated amplitude envelope for the full block
            frame_positions = np.arange(
                n_frames) * time_per_frame * self.sample_rate
            sample_positions = np.arange(self.target_block_size)

            # Interpolate amplitude envelope to match sample positions
            if n_frames > 1:
                interp_amplitudes = np.interp(
                    sample_positions, frame_positions, amplitude_envelope)
            else:
                interp_amplitudes = np.full(
                    self.target_block_size, amplitude_envelope[0])

            # Apply smoothing to reduce artifacts
            # Simple moving average to smooth amplitude changes
            kernel_size = min(32, self.target_block_size // 10)
            if kernel_size > 1:
                kernel = np.ones(kernel_size) / kernel_size
                interp_amplitudes = np.convolve(
                    interp_amplitudes, kernel, mode='same')

            # Generate continuous sine wave
            phase_start = self.phase_accumulators[freq_idx]
            sample_indices = np.arange(self.target_block_size)
            phases = phase_start + \
                self.phase_increments[freq_idx] * sample_indices

            # Apply amplitude envelope to sine wave
            sine_wave = interp_amplitudes * np.sin(phases)

            # Apply fade-in/fade-out to reduce block boundary artifacts
            fade_samples = min(64, self.target_block_size // 20)
            if fade_samples > 0:
                fade_in = np.linspace(0, 1, fade_samples)
                fade_out = np.linspace(1, 0, fade_samples)
                sine_wave[:fade_samples] *= fade_in
                sine_wave[-fade_samples:] *= fade_out

            output_audio += sine_wave

            # Update phase accumulator for continuity
            self.phase_accumulators[freq_idx] = (phase_start +
                                                 self.phase_increments[freq_idx] * self.target_block_size) % (2 * np.pi)

        # Apply overall smoothing and limiting
        max_val = np.max(np.abs(output_audio))
        if max_val > 0:
            # Soft limiting to prevent harsh clipping
            output_audio = np.tanh(output_audio / max_val) * 0.5

        return output_audio.astype(np.float32)
