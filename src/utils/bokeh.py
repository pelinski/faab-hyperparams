# with claude
import bokeh
import bokeh.plotting
import bokeh.models
import bokeh.server.server
from itertools import cycle
import threading
from collections import deque
import numpy as np
from bokeh.palettes import RdBu


class Plotter:
    def __init__(self, signal_vars, spectrogram_vars, sample_rate=44100, y_range=(-1, 1), rollover_blocks=3,
                 plot_update_delay=50, port=5006, block_size=1024, enable_spectrograms=True, max_freq_spectrogram=5000):

        self.signal_vars = signal_vars
        self.spectrogram_vars = spectrogram_vars
        self.y_range = y_range
        self.rollover_blocks = rollover_blocks
        self.plot_update_delay = plot_update_delay
        self.sample_rate = sample_rate
        self.port = port
        self.block_size = block_size
        self.enable_spectrograms = enable_spectrograms
        self.max_freq_spectrogram = max_freq_spectrogram

        # Audio data storage
        self.current_signal_data = {
            "timestamps": deque(maxlen=self.rollover_blocks * block_size),
            **{var: deque(maxlen=self.rollover_blocks * block_size) for var in signal_vars}
        }

        # Spectrogram data storage - rolling accumulation
        if self.enable_spectrograms:
            self.current_spectrogram_data = {}
            self.scale_metadata = {}

        self.data_lock = threading.Lock()
        self.new_signal_data_available = False
        self.new_spectrogram_available = False

    def update_signal_data(self, signal_data, signal_data_len):
        """Update the audio data from your callback"""
        if signal_data_len != self.block_size:
            print(
                f"Warning: Expected block size {self.block_size}, got {signal_data_len}")

        # ref timestamp is audio frame from Bela
        ref_timestamp = signal_data["ref_timestamp"]
        # timestamps in seconds
        ts = [(ref_timestamp/2 + i) /
              self.sample_rate for i in range(signal_data_len)]

        with self.data_lock:
            # Extend deques - automatic rollover when maxlen is reached
            self.current_signal_data["timestamps"].extend(ts)

            # Append new data for each variable
            for var in self.signal_vars:
                if var in signal_data:
                    self.current_signal_data[var].extend(signal_data[var])

            self.new_signal_data_available = True

    def update_spectrograms(self, spectrogram_data):
        """Update spectrogram data from MultiScaleSpectralDiff results."""
        if not self.enable_spectrograms:
            return

        with self.data_lock:

            ref_timestamp = spectrogram_data.get("ref_timestamp", None)
            t0 = ref_timestamp / self.sample_rate if ref_timestamp else 0

            for scale_key, scale_data in spectrogram_data.items():
                if scale_data is None or scale_key == "ref_timestamp":
                    continue

                # Initialize storage for this scale if needed
                if scale_key not in self.current_spectrogram_data:
                    self.current_spectrogram_data[scale_key] = {
                        'accumulated_spectrogram': None,
                        'freq_bins': None,
                        'current_time_start': t0,  # Track the current time window start
                        'frames_per_block': None
                    }

                    # Store metadata
                    self.scale_metadata[scale_key] = {
                        'frequencies': scale_data['frequencies'],
                        'window_size': scale_data['window_size'],
                        'hop_length': scale_data['hop_length']
                    }

                # Get the new spectrogram block
                # Shape: (n_frames, freq_bins)
                new_spec = scale_data['spectrogram']
                n_frames, n_freq_bins = new_spec.shape

                # Initialize or update accumulated spectrogram
                storage = self.current_spectrogram_data[scale_key]

                if storage['accumulated_spectrogram'] is None:
                    # First block - initialize with correct size
                    max_time_frames = self.rollover_blocks * n_frames
                    storage['accumulated_spectrogram'] = np.zeros(
                        (max_time_frames, n_freq_bins), dtype=np.float32)
                    storage['freq_bins'] = n_freq_bins
                    storage['frames_per_block'] = n_frames
                    storage['current_time_start'] = t0
                else:
                    # Update the time window start based on how much data we're shifting
                    hop_length = scale_data['hop_length']
                    time_per_frame = hop_length / self.sample_rate
                    storage['current_time_start'] += n_frames * time_per_frame

                # Shift existing data left and add new data on the right
                accumulated = storage['accumulated_spectrogram']
                accumulated[:-n_frames] = accumulated[n_frames:]  # Shift left
                accumulated[-n_frames:] = new_spec  # Add new block

            self.new_spectrogram_available = True

    def _prepare_spectrogram_for_bokeh(self, scale_key):
        """Prepare accumulated spectrogram for Bokeh display."""
        if scale_key not in self.current_spectrogram_data:
            return None

        storage = self.current_spectrogram_data[scale_key]
        if storage['accumulated_spectrogram'] is None:
            return None

        # Get the accumulated spectrogram
        # Shape: (total_time_frames, freq_bins)
        spectrogram = storage['accumulated_spectrogram']
        frequencies = self.scale_metadata[scale_key]['frequencies']

        # Calculate the current time axis based on the rolling window
        hop_length = self.scale_metadata[scale_key]['hop_length']
        time_per_frame = hop_length / self.sample_rate
        total_frames = spectrogram.shape[0]

        # The time axis should represent the current rolling window
        current_time_start = storage['current_time_start']
        time_duration = total_frames * time_per_frame
        time_start = current_time_start - time_duration  # Start of the visible window
        time_end = current_time_start  # End of the visible window

        # Transpose for Bokeh (freq_bins, time_frames)
        image_data = spectrogram.T

        # Limit frequency range for better visualization
        max_freq = min(self.max_freq_spectrogram, frequencies[-1])
        freq_mask = frequencies <= max_freq
        image_data_cropped = image_data[freq_mask, :]
        freq_end_cropped = frequencies[freq_mask][-1] if np.any(
            freq_mask) else max_freq

        # Calculate color scale
        abs_max = np.max(np.abs(image_data_cropped)
                         ) if image_data_cropped.size > 0 else 1
        if abs_max == 0:
            abs_max = 1

        return {
            'image': [image_data_cropped],
            'x': time_start,  # Now uses the actual current time window
            'y': frequencies[0],
            'dw': time_end - time_start,  # Width of the time window
            'dh': freq_end_cropped - frequencies[0],
            'abs_max': abs_max,
            'window_size': self.scale_metadata[scale_key]['window_size'],
        }

    def _create_bokeh_app(self):
        def app(doc):
            # Create data source for audio plots
            template = {"timestamps": [], **{var: []
                                             for var in self.signal_vars}}
            audio_source = bokeh.models.ColumnDataSource(template)

            # Create line glyphs with different colors
            colors = cycle([
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                "#bcbd22", "#17becf", "#1a55FF", "#FF1A1A"
            ])

            # Create audio plots
            signal_plots = []
            for y_var in self.signal_vars:
                p = bokeh.plotting.figure(frame_width=200, frame_height=120,
                                          x_axis_label="time (s)", y_axis_label="amplitude", title=f"{y_var}")

                if self.y_range is not None:
                    p.y_range = bokeh.models.Range1d(
                        self.y_range[0], self.y_range[1])

                p.x_range.range_padding = 0
                p.line(source=audio_source, x="timestamps", y=y_var,
                       line_color=next(colors), line_width=2)

                signal_plots.append(p)

            # Create spectrogram plots and sources
            spectrogram_plots = []
            spectrogram_sources = {}
            spectrogram_color_mappers = {}

            if self.enable_spectrograms:
                # We'll create these dynamically as spectrograms come in
                # For now, create placeholders for common scales
                scale_names = ['scale_0_128', 'scale_1_256',
                               'scale_2_512', 'scale_3_1024']

                for scale_key in scale_names:
                    # Create data source for this spectrogram
                    spec_source = bokeh.models.ColumnDataSource(
                        {'image': [], 'x': [0], 'y': [0], 'dw': [1], 'dh': [1]})
                    spectrogram_sources[scale_key] = spec_source

                    # Create spectrogram plot
                    p = bokeh.plotting.figure(frame_width=250, frame_height=120, x_axis_label="Time (s)",
                                              y_axis_label="Frequency (Hz)", tools="pan,wheel_zoom,box_zoom,reset")

                    # Create color mapper
                    color_mapper = bokeh.models.LinearColorMapper(
                        palette=RdBu[11][::-1], low=-1, high=1)
                    spectrogram_color_mappers[scale_key] = color_mapper

                    # Add image glyph
                    p.image(source=spec_source, image='image', x='x', y='y',
                            dw='dw', dh='dh', color_mapper=color_mapper, alpha=0.8)

                    # Set ranges
                    p.y_range = bokeh.models.Range1d(
                        0, self.max_freq_spectrogram)

                    # Add colorbar
                    color_bar = bokeh.models.ColorBar(
                        color_mapper=color_mapper, width=8, location=(0, 0), title="Spectral Difference")
                    p.add_layout(color_bar, 'right')

                    # Add grid
                    p.grid.grid_line_alpha = 0.3

                    spectrogram_plots.append(p)

            def arrange_layout(signal_plots, spectrogram_plots):
                # Separate sonified_mix from other signals
                regular_plots = []
                mix_plots = []

                for plot in signal_plots:
                    if "mssd_audio" in plot.title.text:
                        mix_plots.append(plot)
                    else:
                        regular_plots.append(plot)

                # Column 1: Regular audio plots in 3 rows of 4
                audio_grid_rows = []
                for row in range(3):
                    start_idx = row * 4
                    end_idx = start_idx + 4
                    row_plots = regular_plots[start_idx:end_idx] if start_idx < len(
                        regular_plots) else []
                    if row_plots:
                        # Pad row with None if needed to maintain 4 columns
                        while len(row_plots) < 4:
                            row_plots.append(None)
                        audio_grid_rows.append(row_plots)

                # Add mix plot as a separate row if it exists
                if mix_plots:
                    # Create a row with the mix plot and padding
                    mix_row = mix_plots + [None] * (4 - len(mix_plots))
                    audio_grid_rows.append(mix_row)

                audio_column = bokeh.layouts.gridplot(
                    audio_grid_rows,
                    sizing_mode="scale_width"
                ) if audio_grid_rows else None

                # Column 2: Spectrograms in 2 rows of 2
                spectrogram_column = None
                if self.enable_spectrograms and spectrogram_plots:
                    spec_grid_rows = []
                    # Row 1: First 2 spectrograms
                    if len(spectrogram_plots) >= 2:
                        spec_grid_rows.append(spectrogram_plots[:2])
                    # Row 2: Next 2 spectrograms
                    if len(spectrogram_plots) >= 4:
                        spec_grid_rows.append(spectrogram_plots[2:4])
                    elif len(spectrogram_plots) == 3:
                        # If only 3 spectrograms, put the 3rd one alone in row 2
                        spec_grid_rows.append([spectrogram_plots[2], None])

                    if spec_grid_rows:
                        spectrogram_column = bokeh.layouts.gridplot(
                            spec_grid_rows,
                            sizing_mode="scale_width"
                        )

                # Combine columns
                if audio_column and spectrogram_column:
                    layout = bokeh.layouts.row(
                        audio_column,
                        spectrogram_column,
                        sizing_mode="scale_width"
                    )
                elif audio_column:
                    layout = audio_column
                else:
                    layout = bokeh.layouts.row(
                        signal_plots, sizing_mode="scale_width")

                return layout

            def update():
                with self.data_lock:
                    # Update audio plots
                    if self.new_signal_data_available and len(self.current_signal_data["timestamps"]) > 0:
                        new_audio_data = {
                            "timestamps": list(self.current_signal_data["timestamps"]),
                            **{var: list(self.current_signal_data[var]) for var in self.signal_vars}
                        }
                        audio_source.data = new_audio_data
                        self.new_signal_data_available = False

                    # Update spectrogram plots
                    if self.enable_spectrograms and self.new_spectrogram_available:
                        for scale_key, source in spectrogram_sources.items():
                            spec_data = self._prepare_spectrogram_for_bokeh(
                                scale_key)
                            if spec_data:
                                # Update source data
                                source.data = {
                                    'image': spec_data['image'],
                                    'x': [spec_data['x']],
                                    'y': [spec_data['y']],
                                    'dw': [spec_data['dw']],
                                    'dh': [spec_data['dh']]
                                }

                                # Update color mapper range
                                abs_max = spec_data['abs_max']
                                spectrogram_color_mappers[scale_key].low = -abs_max
                                spectrogram_color_mappers[scale_key].high = abs_max

                                # Update plot title with block count
                                plot_idx = list(
                                    spectrogram_sources.keys()).index(scale_key)
                                if plot_idx < len(spectrogram_plots):
                                    window_size = spec_data['window_size']
                                    spectrogram_plots[
                                        plot_idx].title.text = f"Window {window_size}"

                        self.new_spectrogram_available = False

            # Add periodic callback
            doc.add_periodic_callback(update, self.plot_update_delay)
            layout = arrange_layout(signal_plots, spectrogram_plots)
            doc.add_root(layout)

        return app

    def start_server(self):
        """Start the Bokeh server in a separate thread"""
        def run_server():
            app = self._create_bokeh_app()
            server = bokeh.server.server.Server({'/': app}, port=self.port)
            server.start()
            print(f"Bokeh server started at http://localhost:{self.port}")
            server.io_loop.start()

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        return server_thread
