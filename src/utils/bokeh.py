import bokeh
import bokeh.plotting
import bokeh.models
import bokeh.server.server
from itertools import cycle
import threading
from collections import deque


class AudioDataPlotter:
    def __init__(self, y_vars, x_var="timestamps", sample_rate=44100, y_range=None, rollover=10000,
                 plot_update_delay=50, port=5006, block_size=1024):
        self.y_vars = y_vars
        self.x_var = x_var
        self.y_range = y_range
        self.rollover = rollover
        self.plot_update_delay = plot_update_delay
        self.sample_rate = sample_rate
        self.port = port
        self.block_size = block_size

        self.current_data = {
            "timestamps": deque(maxlen=rollover),
            **{var: deque(maxlen=rollover) for var in y_vars}
        }

        self.data_lock = threading.Lock()
        self.new_data_available = False

    def update_data(self, data, data_len):
        """Update the data from your callback"""
        if data_len != self.block_size:
            print(
                f"Warning: Expected block size {self.block_size}, got {data_len}")

        # ref timestamp is audio frame from Bela
        ref_timestamp = data["ref_timestamp"]
        # timestamps in seconds
        ts = [(ref_timestamp/2 + i) / self.sample_rate for i in range(data_len)]

        with self.data_lock:
            # Extend deques - automatic rollover when maxlen is reached
            self.current_data["timestamps"].extend(ts)

            # Append new data for each variable
            for var in self.y_vars:
                if var in data:
                    self.current_data[var].extend(data[var])

            self.new_data_available = True

    def _create_bokeh_app(self):
        def app(doc):
            # Create data source
            template = {"timestamps": [], **{var: [] for var in self.y_vars}}
            source = bokeh.models.ColumnDataSource(template)

            # Create line glyphs with different colors
            colors = cycle([
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                "#bcbd22", "#17becf", "#1a55FF", "#FF1A1A"
            ])

            plots = []
            for y_var in self.y_vars:
                # Create figure
                p = bokeh.plotting.figure(
                    frame_width=250,
                    frame_height=120,
                    x_axis_label="time (s)",
                    y_axis_label="amplitude",
                    title=f"{y_var}")

                if self.y_range is not None:
                    p.y_range = bokeh.models.Range1d(
                        self.y_range[0], self.y_range[1])

                p.x_range.range_padding = 0
                p.line(source=source, x="timestamps", y=y_var,
                       line_color=next(colors), line_width=2)

                plots.append(p)

            # Create layout based on number of plots
            if len(plots) <= 8:
                # First 8 channels in a 4x2 grid
                grid = bokeh.layouts.gridplot(
                    [plots[i:i+4] for i in range(0, len(plots), 4)],
                    sizing_mode="scale_width"
                )
                layout = grid
            else:
                # First 8 channels in a 4x2 grid
                first_8_grid = bokeh.layouts.gridplot(
                    [plots[i:i+4] for i in range(0, 8, 4)],
                    sizing_mode="scale_width"
                )

                # Remaining channels in a 2x2 grid
                remaining_plots = plots[8:]
                remaining_grid = bokeh.layouts.gridplot(
                    [remaining_plots[i:i+2]
                        for i in range(0, len(remaining_plots), 2)],
                    sizing_mode="scale_width"
                )

                # Arrange both grids in columns
                layout = bokeh.layouts.row(
                    first_8_grid, remaining_grid, sizing_mode="scale_width")

            def update():
                with self.data_lock:
                    if self.new_data_available and len(self.current_data["timestamps"]) > 0:
                        # Convert deques to lists for Bokeh
                        new_data = {
                            "timestamps": list(self.current_data["timestamps"]),
                            **{var: list(self.current_data[var]) for var in self.y_vars}
                        }
                        source.data = new_data
                        self.new_data_available = False

            # Add periodic callback
            doc.add_periodic_callback(update, self.plot_update_delay)
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
