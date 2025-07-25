import bokeh
import bokeh.plotting
import bokeh.models
import bokeh.server.server
from itertools import cycle
import threading


class AudioDataPlotter:
    def __init__(self, y_vars, x_var="timestamps", sample_rate=44100, y_range=None, rollover=10000,
                 plot_update_delay=50, port=5006):
        self.y_vars = y_vars
        self.x_var = x_var
        self.y_range = y_range
        self.rollover = rollover
        self.plot_update_delay = plot_update_delay
        self.sample_rate = sample_rate
        self.port = port

        # Shared data storage
        self.current_data = {
            "timestamps": [],
            **{var: [] for var in y_vars}
        }
        self.data_lock = threading.Lock()
        self.t_0 = None

    def update_data(self, data, data_len):
        """Update the data from your callback"""

        ref_timestamp = data["ref_timestamp"]

        ts = [(ref_timestamp + i)/self.sample_rate for i in range(0, data_len)]

        # with self.data_lock:

        # Add timestamp
        self.current_data["timestamps"] = ts

        # Add buffer data - assuming buffers is shape (channels, samples)
        for var in self.y_vars:
            self.current_data[var] = data[var]

        # # Apply rollover to prevent memory issues
        # if len(self.current_data["timestamps"]) > self.rollover:
        #     for key in self.current_data:
        #         self.current_data[key] = self.current_data[key][-self.rollover:]

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

                # Create two separate grids: first 8 channels and last 4 channels
                # First 8 channels in a 4x2 grid
                first_8_grid = bokeh.layouts.gridplot(
                    [plots[i:i+4] for i in range(0, min(8, len(plots)), 4)],
                    sizing_mode="scale_width"
                )

                # Last 4 channels in a 2x2 grid (if there are more than 8 channels)
                if len(plots) > 8:
                    last_4_grid = bokeh.layouts.gridplot(
                        [plots[8+i:8+i+2] for i in range(0, len(plots)-8, 2)],
                        sizing_mode="scale_width"
                    )
                    # Arrange both grids in columns
                    layout = bokeh.layouts.row(
                        first_8_grid, last_4_grid, sizing_mode="scale_width")
                else:
                    layout = first_8_grid

            def update():
                with self.data_lock:
                    if len(self.current_data["timestamps"]) > 0:
                        # Get latest data points
                        new_data = {
                            "timestamps": self.current_data["timestamps"][-1:],
                            **{var: self.current_data[var][-1:] for var in self.y_vars}
                        }
                        source.stream(new_data, self.rollover)

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
