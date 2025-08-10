import torch
import numpy as np
import argparse
import pyaudio
from collections import deque
import queue
import threading
from scipy import signal
from pybela import Streamer
from pythonosc.udp_client import SimpleUDPClient
from utils.utils import load_model, get_device, get_sorted_models, get_models_coordinates, get_models_range, calc_ratio_amplitude_variance, change_model, dc_block, interpolate_signal
from utils.bokeh import Plotter
from utils.mssd import MultiScaleSpectralDiff

# import time
# import psutil


class CallbackState:
    def __init__(self, seq_len, num_models, in_size, out_size, sample_rate, num_blocks_to_compute_avg, num_blocks_to_compute_std, hp_filter_freq, lp_filter_freq, num_of_iterations_in_this_model_check, init_ratio_rising_threshold, init_ratio_falling_threshold, threshold_leak, trigger_width, trigger_idx, osc_ip=None, osc_port=None, plotting_enabled=False, out2Bela=False, play_audio=False, audio_queue_len=10, mssd_enabled=False, model_type="timelin"):

        ### ----------------- start of params definitions ###
        self.seq_len = seq_len
        self.num_models = num_models
        self.in_size = in_size
        self.out_size = out_size
        self.sample_rate = sample_rate
        self.num_blocks_to_compute_avg, self.num_blocks_to_compute_std = num_blocks_to_compute_avg, num_blocks_to_compute_std
        self.hp_filter_freq, self.lp_filter_freq = hp_filter_freq, lp_filter_freq
        self.num_of_iterations_in_this_model_check = num_of_iterations_in_this_model_check
        self.init_ratio_rising_threshold, self.init_ratio_falling_threshold = init_ratio_rising_threshold, init_ratio_falling_threshold
        self.threshold_leak, self.trigger_width = threshold_leak, trigger_width
        self.trigger_idx = trigger_idx
        # self.running_norm = running_norm
        # self.permute_out = permute_out
        self.device = get_device()
        self.osc_ip, self.osc_port, self.osc_client = osc_ip, osc_port, None
        self.out2Bela = out2Bela
        self.play_audio = play_audio
        self.audio_queue_len = audio_queue_len
        self.mssd_enabled = mssd_enabled
        self.model_type = model_type
        self.plotting_enabled = plotting_enabled
        # self.path = path
        ### ----------------- end of params definitions ###

        print(f"Using device: {self.device}")
        print(f"Model type: {self.model_type}")
        print(
            f"\nAudio enabled: {self.play_audio}\nOSC enabled: {self.osc_ip is not None}\nPlotting enabled: {self.plotting_enabled}\nMSSD enabled: {self.mssd_enabled}\n ")

        # init osc server
        if self.osc_ip and self.osc_port:
            self.osc_client = SimpleUDPClient(self.osc_ip, self.osc_port)

        # init bokeh plotter
        if plotting_enabled:

            in_vars = ['gFaabSensor_1', 'gFaabSensor_2', 'gFaabSensor_3', 'gFaabSensor_4',
                       'gFaabSensor_5', 'gFaabSensor_6', 'gFaabSensor_7', 'gFaabSensor_8']

            out_vars = ['out_1', 'out_2', 'out_3', 'out_4']

            spec_vars = ["mssd_scale_0_128", "mssd_scale_1_256",
                         "mssd_scale_2_512", "mssd_scale_3_1024"] if self.mssd_enabled else []

            self.plotter = Plotter(
                signal_vars=[
                    *in_vars, *out_vars, "mssd_audio"] if self.mssd_enabled else [*in_vars, *out_vars],
                spectrogram_vars=spec_vars,
                rollover_blocks=3,
                plot_update_delay=50,
                sample_rate=sample_rate,
                port=5007,
                enable_spectrograms=True,
                max_freq_spectrogram=2000,
            )
            self.plotter.start_server()
        else:
            self.plotter = None
        ### ----------------- start of models init ###

        # preload models
        # models in desc order of train loss, take best 20
        if self.model_type == "timelin":
            path = "src/models/trained/transformer-autoencoder-jan"
        elif self.model_type == "timecomp":
            path = "src/models/trained/transformer-autoencoder-timecomp-jan"
        else:
            raise ValueError(
                f"Unknown model type: {self.model_type}. Use 'timelin' or 'timecomp'.")
        sorted_models = get_sorted_models(path=path, num_models=num_models)
        self.models = {}
        self.models_running_range = {}
        for _id in sorted_models:
            self.models[_id] = load_model(_id, self.model_type, path=path)
            self.models_running_range[_id] = {"min": torch.FloatTensor([0, 0, 0, 0]).to(
                self.device), "max": torch.FloatTensor([0, 0, 0, 0]).to(self.device)}

        # warm up models
        print("Warming up models...")
        for _id in sorted_models:
            # Create dummy input with same shape as real data
            dummy_input = torch.randn(
                self.seq_len, self.in_size).to(self.device)

            # Run forward pass to initialize GPU kernels
            with torch.no_grad():
                _ = self.models[_id].forward_encoder(dummy_input)

        print("Model warmup complete")

        # model space
        # min and max of model outputs (after passing the full dataset)
        self.full_dataset_models_range = get_models_range(path=path)
        # model's chosen 4 hyperparameters mapped to values between 0 and 1
        self.models_coordinates = get_models_coordinates(
            path=path, sorted=True, num_models=num_models)
        starter_id = sorted_models[-1]  # h0o65m8s is nice

        ### ----------------- end of models init ###

        ### ----------------- start of filters init ###
        self.bridge_dc_blocker = {'prev_in': 0, 'prev_out': 0}

        # input bp filters
        self.in_bp = [signal.butter(
            2, [hp_filter_freq, lp_filter_freq], 'bandpass', fs=sample_rate, output='sos') for _ in range(in_size)]
        self.in_bp_zi = [signal.sosfilt_zi(bp) for bp in self.in_bp]
        # output bp filters (each model has its own set of filters)
        self.out_bp, self.out_bp_zi = {}, {}
        for _id in sorted_models:
            self.out_bp[_id] = [signal.butter(
                2, [hp_filter_freq, lp_filter_freq], 'bandpass', fs=sample_rate, output='sos') for _ in range(out_size)]
            self.out_bp_zi[_id] = [signal.sosfilt_zi(
                bp) for bp in self.out_bp[_id]]

        ### ----------------- end of filters init ###

        ### ----------------- start of audio playback init ###
        if self.play_audio:
            def _audio_player(self):
                # Pre-fill queue with silence to prevent initial underrun
                # Stereo silence
                silence = np.zeros(1024, dtype=np.float32).tobytes()
                for _ in range(2):
                    try:
                        self.audio_queue.put_nowait(silence)
                    except queue.Full:
                        break
                while True:
                    try:
                        audio_data = self.audio_queue.get(timeout=1)
                        self.audio_stream.write(audio_data)
                        queue_size = self.audio_queue.qsize()
                        if queue_size == 0:
                            print("Audio buffer underrun")
                    except queue.Empty:
                        print("Audio queue is empty, waiting for data...")
                        continue

            # latency!!! (increase if using bokeh plotter)
            self.audio_queue = queue.Queue(maxsize=self.audio_queue_len)
            self.audio_thread = threading.Thread(
                target=_audio_player, daemon=True, args=(self,))
            self.audio = pyaudio.PyAudio()
            self.audio_stream = self.audio.open(format=pyaudio.paFloat32, channels=2, rate=int(
                sample_rate), output=True, frames_per_buffer=1024)
            self.audio_thread.start()
        ### ----------------- end of audio playback init ###

        ### mssd threading ###
        if self.mssd_enabled:
            self.mssd_scales = [
                {'window_size': 128, 'hop_length': 32},
                {'window_size': 256, 'hop_length': 64},
                {'window_size': 512, 'hop_length': 128},
                {'window_size': 1024, 'hop_length': 1024}
            ]
            self.mssd = MultiScaleSpectralDiff(
                sample_rate=sample_rate, scales=self.mssd_scales, enable_sonification=True)
            self.mssd_thread = threading.Thread(
                target=self.mssd._mssd_worker, daemon=True)
            self.mssd_thread.start()
            ### ----------------- end of mssd threading ###

        #### ----------------- start of variables for the callback ###
        self.model = self.models[starter_id]
        self.gain = 4 * [1.0]
        self.change_model = 0
        self.ratio_rising_threshold, self.ratio_falling_threshold = init_ratio_rising_threshold, init_ratio_falling_threshold
        self.iterations_in_this_model_counter = 0
        self.bridge_piezo_avg = deque(maxlen=num_blocks_to_compute_avg)
        self.model_out_std = deque(maxlen=num_blocks_to_compute_std)
        self.debug_counter = 0
        self.model_perm = [0, 1, 2, 3]
        self.prev_model = self.models[starter_id]  # for crossfading
        ### ----------------- end of variables for the callback ###
# sound_threshold =0.021


async def callback(block, cs, streamer):
    # start = time.perf_counter()
    with torch.no_grad():
        ref_timestamp = block[0]["buffer"]["ref_timestamp"]

        ### ----------------- start of input filtering ###
        # filter input data -- no normalisation because it removes the relative differences between sensors
        filtered_in = [[] for _ in range(cs.in_size)]
        for idx, var in enumerate(block):
            _in = var["buffer"]["data"]
            filtered_in[idx], cs.in_bp_zi[idx] = signal.sosfilt(
                cs.in_bp[idx], _in, zi=cs.in_bp_zi[idx])
        ### ----------------- end of input filtering ###

        ### ----------------- start of model inference ###
        data_tensor = torch.stack([torch.as_tensor(
            filtered_in[idx], dtype=torch.float32) for idx in range(cs.in_size)])  # num_features, 1024
        input = data_tensor.permute(1, 0)  # seq_len, num_features
        # input = input.unsqueeze(0).to(cs.device)  # 1, seq_len, num_features
        out = cs.model.forward_encoder(input.to(cs.device)).permute(
            1, 0)  # num_outputs, seq_len
        ### ----------------- end of model inference ###

        # needed for crossfading
        out_prev_model = None
        if cs.iterations_in_this_model_counter == 0:
            out_prev_model = cs.prev_model.forward_encoder(
                input.to(cs.device)).permute(1, 0)

        # outputs --> [ff_size, num_heads, num_layers, learning_rate]

        ### ----------------- start of output filtering ###
        filtered_out = [[] for _ in range(cs.out_size)]
        for idx in range(cs.out_size):
            _out = out[idx].cpu().numpy().tolist()  # convert to list
            if cs.model_type == "timecomp" and cs.seq_len != cs.model.comp_seq_len:  # interpolate if time compression model
                _out = interpolate_signal(_out, cs.seq_len)
            filtered_out[idx], cs.out_bp_zi[cs.model.id][idx] = signal.sosfilt(
                cs.out_bp[cs.model.id][idx], _out, zi=cs.out_bp_zi[cs.model.id][idx])
            # if this is the first iteration in this model, crossfade with previous model to avoid clicks
            if out_prev_model is not None:
                _out_prev = out_prev_model[idx].cpu().numpy().tolist()
                _out_prev = interpolate_signal(_out_prev, cs.seq_len)
                _filtered_out_prev, cs.out_bp_zi[cs.prev_model.id][idx] = signal.sosfilt(
                    cs.out_bp[cs.prev_model.id][idx], _out_prev, zi=cs.out_bp_zi[cs.prev_model.id][idx])
                fade_steps = np.linspace(0, 1, cs.seq_len)
                filtered_out[idx] = np.array(
                    filtered_out[idx]) * fade_steps + np.array(_filtered_out_prev) * (1 - fade_steps)
        filtered_out = np.array(filtered_out)
        ### ----------------- end of output filtering ###

        in_audio = data_tensor.sum(axis=0).numpy().astype(np.float32)  # in
        out_audio = filtered_out.sum(axis=0).astype(np.float32)  # out

        if cs.mssd_enabled:
            try:
                cs.mssd.mssd_queue.put_nowait((in_audio, out_audio))
            except queue.Full:
                print("MSSD queue is full, dropping audio data")
                cs.mssd.mssd_queue.get_nowait()
                cs.mssd.mssd_queue.put_nowait((in_audio, out_audio))

        if cs.audio_stream:  # could spatialise
            if cs.mssd_enabled:
                mssd_audio = cs.mssd.latest_mssd_audio
                mixed_audio = mssd_audio
            else:
                mixed_audio = out_audio
            stereo_data = np.column_stack(
                [mixed_audio, mixed_audio]).flatten().astype(np.float32)
            try:
                cs.audio_queue.put_nowait(stereo_data.tobytes())
            except queue.Full:
                print("Audio queue is full, dropping audio data")
                cs.audio_queue.get_nowait()
                cs.audio_queue.put_nowait(stereo_data.tobytes())

        if cs.plotter is not None:
            # update spectrograms
            if cs.mssd_enabled:
                try:
                    mssd_per_scale, _ = cs.mssd.mssd_results.get_nowait()
                    cs.plotter.update_spectrograms(
                        {"ref_timestamp": ref_timestamp, **mssd_per_scale})
                except queue.Empty:
                    pass

            # update signals
            plotter_signals_data = {"ref_timestamp": ref_timestamp,
                                    **{f"gFaabSensor_{i+1}": filtered_in[i] for i in range(cs.in_size)},
                                    **{f"out_{i+1}": filtered_out[i] for i in range(cs.out_size)}, **{f"mssd_audio": mixed_audio}
                                    }
            cs.plotter.update_signal_data(
                plotter_signals_data, signal_data_len=cs.seq_len)

        # # permute output
        # if cs.permute_out:
        #     filtered_out = filtered_out[cs.model_perm]

        # # send each feature to Bela or OSC
        for idx in range(cs.out_size):
            if cs.osc_client:
                cs.osc_client.send_message(
                    f'/f{idx+1}', filtered_out[f"out_{idx}"])
            if cs.out2Bela:
                streamer.send_buffer(idx, 'f', cs.seq_len,
                                     filtered_out[f"out_{idx}"])

        bridge_piezo = dc_block(
            block[5]["buffer"]["data"], cs.bridge_dc_blocker)

        ratio = calc_ratio_amplitude_variance(
            filtered_out, bridge_piezo, cs)

        # -- model change --
        past_change_model = cs.change_model

        # leaky threshold
        if cs.iterations_in_this_model_counter % cs.num_of_iterations_in_this_model_check == 0:
            cs.ratio_rising_threshold -= cs.threshold_leak
            cs.ratio_falling_threshold += cs.threshold_leak

        cs.iterations_in_this_model_counter += 1
        # is it time to change model?
        if (ratio > cs.ratio_rising_threshold and past_change_model == 0):
            cs.change_model = 1
        elif (ratio < cs.ratio_falling_threshold and past_change_model == 1):
            cs.change_model = 0

        cs.debug_counter += 1  # for debugging

        # if it's time to change model...
        if (past_change_model == 0 and cs.change_model == 1):
            change_model(filtered_out, cs)

            # #  debug audio callback diagnostics
            # if cs.debug_counter % 50 == 0:
            #     elapsed = time.perf_counter() - start

            #     target = 1024 / (cs.sample_rate)  # Your block period
            #     utilization = (elapsed / target) * 100
            #     print(f"Callback: {elapsed*1000:.1f}ms ({utilization:.1f}% CPU)")
            #     # Check system memory
            #     ram_percent = psutil.virtual_memory().percent

            #     # Check GPU memory if using CUDA
            #     if torch.cuda.is_available():
            #         gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
            #         print(
            #             f"RAM: {ram_percent}%, GPU: {gpu_memory:.1f}MB, Queue: {cs.audio_queue.qsize()}")

            #     # Check audio queue size
            #     print(f"Audio queue size: {cs.audio_queue.qsize()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="faab-callback")
    parser.add_argument('--osc', action='store_true', help='Use OSC server')
    parser.add_argument('--plot', action='store_true',
                        help='Enable Bokeh plotting', default=False)
    parser.add_argument('--out2Bela', action='store_true',
                        help='Send output to Bela', default=False)
    parser.add_argument('--audio', action='store_true',
                        help="Play model output as audio", default=True)
    parser.add_argument('--mssd', action='store_true',
                        help="Enable MSSD processing", default=False)
    parser.add_argument('--model_type', type=str, choices=['timelin', 'timecomp'], default='timelin',
                        help="Type of model to use: 'timelin' for TransformerAutoencoder or 'timecomp' for TransformerTimeAutoencoder")
    args = parser.parse_args()

    if args.osc:
        osc_ip, osc_port = "127.0.0.1", 2222
    else:
        osc_ip, osc_port = None, None

    # streamer = Streamer(ip="192.168.0.199")
    streamer = Streamer()
    streamer.connect()

    sample_rate = streamer.sample_rate / 2  # analog rate is half audio rate

    cs = CallbackState(
        seq_len=1024,  # 1024 samples per block of data received from Bela, also seq_len for the model
        num_models=20,  # number of models to use
        in_size=8,  # number of input variables
        out_size=4,  # number of output variables
        sample_rate=sample_rate,  # Bela sample rate and sample rate we use in Python
        # number of blocks to compute average amplitude for the bridge piezo
        num_blocks_to_compute_avg=10,
        # number of blocks to compute standard deviation for the model output
        num_blocks_to_compute_std=10,
        # high-pass filter frequency for model input and output. 1Hz to remove DC offset
        hp_filter_freq=1,
        lp_filter_freq=5000,  # low-pass filter frequency for model input and output
        # min number of model iterations (forward-passes of blocks of seq_len) before checking if model should change
        num_of_iterations_in_this_model_check=10,
        # initial ratio (avg amplitude / std) schmidt trigger rising threshold for model change
        init_ratio_rising_threshold=30,
        # initial ratio (avg amplitude / std) schmidt trigger falling threshold for model change
        init_ratio_falling_threshold=10,
        threshold_leak=1,  # amount to leak the threshold over time
        trigger_width=25,  # width of the trigger in samples
        trigger_idx=4,  # index of the trigger output
        osc_ip=osc_ip,  # OSC server IP
        osc_port=osc_port,  # OSC server port
        plotting_enabled=args.plot,  # enable Bokeh plotting
        out2Bela=args.out2Bela,
        play_audio=args.audio,
        audio_queue_len=10 if args.plot else 2,  # 10 if using bokeh
        mssd_enabled=args.mssd,
        model_type=args.model_type
        # running_norm=True,  # whether to use running normalization
        # permute_out=False, # permute model output dimensions
        # path="src/models/trained/transformer-autoencoder-jan",
    )
    in_vars = ['gFaabSensor_1', 'gFaabSensor_2', 'gFaabSensor_3', 'gFaabSensor_4',
               'gFaabSensor_5', 'gFaabSensor_6', 'gFaabSensor_7', 'gFaabSensor_8']
    streamer.start_streaming(
        in_vars, on_block_callback=callback, callback_args=(cs, streamer))

    streamer.wait(0)
