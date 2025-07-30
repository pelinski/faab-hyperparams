import torch
import numpy as np
import argparse
import pyaudio
from collections import deque
import queue
import threading
from pybela import Streamer
from pythonosc.udp_client import SimpleUDPClient
from utils.utils import load_model, get_device, get_sorted_models, get_models_coordinates, get_models_range, calc_ratio_amplitude_variance, change_model, dc_block
from utils.bokeh import AudioDataPlotter
from scipy import signal
import time
import psutil
import sys


class CallbackState:
    def __init__(self, seq_len, num_models, in_size, out_size, sample_rate, num_blocks_to_compute_avg, num_blocks_to_compute_std, hp_filter_freq, lp_filter_freq, envelope_len, num_of_iterations_in_this_model_check, init_ratio_rising_threshold, init_ratio_falling_threshold, threshold_leak, trigger_width, trigger_idx, running_norm, permute_out, path, osc_ip=None, osc_port=None, plotter=None, out2Bela=False, play_audio=False):

        # -- params --
        self.seq_len = seq_len
        self.num_models = num_models
        self.in_size = in_size
        self.out_size = out_size
        self.sample_rate = sample_rate
        self.num_blocks_to_compute_avg, self.num_blocks_to_compute_std = num_blocks_to_compute_avg, num_blocks_to_compute_std
        self.hp_filter_freq, self.lp_filter_freq = hp_filter_freq, lp_filter_freq
        self.envelope_len = envelope_len  # for envelopes when changing model
        self.num_of_iterations_in_this_model_check = num_of_iterations_in_this_model_check
        self.init_ratio_rising_threshold, self.init_ratio_falling_threshold = init_ratio_rising_threshold, init_ratio_falling_threshold
        self.threshold_leak, self.trigger_width = threshold_leak, trigger_width
        self.trigger_idx = trigger_idx
        self.running_norm = running_norm
        self.permute_out = permute_out
        self.device = get_device()
        self.path = path
        self.osc_ip, self.osc_port, self.osc_client = osc_ip, osc_port, None
        self.plotter = plotter
        self.out2Bela = out2Bela
        self.play_audio = play_audio

        print(f"Using device: {self.device}, python version {sys.version}")

        # init osc server
        if self.osc_ip and self.osc_port:
            self.osc_client = SimpleUDPClient(self.osc_ip, self.osc_port)

        # init models

        # preload models
        # models in desc order of train loss, take best 20
        sorted_models = get_sorted_models(path=path, num_models=num_models)
        self.models = {}
        self.models_running_range = {}
        for _id in sorted_models:
            self.models[_id] = load_model(_id, path=path)
            self.models_running_range[_id] = {"min": torch.FloatTensor([0, 0, 0, 0]).to(
                self.device), "max": torch.FloatTensor([0, 0, 0, 0]).to(self.device)}

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

        self.bridge_dc_blocker = {'prev_in': 0, 'prev_out': 0}

        # self.out_dc_blocker = {'prev_in': 0, 'prev_out': 0}
        self.in_bp = [signal.butter(
            2, [hp_filter_freq, lp_filter_freq], 'bandpass', fs=sample_rate, output='sos') for _ in range(in_size)]
        self.in_bp_zi = [signal.sosfilt_zi(bp) for bp in self.in_bp]
        self.out_bp = [signal.butter(
            2, [hp_filter_freq, lp_filter_freq], 'bandpass', fs=sample_rate, output='sos') for _ in range(out_size)]
        self.out_bp_zi = [signal.sosfilt_zi(bp) for bp in self.out_bp]

        # envelope
        self.envelope_len = envelope_len

        # variables for the callback
        self.model = self.models[starter_id]
        self.gain = 4 * [1.0]
        self.change_model = 0
        self.ratio_rising_threshold, self.ratio_falling_threshold = init_ratio_rising_threshold, init_ratio_falling_threshold
        self.iterations_in_this_model_counter = 0
        self.bridge_piezo_avg = deque(maxlen=num_blocks_to_compute_avg)
        self.model_out_std = deque(maxlen=num_blocks_to_compute_std)
        self.debug_counter = 0
        self.model_perm = [0, 1, 2, 3]

        # audio output
        if self.play_audio:
            def _audio_player(self):
                # Pre-fill queue with silence to prevent initial underrun
                # Stereo silence
                silence = np.zeros(1024, dtype=np.float32).tobytes()
                for _ in range(10):  # Add 10 blocks of silence
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
                            print("Audio buffer underrun - may cause clicks")
                    except queue.Empty:
                        print("Audio queue is empty, waiting for data...")
                        continue

            self.audio_queue = queue.Queue(maxsize=10)
            self.audio_thread = threading.Thread(
                target=_audio_player, daemon=True, args=(self,))
            self.audio = pyaudio.PyAudio()
            self.audio_stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=int(sample_rate),
                output=True,
                frames_per_buffer=1024)
            self.audio_thread.start()


# sound_threshold =0.021


async def callback(block, cs, streamer):
    start = time.perf_counter()
    with torch.no_grad():
        ref_timestamp = block[0]["buffer"]["ref_timestamp"]

        # filter input data -- no normalisation because it removes the relative differences between sensors
        filtered_in = {var["name"]: [] for var in block}
        for idx, var in enumerate(block):
            filtered_in[var["name"]], cs.in_bp_zi[idx] = signal.sosfilt(
                cs.in_bp[idx], var["buffer"]["data"], zi=cs.in_bp_zi[idx])

        data_tensor = torch.stack([torch.as_tensor(
            filtered_in[var["name"]], dtype=torch.float32) for var in block])  # num_features, 1024
        input = data_tensor.permute(1, 0)  # seq_len, num_features

        cs.iterations_in_this_model_counter += 1
        out = cs.model.forward_encoder(input.to(cs.device)).permute(
            1, 0)  # num_outputs, seq_len
        # outputs --> [ff_size, num_heads, num_layers, learning_rate]

        # filtered out
        filtered_out = [[] for _ in range(cs.out_size)]
        for idx in range(cs.out_size):
            _out = out[idx].cpu().numpy().tolist()  # convert to list
            filtered_out[idx], cs.out_bp_zi[idx] = signal.sosfilt(
                cs.out_bp[idx], _out, zi=cs.out_bp_zi[idx])

        if cs.audio_stream:  # could spatialise
            raw_input_sum = np.array([var["buffer"]["data"]
                                      for var in block]).sum(axis=0).astype(np.float32) / cs.in_size  # sum buffers

            # left_ch = filtered_out.sum(
            #     axis=0).astype(np.float32)/cs.out_size  # sum buffers
            # right_ch = data_tensor.sum(axis=0).numpy().astype(
            #     np.float32)  # sum buffers
            # stereo_data = np.column_stack(
            #     [left_ch, right_ch]).flatten().astype(np.float32)
            try:
                cs.audio_queue.put_nowait(raw_input_sum.tobytes())
            except queue.Full:
                # Remove oldest, add newest - keep audio flowing
                try:
                    cs.audio_queue.get_nowait()
                    cs.audio_queue.put_nowait(raw_input_sum.tobytes())
                except queue.Empty:
                    pass

        if cs.plotter is not None:
            plotter_data = {"ref_timestamp": ref_timestamp,
                            **filtered_in,
                            **{f"out_{i}": filtered_out[i] for i in range(cs.out_size)}}
            cs.plotter.update_data(plotter_data, data_len=cs.seq_len)

        filtered_out = np.array(filtered_out)  # makes model change easier

        # permute output
        if cs.permute_out:
            filtered_out = filtered_out[cs.model_perm]

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

        # is it time to change model?
        if (ratio > cs.ratio_rising_threshold and past_change_model == 0):
            cs.change_model = 1
        elif (ratio < cs.ratio_falling_threshold and past_change_model == 1):
            cs.change_model = 0

        cs.debug_counter += 1  # for debugging
        if (cs.debug_counter % 20 == 0):
            print(ratio)

        # # if it's time to change model...
        # if (past_change_model == 0 and cs.change_model == 1):
        #     change_model(filtered_out, cs)

        # diagnostics

        # In callback, every 50 iterations:
        if cs.debug_counter % 50 == 0:
            elapsed = time.perf_counter() - start

            target = 1024 / (cs.sample_rate)  # Your block period
            utilization = (elapsed / target) * 100
            print(f"Callback: {elapsed*1000:.1f}ms ({utilization:.1f}% CPU)")
            # Check system memory
            ram_percent = psutil.virtual_memory().percent

            # Check GPU memory if using CUDA
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
                print(
                    f"RAM: {ram_percent}%, GPU: {gpu_memory:.1f}MB, Queue: {cs.audio_queue.qsize()}")

            # Check audio queue size
            print(f"Audio queue size: {cs.audio_queue.qsize()}")


if __name__ == "__main__":
    in_vars = ['gFaabSensor_1', 'gFaabSensor_2', 'gFaabSensor_3', 'gFaabSensor_4',
               'gFaabSensor_5', 'gFaabSensor_6', 'gFaabSensor_7', 'gFaabSensor_8']

    out_vars = ['out_0', 'out_1', 'out_2', 'out_3']

    parser = argparse.ArgumentParser(description="faab-callback")
    parser.add_argument('--osc', action='store_true', help='Use OSC server')
    parser.add_argument('--plot', action='store_true',
                        help='Enable Bokeh plotting')
    parser.add_argument('--out2Bela', action='store_true',
                        help='Send output to Bela')
    parser.add_argument('--audio', action='store_true',
                        help="Play model output as audio")
    args = parser.parse_args()

    if args.osc:
        osc_ip, osc_port = "127.0.0.1", 2222
    else:
        osc_ip, osc_port = None, None

    streamer = Streamer()
    streamer.connect()

    sample_rate = streamer.sample_rate / 2  # analog rate is half audio rate

    plotter = None
    if args.plot:
        plotter = AudioDataPlotter(
            y_vars=[*in_vars, *out_vars],
            y_range=(-1, 1),
            rollover=3*1024,
            plot_update_delay=1024*1000/sample_rate,
            sample_rate=sample_rate,
            port=5007,
        )
        plotter.start_server()

    cs = CallbackState(
        seq_len=1024,
        num_models=20,
        in_size=8,
        out_size=4,
        sample_rate=sample_rate,
        num_blocks_to_compute_avg=10,
        num_blocks_to_compute_std=40,
        hp_filter_freq=1,
        lp_filter_freq=5000,
        envelope_len=256,
        num_of_iterations_in_this_model_check=100,
        init_ratio_rising_threshold=2.5,
        init_ratio_falling_threshold=1.3,
        threshold_leak=0.01,
        trigger_width=25,
        trigger_idx=4,
        running_norm=True,
        permute_out=False,
        path="src/models/trained/transformer-autoencoder-jan",
        osc_ip=osc_ip,
        osc_port=osc_port,
        plotter=plotter,
        out2Bela=args.out2Bela,
        play_audio=args.audio
    )

    streamer.start_streaming(
        in_vars, on_block_callback=callback, callback_args=(cs, streamer))

    streamer.wait(0)
