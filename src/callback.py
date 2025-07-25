import torch
import time
import numpy as np
import biquad
import argparse
from collections import deque  # circular buffers
from pybela import Streamer
from pythonosc.udp_client import SimpleUDPClient
from utils.utils import load_model, get_device, get_sorted_models, get_models_coordinates, get_models_range, normalise, calc_ratio_amplitude_variance, change_model
from utils.bokeh import AudioDataPlotter


class CallbackState:
    def __init__(self, seq_len, num_models, out_size, num_blocks_to_compute_avg, num_blocks_to_compute_std, out_hp_filter_freq, out_lp_filter_freq, envelope_len, num_of_iterations_in_this_model_check, init_ratio_rising_threshold, init_ratio_falling_threshold, threshold_leak, trigger_width, trigger_idx, running_norm, permute_out, path, osc_ip=None, osc_port=None, plotter=None):

        # -- params --
        self.seq_len = seq_len
        self.num_models = num_models
        self.out_size = out_size
        self.num_blocks_to_compute_avg, self.num_blocks_to_compute_std = num_blocks_to_compute_avg, num_blocks_to_compute_std
        self.out_hp_filter_freq, self.out_lp_filter_freq = out_hp_filter_freq, out_lp_filter_freq
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

        # model space
        # min and max of model outputs (after passing the full dataset)
        self.full_dataset_models_range = get_models_range(path=path)
        # model's chosen 4 hyperparameters mapped to values between 0 and 1
        self.models_coordinates = get_models_coordinates(
            path=path, sorted=True, num_models=num_models)
        starter_id = sorted_models[-1]  # h0o65m8s is nice

        # filters
        self.bridge_filter = biquad.lowpass(
            sr=streamer.sample_rate, f=1, q=0.707)
        self.out_hp = [biquad.highpass(
            sr=streamer.sample_rate, f=out_hp_filter_freq, q=0.707) for _ in range(out_size)]
        self.out_lp = [biquad.lowpass(
            sr=streamer.sample_rate, f=out_lp_filter_freq, q=0.707) for _ in range(out_size)]

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

# sound_threshold =0.021


async def callback(block, cs, streamer):

    with torch.no_grad():
        ref_timestamp = block[0]["buffer"]["ref_timestamp"]
        plotter_data = {"ref_timestamp": ref_timestamp, **{var["name"]: var["buffer"]["data"]
                                                           for var in block}, **{f"normalised_out_{i}": [] for i in range(cs.out_size)}}

        _raw_data_tensor = torch.stack([torch.as_tensor(
            buffer["buffer"]["data"], dtype=torch.float32) for buffer in block])  # num_features, 1024
        # # split the data into seq_len to feed it into the model
        inputs = _raw_data_tensor.unfold(1, cs.seq_len, cs.seq_len).permute(
            1, 2, 0)  # n, seq_len, num_features
        # for each sequence of seq_len, feed it into the model

        for _input in inputs:
            cs.iterations_in_this_model_counter += 1
            out = cs.model.forward_encoder(_input.to(cs.device)).permute(
                1, 0)  # num_outputs, seq_len
            # outputs --> [ff_size, num_heads, num_layers, learning_rate]

            normalised_out = normalise(out, cs)
            for i in range(cs.out_size):
                plotter_data[f"normalised_out_{i}"] = normalised_out[i].tolist(
                )

            if cs.plotter is not None:
                cs.plotter.update_data(plotter_data, data_len=cs.seq_len)

            # permute output
            if cs.permute_out:
                normalised_out = normalised_out[cs.model_perm]

            # send each feature to Bela or OSC
            for idx, feature in enumerate(normalised_out):
                # dc filter
                filtered_out = cs.out_hp[idx](feature.cpu())
                filtered_out = cs.out_lp[idx](filtered_out).tolist()

                # envelope # TODO
                # filtered_out =

                if cs.osc_client:
                    cs.osc_client.send_message(f'/f{idx+1}', filtered_out)
                else:
                    streamer.send_buffer(idx, 'f', cs.seq_len, filtered_out)

            bridge_piezo_block = block[5]["buffer"]["data"]
            ratio = calc_ratio_amplitude_variance(
                normalised_out, bridge_piezo_block, cs)

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

            # if it's time to change model...
            if (past_change_model == 0 and cs.change_model == 1):
                change_model(normalised_out, cs)


if __name__ == "__main__":

    vars = ['gFaabSensor_1', 'gFaabSensor_2', 'gFaabSensor_3', 'gFaabSensor_4',
            'gFaabSensor_5', 'gFaabSensor_6', 'gFaabSensor_7', 'gFaabSensor_8']

    out_vars = ['normalised_out_0', 'normalised_out_1',
                'normalised_out_2', 'normalised_out_3']

    parser = argparse.ArgumentParser(description="faab-callback")
    parser.add_argument('--osc', action='store_true', help='Use OSC server')
    parser.add_argument('--plot', action='store_true',
                        help='Enable Bokeh plotting')
    args = parser.parse_args()

    if args.osc:
        osc_ip, osc_port = "127.0.0.1", 2222
    else:
        osc_ip, osc_port = None, None

    streamer = Streamer()
    streamer.connect()

    plotter = None
    if args.plot:
        plotter = AudioDataPlotter(
            y_vars=[*vars, *out_vars],
            y_range=(0, 1.0),
            rollover=500,
            plot_update_delay=10,
            sample_rate=streamer.sample_rate,
        )
        plotter.start_server()
        time.sleep(1)  # wait for the server to start

    cs = CallbackState(
        seq_len=1024,
        num_models=20,
        out_size=4,
        num_blocks_to_compute_avg=10,
        num_blocks_to_compute_std=40,
        out_hp_filter_freq=10,
        out_lp_filter_freq=5000,
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
        plotter=plotter
    )

    streamer.start_streaming(
        vars, on_block_callback=callback, callback_args=(cs, streamer))

    streamer.wait(0)
