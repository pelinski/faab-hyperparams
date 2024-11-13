import torch
import numpy as np
import asyncio
import biquad
from collections import deque  # circular buffers
from pybela import Streamer
from pythonosc.udp_client import SimpleUDPClient
from utils.utils import load_model, get_device, get_sorted_models, get_models_coordinates, find_closest_model, get_models_range

class CallbackState:
    def __init__(self, seq_len, num_models, num_blocks_to_compute_avg, num_blocks_to_compute_std, filter,  num_of_iterations_in_this_model_check, init_ratio_rising_threshold, init_ratio_falling_threshold, threshold_leak, trigger_width, trigger_idx, running_norm, permute_out, path, osc_ip, osc_port):

        # -- params --
        self.seq_len = seq_len
        self.num_models = num_models
        self.num_blocks_to_compute_avg = num_blocks_to_compute_avg
        self.num_blocks_to_compute_std = num_blocks_to_compute_std
        self.filter = filter
        self.num_of_iterations_in_this_model_check = num_of_iterations_in_this_model_check
        self.init_ratio_rising_threshold = init_ratio_rising_threshold
        self.init_ratio_falling_threshold = init_ratio_falling_threshold
        self.threshold_leak = threshold_leak
        self.trigger_width = trigger_width
        self.trigger_idx = trigger_idx
        self.running_norm = running_norm
        self.permute_out = permute_out
        self.device = get_device()
        self.path = path
        self.osc_ip = osc_ip
        self.osc_port = osc_port
        
        # init osc server 
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
        # min and max of model outputs (after passing the whole dataset)
        self.full_dataset_models_range = get_models_range(path=path)
        # model's chosen 4 hyperparameters mapped to values between 0 and 1
        self.models_coordinates = get_models_coordinates(
            path=path, sorted=True, num_models=num_models)
        starter_id = sorted_models[-1]  # h0o65m8s is nice

        # variables for the callback
        self.model = self.models[starter_id]
        self.gain = 4 * [1.0]
        self.change_model = 0
        self.ratio_rising_threshold = init_ratio_rising_threshold
        self.ratio_falling_threshold = init_ratio_falling_threshold
        self.iterations_in_this_model_counter = 0
        self.bridge_piezo_avg = deque(maxlen=num_blocks_to_compute_avg)
        self.model_out_std = deque(maxlen=num_blocks_to_compute_std)
        self.debug_counter = 0
        self.model_perm = [0, 1, 2, 3]

# sound_threshold =0.021


async def callback(block, cs, streamer):

    with torch.no_grad():

        _raw_data_tensor = torch.stack([torch.FloatTensor(
            buffer["buffer"]["data"]) for buffer in block])  # num_features, 1024
        # split the data into seq_len to feed it into the model
        inputs = _raw_data_tensor.unfold(1, cs.seq_len, cs.seq_len).permute(
            1, 2, 0)  # n, seq_len, num_features

        # for each sequence of seq_len, feed it into the model
        for _input in inputs:
            cs.iterations_in_this_model_counter += 1
            out = cs.model.forward_encoder(_input.to(cs.device)).permute(
                1, 0)  # num_outputs, seq_len
            # outputs --> [ff_size, num_heads, num_layers, learning_rate]

            # -- normalisation --
            # running normalisation (taking max and min from the current run)
            if cs.running_norm:
                cs.models_running_range[cs.model.id]["min"] = torch.stack(
                    (cs.models_running_range[cs.model.id]["min"], out.min(dim=1).values), dim=0).min(dim=0).values
                cs.models_running_range[cs.model.id]["max"] = torch.stack(
                    (cs.models_running_range[cs.model.id]["max"], out.max(dim=1).values), dim=0).max(dim=0).values

                _min, _max = cs.models_running_range[cs.model.id]["min"], cs.models_running_range[cs.model.id]["max"]

            # absolute normalisation (taking max and min from passing the full dataset)
            else:
                _model_range = cs.full_dataset_models_range[cs.model.id]
                _min, _max = torch.FloatTensor(_model_range["min"]).to(
                    cs.device), torch.FloatTensor(_model_range["max"]).to(cs.device)

            # -- normalise before sending to Bela!! --
            normalised_out = (out - _min.unsqueeze(1)) / \
                (_max - _min).unsqueeze(1)

            # permute output
            if cs.permute_out:
                normalised_out = normalised_out[cs.model_perm]

            # send each feature to Bela
            for idx, feature in enumerate(normalised_out):
                cs.osc_client.send_message(f'/f{idx+1}', feature.tolist())
                # streamer.send_buffer(idx, 'f', cs.seq_len, feature.tolist())

            # -- amplitude --
            _bridge_piezo = cs.filter(block[5]["buffer"]["data"])
            cs.bridge_piezo_avg.append(np.average(_bridge_piezo))
            weighted_avg_bridge_piezo = np.average(
                cs.bridge_piezo_avg, weights=range(1, len(cs.bridge_piezo_avg)+1))

            # -- model variance --
            cs.model_out_std.append(normalised_out.std(dim=1).mean().item())
            weighted_model_out_std = np.average(
                cs.model_out_std, weights=range(1, len(cs.model_out_std)+1))

            # -- model change --
            past_change_model = cs.change_model

            # leaky threshold
            if cs.iterations_in_this_model_counter % cs.num_of_iterations_in_this_model_check == 0:
                cs.ratio_rising_threshold -= cs.threshold_leak
                cs.ratio_falling_threshold += cs.threshold_leak

            # is it time to change model?
            ratio = weighted_model_out_std / weighted_avg_bridge_piezo
            if (ratio > cs.ratio_rising_threshold and past_change_model == 0):
                cs.change_model = 1
            elif (ratio < cs.ratio_falling_threshold and past_change_model == 1):
                cs.change_model = 0

            cs.debug_counter += 1  # for debugging
            if (cs.debug_counter % 20 == 0):
                print(ratio)

            # if it's time to change model...
            if (past_change_model == 0 and cs.change_model == 1):
                # high sound amplitude and model has low variance --> change model
                # streamer.send_buffer(
                #     trigger_idx, 'f', seq_len, trigger_width*[1.0] + (seq_len-trigger_width)*[0.0])  # change trigger

                out_coordinates = normalised_out[:, -1].detach().cpu().tolist()
                out_coordinates = [c * g for c,
                                   g in zip(out_coordinates, cs.gain)]

                # find the closest model to the out_ coordinates
                closest_model, _ = find_closest_model(
                    out_coordinates, cs.models_coordinates)
                cs.model = cs.models[closest_model]
                if cs.permute_out:
                    cs.model_perm = torch.randperm(4)

                # reset counter and thresholds
                cs.iterations_in_this_model_counter = 0
                cs.ratio_rising_threshold, cs.ratio_falling_threshold = cs.init_ratio_rising_threshold, cs.init_ratio_falling_threshold

                print(cs.model.id, np.round(out_coordinates, 4))

            else:
                pass
                # streamer.send_buffer(
                #     trigger_idx, 'f', seq_len, seq_len*[0.0])  # change trigger

if __name__ == "__main__":

    streamer = Streamer()
    streamer.connect()
    vars = ['gFaabSensor_1', 'gFaabSensor_2', 'gFaabSensor_3', 'gFaabSensor_4',
            'gFaabSensor_5', 'gFaabSensor_6', 'gFaabSensor_7', 'gFaabSensor_8']

    cs = CallbackState(
        seq_len=512,
        num_models=20,
        num_blocks_to_compute_avg=10,
        num_blocks_to_compute_std=40,
        filter=biquad.lowpass(sr=streamer.sample_rate, f=1, q=0.707),
        num_of_iterations_in_this_model_check=20,
        init_ratio_rising_threshold=2.5,
        init_ratio_falling_threshold=1.3,
        threshold_leak=0.1,
        trigger_width=25,
        trigger_idx=4,
        running_norm=True,
        permute_out=False,
        path="src/models/trained/transformer-autoencoder",
        osc_ip = "127.0.0.1",
        osc_port=2222
    )

    
    streamer.start_streaming(
        vars, on_block_callback=callback, callback_args=(cs, streamer))

    async def wait_forever():
        await asyncio.Future()
    asyncio.run(wait_forever())
