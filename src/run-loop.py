import torch
import numpy as np
import asyncio
import biquad
from collections import deque  # circular buffers
from pybela import Streamer
from utils import load_model, get_device, get_sorted_models, get_models_coordinates, find_closest_model, get_models_range

path = "src/models/trained"
device = get_device()

streamer = Streamer()
vars = ['gFaabSensor_1', 'gFaabSensor_2', 'gFaabSensor_3', 'gFaabSensor_4',
        'gFaabSensor_5', 'gFaabSensor_6', 'gFaabSensor_7', 'gFaabSensor_8']
streamer.connect()

# -- params --
seq_len = 512
num_models = 20
num_blocks_to_compute_avg = 10
num_blocks_to_compute_std = 40
trigger_width = 25
trigger_idx = 4
running_norm = True

# preload models
# models in desc order of train loss, take best 20
sorted_models = get_sorted_models(path=path, num_models=num_models)
models = {}
models_running_range = {}
for _id in sorted_models:
    models[_id] = load_model(_id, path=path)
    models_running_range[_id] = {"min": torch.FloatTensor([0, 0, 0, 0]).to(
        device), "max": torch.FloatTensor([0, 0, 0, 0]).to(device)}

# model space
# min and max of model outputs (after passing the whole dataset)
full_dataset_models_range = get_models_range(path=path)
# model's chosen 4 hyperparameters mapped to values between 0 and 1
models_coordinates = get_models_coordinates(
    path=path, sorted=True, num_models=num_models)


# init averages
bridge_piezo_avg, model_out_std = deque(
    maxlen=num_blocks_to_compute_avg),  deque(maxlen=num_blocks_to_compute_std)

# init model
starter_id = sorted_models[-1]  # h0o65m8s is nice
model = models[starter_id]
change_model = 0

# settings
filter = biquad.lowpass(sr=streamer.sample_rate, f=1, q=0.707)
gain = 4*[1.0]
# sound_threshold =0.021
init_ratio_rising_threshold, init_ratio_falling_threshold = 2.5, 1.3  # 0.15, 0.1
ratio_rising_threshold, ratio_falling_threshold = init_ratio_rising_threshold, init_ratio_falling_threshold
threshold_leak = 0.1
num_of_iterations_in_this_model_check = 20

# counters
debug_counter = 0
iterations_in_this_model_counter = 0


# ratio_rising, ratio_falling = 0.15, 0.1
ratio_rising, ratio_falling = 2.5, 1.3
counter = 0


async def callback(block):

    # global variables so that the state is kept between callback calls
    global model_out_std, bridge_piezo_avg, model, gain, change_model, ratio_rising_threshold, ratio_falling_threshold, iterations_in_this_model_counter, debug_counter

    with torch.no_grad():

        _raw_data_tensor = torch.stack([torch.FloatTensor(
            buffer["buffer"]["data"]) for buffer in block])  # num_features, 1024
        # split the data into seq_len to feed it into the model
        inputs = _raw_data_tensor.unfold(1, seq_len, seq_len).permute(
            1, 2, 0)  # n, seq_len, num_features

        # for each sequence of seq_len, feed it into the model
        for _input in inputs:
            iterations_in_this_model_counter += 1
            out = model.forward_encoder(_input.to(device)).permute(
                1, 0)  # num_outputs, seq_len
            # outputs --> [ff_size, num_heads, num_layers, learning_rate]

            # -- normalisation --

            # running normalisation (taking max and min from the current run)
            if running_norm:
                models_running_range[_id]["min"] = torch.stack(
                    (models_running_range[_id]["min"], out.min(dim=1).values), dim=0).min(dim=0).values
                models_running_range[_id]["max"] = torch.stack(
                    (models_running_range[_id]["max"], out.max(dim=1).values), dim=0).max(dim=0).values

                _min, _max = models_running_range[_id]["min"], models_running_range[_id]["max"]

            # absolute normalisation (taking max and min from passing the full dataset)
            else:
                _model_range = full_dataset_models_range[model.id]
                _min, _max = torch.FloatTensor(_model_range["min"]).to(
                    device), torch.FloatTensor(_model_range["max"]).to(device)

            # -- normalise before sending to Bela!! --
            normalised_out = (out - _min.unsqueeze(1)) / \
                (_max - _min).unsqueeze(1)

            # send each feature to Bela
            for idx, feature in enumerate(normalised_out):
                streamer.send_buffer(idx, 'f', seq_len, feature.tolist())

            # -- amplitude --
            _bridge_piezo = filter(block[5]["buffer"]["data"])
            bridge_piezo_avg.append(np.average(_bridge_piezo))
            weighted_avg_bridge_piezo = np.round(np.average(
                bridge_piezo_avg, weights=range(1, len(bridge_piezo_avg)+1)), 5)

            # -- model variance --
            model_out_std.append(normalised_out.std(dim=1).mean().item())
            weighted_model_out_std = np.average(
                model_out_std, weights=range(1, len(model_out_std)+1))

            # -- model change --
            past_change_model = change_model

            # leaky threshold
            if iterations_in_this_model_counter % num_of_iterations_in_this_model_check == 0:
                ratio_rising_threshold -= threshold_leak
                ratio_falling_threshold += threshold_leak

            ratio = weighted_model_out_std / weighted_avg_bridge_piezo
            if (ratio > ratio_rising_threshold and past_change_model == 0):
                change_model = 1
            elif (ratio < ratio_falling_threshold and past_change_model == 1):
                change_model = 0

            debug_counter += 1  # for debugging
            if (debug_counter % 20 == 0):
                print(ratio)

            if (past_change_model == 0 and change_model == 1):
                # high sound amplitude and model has low variance --> change model
                # streamer.send_buffer(
                #     trigger_idx, 'f', seq_len, trigger_width*[1.0] + (seq_len-trigger_width)*[0.0])  # change trigger

                out_coordinates = normalised_out[:, -1].detach().cpu().tolist()
                out_coordinates = [c * g for c,
                                   g in zip(out_coordinates, gain)]

                # find the closest model to the out_ coordinates
                closest_model, _ = find_closest_model(out_coordinates, models_coordinates)
                model = models[closest_model]
                iterations_in_this_model_counter = 0
                ratio_rising_threshold, ratio_falling_threshold = init_ratio_rising_threshold, init_ratio_falling_threshold

                print(model.id, np.round(out_coordinates, 4))

            else:
                pass
                # streamer.send_buffer(
                #     trigger_idx, 'f', seq_len, seq_len*[0.0])  # change trigger


streamer.start_streaming(vars, on_block_callback=callback)


async def wait_forever():
    await asyncio.Future()
asyncio.run(wait_forever())
