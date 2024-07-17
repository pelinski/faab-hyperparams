import torch
import numpy as np
import asyncio
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


# init average
model_avg = torch.empty(0).to(device)
# init model
starter_id = sorted_models[0]
model = models[starter_id]

# settings
gain = 4*[1.0]
counter = 0


async def callback(block):

    # global variables so that the state is kept between callback calls
    global model_avg, model_min, model_max, model, gain, counter

    with torch.no_grad():

        _raw_data_tensor = torch.stack([torch.FloatTensor(
            buffer["buffer"]["data"]) for buffer in block])  # num_features, 1024
        # split the data into seq_len to feed it into the model
        inputs = _raw_data_tensor.unfold(1, seq_len, seq_len).permute(
            1, 2, 0)  # n, seq_len, num_features

        # for each sequence of seq_len, feed it into the model
        for _input in inputs:
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
                counter += 1

                if counter < num_blocks_to_compute_avg:
                    streamer.send_buffer(
                        trigger_idx, 'f', seq_len, seq_len*[0.0])
                elif counter == num_blocks_to_compute_avg:
                    streamer.send_buffer(
                        trigger_idx, 'f', seq_len, trigger_width*[1.0] + (seq_len-trigger_width)*[0.0])
                    counter = 0

            model_avg = torch.cat(
                (model_avg, normalised_out.mean(dim=1).unsqueeze(0)), dim=0)

        if len(model_avg) == num_blocks_to_compute_avg:
            # -- gain --
            _avg = model_avg.mean(dim=0).detach().cpu().tolist()
            # multiply the final averaged value by a tuned gain
            _avg = [a * g for a, g in zip(_avg, gain)]

            # -- map to model --
            # find the closest model to the _avg coordinates
            closest_model, _ = find_closest_model(_avg, models_coordinates)
            model = models[closest_model]

            # -- reset avg --
            model_avg = torch.empty(0).to(device)

            print(model.id, np.round(_avg, 4))


streamer.start_streaming(vars, on_block_callback=callback)


async def wait_forever():
    await asyncio.Future()
asyncio.run(wait_forever())
