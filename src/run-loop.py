import torch
import numpy as np
import asyncio
from pybela import Streamer
from utils import load_model, get_device, get_all_run_ids, get_models_coordinates, find_closest_model, get_models_range

path = "src/models/trained"
device = get_device()

streamer = Streamer()
vars = ['gFaabSensor_1', 'gFaabSensor_2', 'gFaabSensor_3', 'gFaabSensor_4',
        'gFaabSensor_5', 'gFaabSensor_6', 'gFaabSensor_7', 'gFaabSensor_8']
seq_len = 512
streamer.connect()

# preload models
run_ids = get_all_run_ids(path=path)
starter_id = run_ids[0]
models = {}
models_running_range = {}
for _id in run_ids:
    models[_id] = load_model(_id, path=path)
    models_running_range[_id] = {"min": torch.FloatTensor([0, 0, 0, 0]).to(
        device), "max": torch.FloatTensor([1, 1, 1, 1]).to(device)}

# model space
# min and max of model outputs (after passing the whole dataset)
full_dataset_models_range = get_models_range(path=path)
# model's chosen 4 hyperparameters mapped to values between 0 and 1
models_coordinates = get_models_coordinates(path=path)

num_blocks_to_compute_avg = 10

# init average and model
model_avg = torch.empty(0).to(device)
model_min = torch.empty(0).to(device)
model_max = torch.empty(0).to(device)

model = models[starter_id]

# settings
running_norm = True


async def callback(block):

    # global variables so that the state is kept between callback calls
    global model_avg, model_min, model_max, model

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

            # for idx,feature in enumerate(out): # send each feature to Bela
            #     streamer.send_buffer(idx, 'f', seq_len, feature.tolist())
            model_avg = torch.cat(
                (model_avg, out.mean(dim=1).unsqueeze(0)), dim=0)

            if running_norm:
                model_min = torch.cat(
                    (model_min, out.min(dim=1).values.unsqueeze(0)), dim=0)
                model_max = torch.cat(
                    (model_max, out.max(dim=1).values.unsqueeze(0)), dim=0)

        if len(model_avg) == num_blocks_to_compute_avg:
            
            # -- normalisation -- 
            
            # average model output over 512 * num_blocks_to_compute_avg
            _avg = model_avg.mean(dim=0)

            # model output has an arbitrary range, so normalise the model output

            # running normalisation (taking max and min from the current run)
            if running_norm:
                models_running_range[_id]["min"] = torch.stack(
                    (models_running_range[_id]["min"], model_min.min(dim=0).values), dim=0).min(dim=0).values
                models_running_range[_id]["max"] = torch.stack(
                    (models_running_range[_id]["max"], model_max.max(dim=0).values), dim=0).max(dim=0).values

                _min, _max = models_running_range[_id]["min"], models_running_range[_id]["max"]

            # absolute normalisation (taking max and min from passing the full dataset)
            else:
                _model_range = full_dataset_models_range[model.id]
                _min, _max = torch.FloatTensor(_model_range["min"]).to(
                    device), torch.FloatTensor(_model_range["max"]).to(device)

            _avg = (_avg - _min) / (_max - _min)
            _avg = _avg.detach().cpu().tolist()

            # -- gain -- 
            # multiply the final averaged value by a tuned gain
            gain = [1.35, 1.1, 1.5, 1.5]
            _avg = [a * g for a, g in zip(_avg, gain)]

            # -- map to model --
            # find the closest model to the _avg coordinates
            closest_model, _ = find_closest_model(_avg, models_coordinates)
            model = models[closest_model]

            # -- reset avg --
            model_avg = torch.empty(0).to(device)

            print(model.id, _avg)

streamer.start_streaming(vars, on_block_callback=callback)
async def wait_forever():
    await asyncio.Future()
asyncio.run(wait_forever())
