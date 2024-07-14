import torch
import asyncio
import numpy as np
from pybela import Streamer
from utils import load_model, get_device, get_all_run_ids, get_models_coordinates, find_closest_model, get_models_range

device = get_device()

# preload models
run_ids = get_all_run_ids()
starter_id = run_ids[0]
models = {}
for _id in run_ids:
    models[_id] = load_model(_id)

models_range = get_models_range()
models_coordinates = get_models_coordinates() 

streamer = Streamer()
vars = ['gFaabSensor_1', 'gFaabSensor_2', 'gFaabSensor_3', 'gFaabSensor_4', 'gFaabSensor_5', 'gFaabSensor_6', 'gFaabSensor_7', 'gFaabSensor_8']
seq_len = 512
num_blocks_to_compute_avg = 10
# avg = torch.empty(0).to(device)

avg =  torch.empty(0).to(device)
model = models[starter_id]
async def callback(block):
    global avg, model
    _raw_data_tensor = torch.stack([torch.FloatTensor(buffer["buffer"]["data"]) for buffer in block]) # num_features, 1024
    # split the data into seq_len to feed it into the model
    inputs = _raw_data_tensor.unfold(1, seq_len, seq_len).permute(1,2,0) # n, seq_len, num_features  

    # for each sequence of seq_len, feed it into the model
    for _input in inputs:
        out = model.forward_encoder(_input.to(device)).permute(1,0) # num_outputs, seq_len
        
        for idx,feature in enumerate(out): # send each feature to Bela
            streamer.send_buffer(idx, 'f', seq_len, feature.tolist())

        avg = torch.cat((avg, out.mean(dim=1).unsqueeze(0)), dim=0)
        

    if len(avg) == num_blocks_to_compute_avg: 
        _avg = avg.mean(dim=0)
        _model_range = models_range[model.id]
        _min = torch.FloatTensor(_model_range["min"]).to(device)
        _max = torch.FloatTensor(_model_range["max"]).to(device)        
        _avg = (_avg - _min) / (_max - _min)
        _avg = _avg.detach().cpu().tolist()
                
        avg =  torch.empty(0).to(device)
        closest_model, _ = find_closest_model(_avg, models_coordinates)
        model = models[closest_model]


if streamer.connect():
    streamer.start_streaming(vars, on_block_callback=callback)
    asyncio.run(asyncio.sleep(10))
    streamer.stop_streaming()
