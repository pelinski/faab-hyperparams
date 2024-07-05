import torch
import sys
import pickle
import numpy as np
from pybela import Logger


class Dataset(torch.utils.data.Dataset):
    def __init__(self, pickle_path="", seq_len=0, device=torch.device("cpu"), n_items=0, normalise = False):
        super().__init__()
        self.pickle_path = pickle_path
        self.device = device
        self.timestamp =  pickle_path.split('/')[-1].split('_')[0]

        # self._seq_len = seq_len
        self.seq_len = seq_len
        # self._raw_data_tensor = None
        self.feature_names = []
        # self.raw_data = {}
        self.num_features = None
        self.inputs = None

        if pickle_path != "" and seq_len != 0:
            _raw_data = self.load_data_from_pickle(self.pickle_path)
            self.num_features = len(_raw_data.keys())
            self.feature_names = list(_raw_data.keys())

            _raw_data_tensor = self.process_raw_data_into_torch_tensor(
                _raw_data)
            self.inputs = _raw_data_tensor.unfold(
                1, self.seq_len, self.seq_len).permute(1, 2, 0).to(self.device)  # n, seq_len, num_features

            if n_items != 0:
                assert n_items < len(
                    self.inputs), "n_items must be less than the number of samples in the dataset"
                self.inputs = self.inputs[:n_items+1]

            # normalise
            if normalise:
                self.inputs = self.normalise(self.inputs)

    def process_raw_data_into_torch_tensor(self, _raw_data):
        _min_len = min([len(_raw_data[key])
                       for key in _raw_data.keys()])
        _total_data_points = (_min_len - _min_len % self.seq_len)

        _raw_data_tensor = torch.empty(
            (self.num_features, _total_data_points))

        for i, key in enumerate(_raw_data.keys()):
            _raw_data_tensor[i] = torch.FloatTensor(
                _raw_data[key][:_total_data_points])

        return _raw_data_tensor

    # @property
    # def seq_len(self):
    #     return self._seq_len

    # @seq_len.setter  # updates inputs
    # def seq_len(self, new_seq_len):
    #     self._seq_len = new_seq_len
    #     self.inputs = self._raw_data_tensor.unfold(
    #         1, self.seq_len, self.seq_len).permute(1, 2, 0).to(self.device)
    #     print("seq_len updated. inputs updated.")
    
    # TODO denormalise function
    
    def normalise(self, data):
        min_val = data.min()
        max_val = data.max()
        data = (data - min_val) / (max_val - min_val)

        return data

    def load_data_from_pickle(self, file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data

    def load_dataset_from_pickle(self, pickle_path):
        with open(pickle_path, 'rb') as file:
            dataset = pickle.load(file)
        return dataset

    # def convert_to_csv(self, path=None):
    #     df = pd.DataFrame(self.raw_data_tensor.T.cpu().numpy())
    #     if path is None:
    #         path = dataset.pickle_path.split('.')[0] + '.csv'
    #     df.to_csv(path, index=False)
    #     return path

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.inputs[idx]


class DatasetPred(Dataset):
    def __init__(self, pickle_path="", seq_len=0, device=torch.device("cpu"), n_items=0):
        super().__init__(pickle_path, seq_len, device, n_items)
        self.targets = None

        if pickle_path != "" and seq_len != 0:
            self.targets = self.inputs[1:]
            self.inputs = self.inputs[:-1]  # first do targets and then inputs

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def convert_binary_dataset_to_dict(date, vars, save_as_pickle=True):
    data = {}
    raw_data = {}

    for var in vars:
        _data = Logger().read_binary_file( f'data/{date}/{var}.bin', timestamp_mode="dense")
        raw_data[var] = _data
        
        print(f"Processing {var}...")
        
        data[var] = [item for _buffer in _data["buffers"] for item in _buffer["data"]]
        time = abs(raw_data[var]['buffers'][0]['ref_timestamp'] - raw_data[var]['buffers'][-1]['ref_timestamp'])/44100/60
        
        print(f'Processed {var} – {len(data[var])} points, {np.round(time,3)} min')
        
    if save_as_pickle:
        filename = f'data/{date}/raw.pkl'
        with open(filename, 'wb') as file: 
            pickle.dump(data, file) 
            
    return data
        
        
        
    
    

if __name__ == "__main__":
    seq_len = 512
    n_features = 8

    data_path = sys.argv[1] if len(
        sys.argv) > 1 else 'src/dataset/data/202407041315_raw.pkl'
    
    note = "" # REMEMBER TO ADD NOTES!!
    
    dataset = Dataset(
        pickle_path=data_path, seq_len=seq_len, device="cpu")
    first_item = dataset[0]
    path = f'src/dataset/data/{dataset.timestamp}_processed_{seq_len}.pkl'
    with open(path, 'wb') as file:
        pickle.dump(dataset, file)

    note_filename = f'{dataset.timestamp}_note.txt'
    with open(f'src/dataset/data/{note_filename}', 'w') as file:
        file.write(note)
        
    # dataset = Dataset(
    #     pickle_path=data_path, seq_len=seq_len, device="cpu", n_items=20)
    # first_item = dataset[0]
    # path = f'src/dataset/processed_dataset_{seq_len}.pkl'
    # with open(path, 'wb') as file:
    #     pickle.dump(dataset, file)

    # dataset_pred = DatasetPred(
    #     pickle_path=data_path, seq_len=seq_len, device="cpu")
    # path = f'src/dataset/processed_dataset_pred_{seq_len}.pkl'
    # with open(path, 'wb') as file:
    #     pickle.dump(dataset_pred, file)

    # path = f'src/dataset/processed_dataset_pred_{seq_len}_mini.pkl'
    # dataset_pred_mini = DatasetPred(
    #     pickle_path=data_path, seq_len=seq_len, device="cpu", n_items=124)
    # with open(path, 'wb') as file:
    #     pickle.dump(dataset_pred_mini, file)

    # path = dataset.convert_to_csv()
