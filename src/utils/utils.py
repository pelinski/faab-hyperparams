import argparse
import yaml
import torch 
import json
import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.models.widgets import Panel, Tabs
from bokeh.models import Legend
from bokeh.embed import file_html
from bokeh.resources import CDN
from models import TransformerAutoencoder

# -- train loop --


def get_device():
    # return torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu") # uncomment for mac m1!!! -- comment for clusters
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_hyperparams():
    """Loads hyperparameters from a config yaml file.

    Returns:
        (dict): hyperparams
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-hp', '--hyperparameters', type=str,
                        default='src/configs/transformer.yaml', help='path to hyperparameters file')
    parser.add_argument("--project", help="project",
                        default="test", type=str)
    parser.add_argument("--pickle_path", help="dataset path",
                        default="src/dataset/processed_dataset_512.pkl", type=str)
    parser.add_argument("--weights_path", help="weights path", default=None, type=str)
    parser.add_argument("--seq_len", help="maximum sequence length",
                        default=512, type=int)
    parser.add_argument("--pred", help="prediction task",
                        default=False, type=bool)
    parser.add_argument("--batch_size", help="batch size",
                        default=64, type=int)
    parser.add_argument("--feat_in_size", help="feature input size",
                        default=8, type=int)
    parser.add_argument("--d_model", help="model dimension",
                        default=4, type=int)
    parser.add_argument("--ff_size", help="ff size",
                        default=12, type=int)
    parser.add_argument("--num_layers", help="number of layers",
                        default=4, type=int)
    parser.add_argument("--model", help="model",
                        default="transformer", type=str)
    parser.add_argument(
        "--num_heads", help="number of heads for multihead attention", default=16, type=int)
    parser.add_argument(
        "--pe_scale_factor", help="positional encoding scale factor", default=1.0, type=float)
    parser.add_argument("--mask", help="mask",
                        default=False, type=bool)
    parser.add_argument("--max_grad_norm",
                        help="maximum gradient norm", default=0, type=float)
    parser.add_argument("--dropout", help="dropout", default=0.5, type=float)
    parser.add_argument(
        "--epochs", help="number of training epochs", default=500, type=int)
    parser.add_argument(
        "--optimizer", help="optimizer algorithm", default='rmsprop', type=str)
    parser.add_argument("--criterion", help="loss criterion", default='mse', type=str)
    parser.add_argument("--learning_rate",
                        help="learning rate", default=0.0001, type=float)
    parser.add_argument("--scheduler_step_size",
                        help="learning rate scheduler step size", default=0, type=int)
    parser.add_argument(
        "--scheduler_gamma", help="learning rate scheduler gamma", default=0.1, type=float)
    parser.add_argument("--num_encoder_layers",
                        help="number of encoder layers", default=7, type=int,)
    parser.add_argument("--plotter_samples",
                        help="number of samples to plot", default=5, type=int,)

    args = parser.parse_args()

    if args.hyperparameters:
        with open(args.hyperparameters, "r") as f:
            hp = yaml.safe_load(f)
    else:
        hp = {}

    hyperparams = {"project": hp["project"] if "project" in hp else args.project,
                   "pickle_path": hp["pickle_path"] if "pickle_path" in hp else args.pickle_path,
                   "weights_path": hp["weights_path"] if "weights_path" in hp else args.weights_path,
                   "seq_len": hp["seq_len"] if "seq_len" in hp else args.seq_len,
                   "pred": hp["pred"] if "pred" in hp else args.pred,
                   "batch_size": hp["batch_size"] if "batch_size" in hp else args.batch_size,
                   "feat_in_size": hp["feat_in_size"] if "feat_in_size" in hp else args.feat_in_size,
                   "d_model": hp["d_model"] if "d_model" in hp else args.d_model,
                   "ff_size": hp["ff_size"] if "ff_size" in hp else args.ff_size,
                   "num_layers": hp["num_layers"] if "num_layers" in hp else args.num_layers,
                   "model": hp["model"] if "model" in hp else args.model,
                   "max_grad_norm": hp["max_grad_norm"] if "max_grad_norm" in hp else args.max_grad_norm,
                   "dropout": hp["dropout"] if "dropout" in hp else args.dropout,
                   "epochs": hp["epochs"] if "epochs" in hp else args.epochs,
                   "optimizer": hp["optimizer"] if "optimizer" in hp else args.optimizer,
                    "criterion": hp["criterion"] if "criterion" in hp else args.criterion,
                   "learning_rate": hp["learning_rate"] if "learning_rate" in hp else args.learning_rate,
                   "scheduler_step_size": hp["scheduler_step_size"] if "scheduler_step_size" in hp else args.scheduler_step_size,
                   "scheduler_gamma": hp["scheduler_gamma"] if "scheduler_gamma" in hp else args.scheduler_gamma,
                   "num_encoder_layers": hp["num_encoder_layers"] if "num_encoder_layers" in hp else args.num_encoder_layers,
                   "plotter_samples": hp["plotter_samples"] if "plotter_samples" in hp else args.plotter_samples,
                   "device": get_device()
                   }
    if hyperparams["model"] == "transformer":
        hyperparams.update(
            {
                "num_heads": hp["num_heads"] if "num_heads" in hp else args.num_heads,
                "pe_scale_factor": hp["pe_scale_factor"] if "pe_scale_factor" in hp else args.pe_scale_factor,
                "mask": hp["mask"] if "mask" in hp else args.mask,
            }
        )

    return dict(hyperparams), args.hyperparameters


def get_html_plot(outputs, targets, feature_names):
    """Plots model predictions and targets for a given batch in html

    Args:
        outputs (torch.Tensor): Model outputs with shape (batch_size, seq_length, out_size)
        targets (torch.Tensor): Model targets with shape (batch_size, seq_length, out_size)
        feature_names (list): List of feature names

    Returns:
        file_html (string): html string with the plot
    """

    num_features = outputs.shape[2]

    feature_tabs = []
    for feature_idx in range(num_features):

        seq_tabs = []
        for seq_idx in range(len(outputs)):

            x_out = np.arange(0, len(outputs[seq_idx, :, feature_idx]))
            _outputs = outputs[seq_idx, :,
                               feature_idx].flatten().detach().cpu().numpy()
            _targets = targets[seq_idx, :,
                               feature_idx].flatten().detach().cpu().numpy()

            fig = figure(plot_width=550, plot_height=300,
                         name=feature_names[feature_idx])
            fig.xaxis.axis_label = "Frames elapsed"
            fig.yaxis.axis_label = "Value"

            # fig.y_range.start = 0
            # fig.y_range.end = 0.04

            l_out = fig.line(x_out, _outputs,
                             color="blue")
            l_tgt = fig.line(x_out, _targets,
                             color="red")
            legend = Legend(items=[("Output", [l_out]), ("Target",   [l_tgt])])
            fig.add_layout(legend, 'right')

            seq_tab = Panel(child=fig, title=str(seq_idx+1))
            seq_tabs.append(seq_tab)

        feature_tab = Panel(child=Tabs(tabs=seq_tabs),
                            title=feature_names[feature_idx])
        feature_tabs.append(feature_tab)

    html = file_html(Tabs(tabs=feature_tabs), CDN, "pad")

    return html


# -- run loop --

def get_all_run_ids(path='src/models/trained'):
    return json.load(open(f'{path}/run_ids.json'))


def get_sorted_models(path='src/models/trained', num_models=0):
    sorted_models = json.load(open(f'{path}/models_ordered_by_asc_loss.json'))
    if num_models == 0:
        num_models = len(sorted_models)
    return sorted_models[:num_models]


def load_config(_id, epochs=500, path='src/models/trained'):
    return json.load(open(f'{path}/transformer_run_{_id}_{epochs}.json'))


def load_train_loss(_id, epochs=500, path='src/models/trained'):
    return json.load(open(f'{path}/transformer_run_{_id}_{epochs}_metrics.json'))["train_loss"]


def load_model(_id, path='src/models/trained'):

    id_ep = json.load(open(f'{path}/id_ep.json'))
    epochs = id_ep[_id]
    config = load_config(_id, epochs, path)
    model = TransformerAutoencoder(d_model=config["d_model"], feat_in_size=config["feat_in_size"], num_heads=config["num_heads"], ff_size=config["ff_size"],
                                   dropout=config["dropout"], num_layers=config["num_layers"], max_len=config["seq_len"], pe_scale_factor=config["pe_scale_factor"], mask=config["mask"], id=_id)
    model.load_state_dict(torch.load(
        f'{path}/transformer_run_{_id}_{epochs}.model', map_location=get_device()))
    model = model.to(get_device())
    model.eval()

    return model


def get_models_coordinates(path='src/models/trained', sorted=True, num_models=0):
    scaled_params = json.load(open(f'{path}/scaled_params.json'))
    if not sorted:
        return scaled_params
    sorted_models = get_sorted_models(path, num_models=num_models)
    sorted_scaled_params = {key: scaled_params[key] for key in sorted_models}
    return sorted_scaled_params


def get_models_range(path='src/models/trained'):
    return json.load(open(f'{path}/models_range.json'))


def find_closest_model(output_coordinates, scaled_model_coordinates):

    model_keys, scaled_model_coordinates = zip(*scaled_model_coordinates.items())
    scaled_model_coordinates = np.array(scaled_model_coordinates)
    
    # Calculate the Euclidean distances
    distances = np.linalg.norm(
        scaled_model_coordinates - output_coordinates, axis=1)

    # Find the index of the row with the smallest distance
    closest_row_index = np.argmin(distances)

    # Closest row
    closest_model = model_keys[closest_row_index]
    closest_model_coordinates = scaled_model_coordinates[closest_row_index]

    return closest_model, closest_model_coordinates


def _scale_params(epochs={}, path='src/models/trained',):
    """Maps the hyperparameters of the trained models to a 0-1 scale.

    Args:
        epochs (dict, optional): Epochs each model has been trained. 
        path (str, optional): Path where the models are. Defaults to 'src/models/trained'.

    Returns:
        dict of lists: Returns a dict with the scaled hyperparameters in the following order: ff_size, num_heads, num_layers, learning_rate
    """
    run_ids = get_all_run_ids(path)
    params = ["ff_size", "num_heads", "num_layers", "learning_rate"]

    _id_config = {}
    for _id in run_ids:
        _id_config[_id] = {key: load_config(_id, epochs[_id], path)[
            key] for key in params}

    ranges, mapped_ranges = {
        "ff_size": [8, 16, 32, 64, 128, 256],
        "num_heads": [1, 2, 4],
        "num_layers": [1, 2, 3, 4, 5, 6, 7, 8]
    }, {}
    for key in ranges:
        mapped_ranges[key] = {
            value: idx / (len(ranges[key])-1) for idx, value in enumerate(ranges[key])}

    # Apply a different function to each column
    df = pd.DataFrame.from_dict(_id_config, orient='index')
    column_functions = {
        "ff_size": lambda x: mapped_ranges["ff_size"][x],
        "num_heads": lambda x: mapped_ranges["num_heads"][x],
        "num_layers": lambda x: mapped_ranges["num_layers"][x],
        "learning_rate": lambda x: x*1000
    }
    for column, func in column_functions.items():
        if column in df.columns:
            df[column] = df[column].apply(func)

    scaled_model_coordinates = {index: row.tolist()
                                for index, row in df.iterrows()}
    return scaled_model_coordinates


class weighted_MSELoss(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
    def forward(self,inputs,targets):
        """Computes the weighted mean squared error loss.

        Args:
            input (torch.Tensor): Model outputs
            target (torch.Tensor): Model targets
            weights (torch.Tensor): Weights for each feature

        Returns:
            torch.Tensor: Weighted mean squared error loss
        """
        loss =  ((inputs - targets)**2 )*self.weights
        return torch.sqrt(loss.mean())
    
