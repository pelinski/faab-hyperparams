import argparse
import yaml
import torch
import numpy as np
from bokeh.plotting import figure
from bokeh.models.widgets import Panel, Tabs
from bokeh.models import Legend
from bokeh.embed import file_html
from bokeh.resources import CDN


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
                   "learning_rate": hp["learning_rate"] if "learning_rate" in hp else args.learning_rate,
                   "scheduler_step_size": hp["scheduler_step_size"] if "scheduler_step_size" in hp else args.scheduler_step_size,
                   "scheduler_gamma": hp["scheduler_gamma"] if "scheduler_gamma" in hp else args.scheduler_gamma,
                   "num_encoder_layers": hp["num_encoder_layers"] if "num_encoder_layers" in hp else args.num_encoder_layers,
                   "plotter_samples": hp["plotter_samples"] if "plotter_samples" in hp else args.plotter_samples,
                   "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
