import torch
import wandb
import os
import numpy as np
from tqdm import tqdm
from model import TransformerAutoencoder, LSTM
from dataset import Dataset, DatasetPred
from utils import get_html_plot, load_hyperparams


hyperparameters, hp_path = load_hyperparams()

os.environ["WANDB_MODE"] = "offline"  # testing

with wandb.init(project=hyperparameters["project"], config=hyperparameters,  settings=wandb.Settings(start_method="fork"), job_type="train"):

    config = wandb.config
    print(config)

    wandb.save(hp_path)

    if config.pred:
        dataset = Dataset().load_dataset_from_pickle(pickle_path=config.pickle_path)
    else:
        dataset = DatasetPred().load_dataset_from_pickle(pickle_path=config.pickle_path)

    if torch.isnan(dataset.inputs).any():
        print("Dataset contains NaN values")
        exit()

    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False)

    if config.model == "lstm":
        model = LSTM(feat_in_size=config.feat_in_size, feat_out_size=config.feat_in_size, ff_size=config.ff_size, num_layers=config.num_layers,
                     hidden_size=config.d_model, seq_len=config.seq_len, dropout=config.dropout, proj_size=config.proj_size).to(config.device)
    else:
        model = TransformerAutoencoder(d_model=config.d_model, feat_in_size=config.feat_in_size, num_heads=config.num_heads, ff_size=config.ff_size,
                                       dropout=config.dropout, num_layers=config.num_layers, max_len=config.seq_len, pe_scale_factor=config.pe_scale_factor, mask=config.mask).to(config.device)

    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=config.learning_rate)
    else:
        raise ValueError("Invalid optimizer")

    if config.scheduler_step_size > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
    else:
        scheduler = None

    criterion = torch.nn.MSELoss(reduction='mean')

    print("Running on device: {}".format(config.device))
    last_10_avg_batch_losses = []
    for epoch in range(1, config.epochs+1):

        print("\033[1m Epoch: {}".format(epoch))

        # training loop
        model.train()
        train_it_losses = np.array([])
        html, plotter = "", 1  # plot only the sample of the first batch of the test set
        for batch_idx, (train_inputs, train_targets) in enumerate(tqdm(train_dataloader)):

            # (batch_size, seq_len, feature_size)
            train_inputs, train_targets = train_inputs.to(
                device=config.device, non_blocking=True), train_targets.to(
                device=config.device, non_blocking=True)

            if config.model == "lstm":
                out, _ = model(train_inputs)
            else:
                out = model(train_inputs)

            optimizer.zero_grad(set_to_none=True)  # lower memory footprint
            train_loss = torch.sqrt(criterion(out, train_targets))
            train_it_losses = np.append(train_it_losses, train_loss.item())
            train_loss.backward()

            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.max_grad_norm)  # Gradient clipping
            optimizer.step()
            if config.scheduler_step_size > 0:
                scheduler.step()

            # plot and save
            if plotter:
                plotter = 0
                html = get_html_plot(
                    out[:10], train_inputs[:10], dataset.feature_names)
            if epoch % 50 == 0:
                save_filename = os.path.join(
                    wandb.run.dir, f'transformer_run_{wandb.run.id}_{epoch}.model')
                torch.save(model.state_dict(), save_filename)
                wandb.save(save_filename, base_path=wandb.run.dir)

        if torch.isnan(train_loss).any():
            print("train loss is nan :(")
            break

        # mean of the batch losses
        losses = {"train_loss": train_it_losses.mean().round(8)}
        wandb.log({"epoch": epoch, **losses,
                  "plotter_samples": wandb.Html(html)})
        print(losses)

        # If the list length is 10 and all losses are the same, stop the training
        last_10_avg_batch_losses.append(train_it_losses.mean().item())
        if len(last_10_avg_batch_losses) > 10:
            last_10_avg_batch_losses.pop(0)
        # all elements are the same to the 8th decimal (rounded above)
        if len(last_10_avg_batch_losses) == 10 and len(set(last_10_avg_batch_losses)) == 1:
            print("Loss hasn't changed for the last 10 epochs. Stopping training.")
            break
