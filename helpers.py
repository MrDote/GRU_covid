import os
import tempfile
import numpy as np
import pandas as pd

from ray import train
from ray.train import Checkpoint, ScalingConfig

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from torchinfo import summary

from typing import List
from GRU import EarlyStopper

from config import *



def filter_signal_noise(df: pd.DataFrame):
    # return df[df["signal_to_noise"] >= 1].reset_index()
    return df[df["SN_filter"] == 1].reset_index()


def preprocess(df: pd.DataFrame, cols: List):
    for col in cols:
        df.loc[:, col] = df[col].map(lambda x: [mapping[i] for i in x])
    return df


def to_np_array(df: pd.DataFrame, cols: List, dtype: np.dtype = np.int32):
    return np.array(df[cols].values.tolist(), dtype=dtype)


def convert_transpose(t: np.ndarray):
    return torch.from_numpy(t).transpose(1, 2)


def MCRMSE(true: torch.Tensor, preds: torch.Tensor):
    # weights = torch.tensor([0.3, 0.3, 0.3, 0.05, 0.05])
    # individual_batch = torch.mean(torch.sqrt(torch.mean(torch.square(true - preds), dim=1) * weights), dim=1)
    individual_batch = torch.mean(torch.sqrt(torch.mean(torch.square(true - preds), dim=1)), dim=1)
    return torch.mean(individual_batch)


def run_summary(model: nn.Module, batch_size = 20, seq_scored = 107, n_feature_cols = 3):
    summary(model, input_data=torch.randint(0, len(mapping), (batch_size, seq_scored, n_feature_cols)))


def post_process(model: nn.Module, pub_dataset: pd.DataFrame, priv_dataset: pd.DataFrame, weights):
    model.load_state_dict(weights)
    model.eval()

    with torch.no_grad():

        final_preds = pd.DataFrame()

        #* make predictions on test data
        for df in [pub_dataset, priv_dataset]:
            X = preprocess(df, feature_cols)
            X = to_np_array(df, feature_cols, np.int32)
            X = convert_transpose(X)

            #* make predictions on the whole set
            preds: np.ndarray = model(X).numpy()

            df_preds = pd.DataFrame()

            #* make a df from every prediction
            for (i, uid) in enumerate(df["id"]):
                pred = preds[i]

                new_df = pd.DataFrame(pred, columns=target_cols)
                new_df["id_seqpos"] = [f"{uid}_{index}" for index in range(len(pred))]
                new_df = new_df.set_index("id_seqpos")
                
                df_preds = pd.concat([df_preds, new_df])
            
            final_preds = pd.concat([final_preds, df_preds])

        return final_preds


def train_model(model: nn.Module, train_loader: data.DataLoader, test_dataset: data.TensorDataset, optimizer: optim.Optimizer, n_epochs: int = n_epochs, early_stopping: bool = early):
    weights = None
    train_loss = []
    test_loss = []

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7)

    if early_stopping:
        early_stopper = EarlyStopper(patience, min_delta)

    model.train()
    print("=> Starting training")

    for epoch in range(n_epochs):

        print(f"Epoch: {epoch+1}")

        total_loss = 0
        total_samples = 0

        model.train()

        #* looping over each batch
        for X_batch, y_batch in train_loader:
            
            y_pred = model(X_batch)

            loss = MCRMSE(y_pred, y_batch)

            total_loss += loss.item() * len(X_batch)
            total_samples += len(X_batch)

            model.zero_grad()
            loss.backward()
            optimizer.step()
        
        average_epoch_loss = total_loss / total_samples
        print(f"train loss: {average_epoch_loss}")

        train_loss.append(average_epoch_loss)


        model.eval()
        with torch.no_grad():
            X = test_dataset.tensors[0]
            y = test_dataset.tensors[1]
            
            y_pred = model(X)

            l = MCRMSE(y_pred, y).item()
        
            test_loss.append(l)
            print(f"test loss: {l}")

            if early_stopping:
                if early_stopper.early_stop(l):
                    break
            
                if early_stopper.counter == 0:
                    weights = model.state_dict()
            
            else:
                weights = model.state_dict()

            # with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            #     name = train.get_context().get_trial_name()

            #     torch.save(
            #         model.state_dict(),
            #         os.path.join(temp_checkpoint_dir, "weights.pt"),
            #     )
                
            #     checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)

            #     train.report({"loss": l}, checkpoint = checkpoint)
        
            
            # torch.save(weights, "./weights/weights.pt")
            
        scheduler.step(test_loss[-1])

    return weights, train_loss, test_loss
