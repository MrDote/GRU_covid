import torch

from GRU import GRU
from config import *
from helpers import *
from loaders import make_loaders

from ray import train, tune


def objective_wrapper():
    train_loader, dataset_test = make_loaders(100)
    
    def objective(config):
        model = GRU(68, embedding_dim, hidden_dim, dropout=config["dropout"])
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["lr"]
        )

        while True:
            _, _, test_err = train_model(model, train_loader, dataset_test, optimizer, 1)
            train.report({"loss": test_err[-1]})

    return objective


def create_tuner(algo, search_space):
    return tune.Tuner(
        objective_wrapper(),
        tune_config=tune.TuneConfig(
            metric="loss",
            num_samples=2,
            mode="min",
            search_alg=algo,
        ),
        run_config=train.RunConfig(
            stop={"training_iteration": 5},
        ),
        param_space=search_space,
    )