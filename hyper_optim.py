import torch

from GRU import GRU
from config import *
from helpers import *
from loaders import make_loaders

from ray import train, tune
from ray.tune.schedulers import ASHAScheduler



def objective_wrapper(num_samples):
    train_loader, dataset_test = make_loaders(num_samples)
    
    def objective(config):
        model = GRU(68, embedding_dim, hidden_dim, dropout=config["dropout"], num_layers=config["num_layers"])
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["lr"]
        )

        while True:
            _, _, test_err = train_model(model, train_loader, dataset_test, optimizer, n_epochs=1)
            train.report({"loss": test_err[-1]})

    return objective


def create_tuner(algo, search_space, samples=n_samples, n_models = 1, max_epochs = 1):
    return tune.Tuner(
        objective_wrapper(samples),
        tune_config=tune.TuneConfig(
            metric="loss",
            num_samples=n_models,
            mode="min",
            search_alg=algo,
            scheduler=ASHAScheduler(
                time_attr="training_iteration",
                grace_period=5,
                max_t=max_epochs,
                reduction_factor=2,
                brackets=1
            ),
        ),
        # run_config=train.RunConfig(
        #     stop={"training_iteration": max_epochs},
        # ),
        param_space=search_space,
    )