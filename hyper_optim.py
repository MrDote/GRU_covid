import torch

from GRU import GRU
from config import *
from helpers import *
from loaders import make_loaders

from ray import train, tune
from ray.tune.schedulers import ASHAScheduler



# class MyTuner(tune.Tuner):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.best_model = None

#     def save_checkpoint(self, checkpoint_dir):
#         if self.best_model is not None:
#             self.best_model.save_weights(checkpoint_dir + "/best_model_weights.h5")

#     def on_trial_result(self, trial_id, result):
#         if self.best_model is None or result[self.checkpoint_score_attr] < self.best_model_score:
#             self.best_model = self.get_best_model()
#             self.best_model_score = result[self.checkpoint_score_attr]




def objective_wrapper(num_samples):
    train_loader, dataset_test = make_loaders(num_samples)
    
    def objective(config):
        model = GRU(68, embedding_dim, hidden_dim, dropout=config["dropout"], num_layers=config["num_layers"])
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["lr"]
        )

        while True:
            train_model(model, train_loader, dataset_test, optimizer, n_epochs=1, lr_name=round(config["lr"], 8))

    return objective



def create_tuner(algo, search_space, samples=n_samples, n_models = 1, max_epochs = 1, grace_period = 0, reduction_factor = 2):
    return tune.Tuner(
        objective_wrapper(samples),
        tune_config=tune.TuneConfig(
            metric="loss",
            num_samples=n_models,
            mode="min",
            search_alg=algo,
            scheduler=ASHAScheduler(
                time_attr="training_iteration",
                grace_period=grace_period,
                max_t=max_epochs,
                reduction_factor=reduction_factor,
                brackets=1
            ),
        ),
        run_config=train.RunConfig(
            verbose=0
        ),
        param_space=search_space
    )