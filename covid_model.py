import os
os.environ['RAY_AIR_NEW_OUTPUT'] = '0'

# import torch

from ray import train, tune
from ray.tune.search.optuna import OptunaSearch

from config import *
from loaders import make_loaders
from hyper_optim import objective_wrapper
from GRU import GRU


# np.random.seed(seed) 
# torch.manual_seed(seed)


# TODO:
#* weight init
#* ray tune for param optimization: add early stopping to stop criteria
#* check optim.step returning loss


#! DATA

#* combined mapping








#! MODEL

# model = GRU_model()
# model = GRU_model(68)

# train_loader, dataset_test = make_loaders()
# weights, train_err, test_err = train_model(model, train_loader, dataset_test)


#* save model weights
# torch.save(weights, 'weights/covid/weights.pt')
# weights = torch.load('weights/covid/weights.pt')


# print(train_err)
# print(test_err)



# run_summary(model)








#! HYPERPARAMETER OPTIMIZATION

search_space = {"lr": tune.loguniform(1e-4, 1e-3),
                "dropout": tune.uniform(0.2, 0.5)}

algo = OptunaSearch()

tuner = tune.Tuner(
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


results = tuner.fit()
print("Best config is:", results.get_best_result().config)









#! POST-PROCESSING

# test = pd.read_json(data_dir + 'test.json', lines=True)
# test_pub = test[test["seq_length"] == train_seq_len]
# test_priv = test[test["seq_length"] == test_seq_len]

# final_preds = post_process(model, test_pub, test_priv, weights)
# final_preds.to_csv("results/covid.csv.gz", compression="gzip")