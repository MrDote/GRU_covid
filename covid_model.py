import os
os.environ['RAY_AIR_NEW_OUTPUT'] = '0'

from ray import tune
from ray.tune.search.optuna import OptunaSearch

from config import *
from loaders import make_loaders
from helpers import train_model
from hyper_optim import *
from GRU import GRU


# np.random.seed(seed)
# torch.manual_seed(seed)


# TODO:
#* weight init
#* ray tune for param optimization: add early stopping to stop criteria
#* check optim.step returning loss



#! CHECK MODEL ARCHITECTURE

# model = GRU_model()

# run_summary(model)







#! HYPERPARAMETER OPTIMIZATION

search_space = {"lr": tune.loguniform(5e-5, 5e-4),
                "dropout": tune.uniform(0.2, 0.5),
                "num_layers": tune.randint(1, 3)
                }

algo = OptunaSearch()


tuner: tune.Tuner = create_tuner(algo, search_space, samples = 300, n_models = 5, max_epochs = 10)

results = tuner.fit()
print("Best config is:", results.get_best_result().config)









#! TRAINING MODEL

# model = GRU(68)

# train_loader, dataset_test = make_loaders(n_samples)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# early_stopper = EarlyStopper(patience, min_delta)

# weights, train_err, test_err = train_model(model, train_loader, dataset_test, optimizer)


#* save model weights
# torch.save(weights, 'weights/covid/weights.pt')

#* print training and testing errors
# print(train_err)
# print(test_err)












#! POST-PROCESSING

# test = pd.read_json(data_dir + 'test.json', lines=True)
# test_pub = test[test["seq_length"] == train_seq_len]
# test_priv = test[test["seq_length"] == test_seq_len]

# weights = torch.load('weights/covid/weights.pt')
# model = GRU()

# final_preds = post_process(model, test_pub, test_priv, weights)
# final_preds.to_csv("results/covid.csv.gz", compression="gzip")