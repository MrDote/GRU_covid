from ray import tune
from ray.tune.search.optuna import OptunaSearch

from config import *
from hyper_optim import *
from GRU import GRU


if seed:
    np.random.seed(seed)
    torch.manual_seed(seed)


# TODO:
#* scored columns higher weighting



#! CHECK MODEL ARCHITECTURE

# model = GRU()

# run_summary(model)







#! HYPERPARAMETER OPTIMIZATION

# search_space = {
#                 # "lr": tune.loguniform(6e-4, 5e-3),
#                 # "lr": tune.choice([3e-4, 5e-4, 7e-4, 9e-4]),
#                 "lr": 0.0013,
#                 # "dropout": tune.uniform(0.2, 0.5),
#                 "dropout": tune.choice([0.2, 0.21]),
#                 # "num_layers": tune.randint(1, 3)
#                 "num_layers": 2
#                 }

# algo = OptunaSearch()


# tuner: tune.Tuner = create_tuner(algo, search_space, samples = None, n_models = 8, max_epochs = 30, grace_period = 5, reduction_factor=2)


# results = tuner.fit()
# print("Best config is:", results.get_best_result().config)
# print(results.get_dataframe())









#! TRAINING MODEL

model = GRU(68)

train_loader, dataset_test = make_loaders(None)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

weights, train_err, test_err = train_model(model, train_loader, dataset_test, optimizer, n_epochs=60, early_stopping=False)


#* save model weights
torch.save(weights, "weights/weights.pt")

#* print training and testing errors
print(train_err)
print(test_err)


#* plot loss
import matplotlib.pyplot as plt

plt.plot(train_err)
plt.plot(test_err)

plt.show()






#! POST-PROCESSING


# test = pd.read_json(data_dir + 'test.json', lines=True)
# test_pub = test[test["seq_length"] == train_seq_len]
# test_priv = test[test["seq_length"] == test_seq_len]

# weights = torch.load('weights/weights.pt')
# # weights = torch.load("/Users/antonbelov/ray_results/objective_2023-11-10_16-57-31/objective_4a6a9086/checkpoint_000009/objective_4a6a9086.pt")
# model = GRU()

# final_preds = post_process(model, test_pub, test_priv, weights)
# final_preds.to_csv("results/covid.csv.gz", compression="gzip")