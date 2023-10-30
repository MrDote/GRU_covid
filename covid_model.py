import torch

import torch.utils.data as data
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from helpers import *
from config import *
from GRU import *


# np.random.seed(seed) 
# torch.manual_seed(seed)


# TODO: try weight init


#! DATA

#* combined mapping



train = pd.read_json(data_dir + 'train.json', lines=True)


train = filter_signal_noise(train)
train = preprocess(train, feature_cols)


x_train, x_test, y_train, y_test = train_test_split(to_np_array(train, feature_cols, np.int32), to_np_array(train, target_cols, np.float32), test_size=.1, random_state=34)


x_train, x_test, y_train, y_test = convert_transpose(x_train), convert_transpose(x_test), convert_transpose(y_train), convert_transpose(y_test)



dataset_train = data.TensorDataset(x_train, y_train)
train_loader = data.DataLoader(dataset_train, batch_size, shuffle = True)


dataset_test = data.TensorDataset(x_test, y_test)








#! MODEL

model = GRU()
# model = GRU(68)


# weights, train_err, test_err = train_model(model, train_loader, dataset_test)


#* save model weights
# torch.save(weights, 'weights/covid/weights.pt')
# weights = torch.load('weights/covid/weights.pt')


# print(train_err)
# print(test_err)




# run_summary(model)





#! POST-PROCESSING

# test = pd.read_json(data_dir + 'test.json', lines=True)
# test_pub = test[test["seq_length"] == train_seq_len]
# test_priv = test[test["seq_length"] == test_seq_len]

# final_preds = post_process(model, test_pub, test_priv, weights)
# final_preds.to_csv("results/covid.csv.gz", compression="gzip")