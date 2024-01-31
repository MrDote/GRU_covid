import os
work_dir = os.getcwd()

seed = None

data_dir = "data/covid/"

feature_cols = ['sequence', 'structure', 'predicted_loop_type']
target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C']

mapping = {x:i for i, x in enumerate('().ACGUBEHIMSX')}

train_seq_len = 107
test_seq_len = 130


n_samples = 100
batch_size = 60
n_epochs = 50
learning_rate = 0.001


#* model
embedding_dim = 200
hidden_dim = 256
num_layers = 3
bidirectional = True
dropout_embeds = 0.2


#* early stopping (training)
early = False
patience = 7
min_delta = 0