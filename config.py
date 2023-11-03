seed = 42

data_dir = "data/covid/"

feature_cols = ['sequence', 'structure', 'predicted_loop_type']
target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C']

mapping = {x:i for i, x in enumerate('().ACGUBEHIMSX')}

train_seq_len = 107
test_seq_len = 130


batch_size = 64
n_epochs = 5
learning_rate = 0.001


#* model
embedding_dim = 150
hidden_dim = 256
num_layers = 1
bidirectional = True
dropout = 0.3


#* early stopping
early = False
patience = 2
min_delta = 0