import torch.nn as nn

from config import *



class GRU(nn.Module):
    def __init__(self, pred_len=None, embedding_dim=embedding_dim, hidden_dim=hidden_dim, dropout=dropout, num_layers=num_layers, bidirectional=bidirectional, in_size=3, out_size=5):
        super().__init__()
        self.embeddings = nn.Embedding(len(mapping), embedding_dim)
        self.gru = nn.GRU(embedding_dim*in_size, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout1d(p=dropout)
        self.linear = nn.Linear(hidden_dim*2, out_size)

        self.pred_len = pred_len


    def forward(self, inputs):
        x = self.embeddings(inputs)
        x = x.reshape((x.shape[0], x.shape[1], -1))
        x, _ = self.gru(x)
        x = self.dropout(x)
        x = self.linear(x)

        if self.pred_len:
            x = x[:, : self.pred_len, :]
    
        return x



class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')


    def early_stop(self, validation_loss):
        print(f"Counter: {self.counter}")
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False