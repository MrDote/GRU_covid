import torch
import torch.nn as nn
from torchinfo import summary
import torch.optim as optim

import numpy as np

# import random
# random.seed(42)
# torch.manual_seed(42)


# import nlkt



#* 2 words to the left, 2 to the right

CONTEXT_SIZE = 3

raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure spirits of computer with our spells.""".split()


vocab = set(raw_text)
vocab_size = len(vocab)

# print(vocab_size)

word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {v: k for k, v in word_to_ix.items()}

data = []

for i in range(CONTEXT_SIZE, len(raw_text) - CONTEXT_SIZE):
    context = (
        [raw_text[i - j - 1] for j in range(CONTEXT_SIZE)]
        + [raw_text[i + j + 1] for j in range(CONTEXT_SIZE)]
    )
    target = raw_text[i]
    data.append((context, target))




class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=0)


    def forward(self, inputs):
        # x = self.embeddings(inputs).mean(dim=0)
        x = torch.sum(self.embeddings(inputs), dim=0)
        # print(x.shape)
        x = self.linear(x)
        x = self.log_softmax(x)
        return x






# test_input = torch.tensor([word_to_ix[w] for w in data[0][0]], dtype=torch.long)


model = CBOW(vocab_size, 20, CONTEXT_SIZE)
# print(model(test_input))
# summary(model, input_data=torch.randint(0, 10, (1, 4), dtype=torch.long))


loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
losses = []


for epoch in range(1):
    total_loss = 0
    for context, target in data:

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in tensors)
        context_idxs = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
        print(context_idxs)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_idxs)
        # print(log_probs.shape)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a tensor)
        # print(nn.functional.one_hot(torch.tensor([word_to_ix[target]], dtype=torch.long)))

        # print(ix_to_word[log_probs.argmax().item()])
        # print(target)
        # print('----------------------')
        loss = loss_function(log_probs, torch.tensor(word_to_ix[target]))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()
    losses.append(total_loss)


# print(losses)


def predict_word(words):
    words = torch.tensor([word_to_ix[w] for w in context], dtype=torch.long)
    output = model(words)
    res = torch.exp(output)
    top_args = np.argpartition(res, -4)[-4:]

    # print(res)
    # print(top_args)

    # print(ix_to_word[res.argmax().item()])
    print([ix_to_word[i.item()] for i in top_args])



#* ans: other
with torch.no_grad():
    input = 'they As processes manipulate'.split()
    print(input)
    predict_word(input)