# NLP

## Continuous Bag of Words (CBOW)

Aim: predict target words based on context (words around the target word)

Architecture / layers:
    - Input -> embedding (or one-hot vector), each element is specific to feature of embedding (or individual word)
    - Hidden -> FC n_neurons = n_features (or n_words)