# COVID-19 mRNA Vaccine Degradation Prediction Submission

This is the model used to predict the mRNA vaccine degradation rates at different locations along various RNA sequences as part of the Kaggle competition.


## Model

The model used is a bidirectional GRU with an Embedding layer before and Linear layer after it. Dropout and early stopping is added later on. The full list of suggestions (implemented and left to explore) to improve the model's performance can be found [below](#performance-improvement-suggestions).



## Required Packages

- Ray
- PyTorch
- NumPy
- Pandas
- Torchinfo
- Typing
- Scikit-learn




## Performance improvement suggestions

1. Different target column weighting (some more important to get right than others)

2. Schedulers + EarlyStoppers

3. Dropout (internal + between layers)

4. Stratify dataset when train_test_split

5. Errors: 
    - NaN individual positions with large errors (error > max_err & value/error < 1.5) + edit loss_fn so NaNs don't contribute
    - Randomly perturb features within the error boundaries

6. Contribution of samples within clusters: 
    - Lower for dense clusters: calculate edit (Levenshtein) distance -> cluster -> multiply sample weights by 1/sqrt(count_in_cluster)
    - Increase for similar to eval dataset
    - Use all samples, but lowe based on signal_to_noise


8. Training augmentation and test time augmentation: reversed sequences (+ labels/other features), generate possible structure yourself (& reverse)

9. Pseudo-labeling: unlabeled data is labelled by supervised-learnt network -> retrain network based on og + pseudo-labels
    Check what data works best (i.e. lower variance for specific [sequences of] entries between different single models) 
    Apply same suggestions as above to pseudo-labels
    Switch training on actual and PL (5:2); skip epochs if performance significantly deteriorates OR few epochs of PL at beginning of each fold
    Check if eval accuracy is deteriorated
    Focus on model correlation (diversity), rather than feature/model picking
    Do same for randomly generated data/sequences

10. Clustering methods on all single models -> single mean with each cluster -> assign weights to aggregate these clusters

11. Weighted blending (ensemble learning)

12. 2nd stage stacking

13. Applied weight optimization

14. XGBoost

15. k-fold stratification (CV)

16. Feature engineering: matrices to specify the neighbors of each node's pair; structure adjacency matrix; distance to key positions (pairs, breaks etc); separate node and edge features; angle information; spectral representations; entropy features

17. Networks to try: AE pretrain, Attn, GCN layer

18. Weighted average of variation of n models

19. Feature extraction: eternafold, vienna, nupack, contrafold and rnasoft

20. ARNIE: software package of 6 libraries for generating xtra info for each sequence -> Base Pairing Probability matrix

21. 3D foldings + distance between RNA bases