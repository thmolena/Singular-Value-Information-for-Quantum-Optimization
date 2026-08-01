# Configurations

The audit has no hidden configuration file. Registered constants are explicit
in `src/hqml_spectral_qaoa/experiment.py`: the 48 BFS starts, 9-by-9 angle
grid, four isomorphism-grouped folds, ridge penalty grid, rank six, eight
landmarks, Gaussian bandwidth four, and 5% sampled-row gate. Hyperparameters
are selected on development groups only; target objective labels are not
configuration inputs.
