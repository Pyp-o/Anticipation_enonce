import optuna

# PART 1 get important results
"""
def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2

study = optuna.create_study()
study.optimize(objective, n_trials=100)

best_params = study.best_params
best_value = study.best_value
best_trial = study.best_trials
all_trials = study.trials
n_trials = len(study.trials)
found_x = best_params["x"]
print(n_trials)

study.optimize(objective, n_trials=100)
n_trials = len(study.trials)
print(n_trials)
found_x = best_params["x"]
print(found_x)
"""

#PART 2 initialize all hyperparameters
"""
def objective(trial):
    # Categorical parameter
    optimizer = trial.suggest_categorical("optimizer", ["MomentumSGD", "Adam"])

    # Integer parameter
    num_layers = trial.suggest_int("num_layers", 1, 3)

    # Integer parameter (log)
    num_channels = trial.suggest_int("num_channels", 32, 512, log=True)

    # Integer parameter (discretized)
    num_units = trial.suggest_int("num_units", 10, 100, step=5)

    # Floating point parameter
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 1.0)

    # Floating point parameter (log)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    # Floating point parameter (discretized)
    drop_path_rate = trial.suggest_float("drop_path_rate", 0.0, 1.0, step=0.1)

study = optuna.create_study()
study.optimize(objective, n_trials=1)
"""

#PART 3 change sampler (grid search, random search, etc)
"""
study = optuna.create_study() #default sampler is TPE (Tree-structured Parzen Estimator)
print(f"Sampler is {study.sampler.__class__.__name__}")

study = optuna.create_study(sampler=optuna.samplers.RandomSampler())
print(f"Sampler is {study.sampler.__class__.__name__}")

study = optuna.create_study(sampler=optuna.samplers.CmaEsSampler())
print(f"Sampler is {study.sampler.__class__.__name__}")

study = optuna.create_study(sampler=optuna.samplers.GridSampler)
print(f"Sampler is {study.sampler.__class__.__name__}")
"""

#PART4 pruning algorithms (early stop)
# je fais un changement pour voir