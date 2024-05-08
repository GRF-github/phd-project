import optuna

study = optuna.create_study(
    study_name="cv-fold-0-fgp_mlp",
    direction='maximize',
    storage="sqlite:////home/guillermo/PycharmProjects/cmmrt/results/IWWBIO_results/cv.db",
    load_if_exists=True,
    pruner=optuna.pruners.MedianPruner()
)

trials = study.trials

params_list = []
for trial in trials:
    params = trial.params
    params["score"] = trial.values[0]
    params_list.append(params)

import pandas as pd
pd.DataFrame(params_list).to_csv("/tmp/test.csv")