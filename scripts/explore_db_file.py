import optuna
import pandas as pd


for i in range(0, 5):
    for j in ["fgp_mlp", "desc_mlp", "full_mlp"]:

        study = optuna.create_study(
            study_name=f"cv-fold-{i}-{j}",
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

        pd.DataFrame(params_list).to_csv(f"/home/guillermo/PycharmProjects/cmmrt/results/fold{i}_{j}.csv")
