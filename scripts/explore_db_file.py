import optuna
import pandas as pd

params_list = []
for i in range(0, 5):
    for j in ["fgp_mlp", "desc_mlp", "full_mlp"]:

        study = optuna.create_study(
            study_name=f"cv-fold-{i}-{j}",
            direction='maximize',
            storage="sqlite:///./results/IWWBIO_results/cv.db",
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner()
        )

        trials = study.trials

        params = {}
        params['fold'] = i
        params['feature'] = j

        for trial in trials:
            params.update(trial.params)

            if trial.values is not None:
                params["score"] = trial.values[0]

            params_list.append(params)

pd.DataFrame(params_list).to_csv(f"./results/optuna_parms.csv")
