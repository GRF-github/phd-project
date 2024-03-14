/home/grf/anaconda3/bin/python /home/grf/PycharmProjects/cmmrt/main.py 
Running smoke test...
Loading data
Starting tune_and_fit with data with dim (2000,7883)
Preprocessing...
Training with descriptors
/home/grf/anaconda3/lib/python3.11/site-packages/sklearn/utils/extmath.py:1051: RuntimeWarning: invalid value encountered in divide
  updated_mean = (last_sum + new_sum) / updated_sample_count
/home/grf/anaconda3/lib/python3.11/site-packages/sklearn/utils/extmath.py:1056: RuntimeWarning: invalid value encountered in divide
  T = new_sum / new_sample_count
/home/grf/anaconda3/lib/python3.11/site-packages/sklearn/utils/extmath.py:1076: RuntimeWarning: invalid value encountered in divide
  new_unnormalized_variance -= correction**2 / new_sample_count
/home/grf/anaconda3/lib/python3.11/site-packages/sklearn/impute/_base.py:555: UserWarning: Skipping features without any observed values: [161]. At least one non-missing value is needed for imputation with strategy='median'.
  warnings.warn(
/home/grf/anaconda3/lib/python3.11/site-packages/sklearn/impute/_base.py:555: UserWarning: Skipping features without any observed values: [161]. At least one non-missing value is needed for imputation with strategy='median'.
  warnings.warn(
Creating DNN
Param search
[I 2024-03-14 08:46:57,657] A new study created in RDB with name: descriptors-nnet-fold-0-dnn
Starting 1 trials
1512 128 0.5 0.1 gelu
1512 128 0.5 0.1 gelu
[I 2024-03-14 08:47:24,977] Trial 0 finished with value: -4.245265960693359 and parameters: {'hidden_1': 1512, 'T0': 92, 'hidden_2': 348, 'dropout_1': 0.577125787136993, 'dropout_2': 0.05335335013580811, 'activation': 'relu', 'lr': 0.0004295501064348733, 'annealing_rounds': 5, 'swa_epochs': 60, 'var_p': 0.9117266445691626}. Best is trial 0 with value: -4.245265960693359.
Training
1512 128 0.5 0.1 gelu
Saving preprocessor and DNN
/home/grf/anaconda3/lib/python3.11/site-packages/sklearn/impute/_base.py:555: UserWarning: Skipping features without any observed values: [161]. At least one non-missing value is needed for imputation with strategy='median'.
  warnings.warn(
Saving intermediate results:
Starting tune_and_fit with data with dim (2000,7883)
Preprocessing...
Training with descriptors
/home/grf/anaconda3/lib/python3.11/site-packages/sklearn/utils/extmath.py:1051: RuntimeWarning: invalid value encountered in divide
  updated_mean = (last_sum + new_sum) / updated_sample_count
/home/grf/anaconda3/lib/python3.11/site-packages/sklearn/utils/extmath.py:1056: RuntimeWarning: invalid value encountered in divide
  T = new_sum / new_sample_count
/home/grf/anaconda3/lib/python3.11/site-packages/sklearn/utils/extmath.py:1076: RuntimeWarning: invalid value encountered in divide
  new_unnormalized_variance -= correction**2 / new_sample_count
/home/grf/anaconda3/lib/python3.11/site-packages/sklearn/impute/_base.py:555: UserWarning: Skipping features without any observed values: [161]. At least one non-missing value is needed for imputation with strategy='median'.
  warnings.warn(
/home/grf/anaconda3/lib/python3.11/site-packages/sklearn/impute/_base.py:555: UserWarning: Skipping features without any observed values: [161]. At least one non-missing value is needed for imputation with strategy='median'.
  warnings.warn(
Creating DNN
Param search
[I 2024-03-14 08:47:57,133] A new study created in RDB with name: descriptors-nnet-fold-1-dnn
Starting 1 trials
1512 128 0.5 0.1 gelu
1512 128 0.5 0.1 gelu
[I 2024-03-14 08:48:09,284] Trial 0 finished with value: -4.160335540771484 and parameters: {'hidden_1': 2048, 'T0': 53, 'hidden_2': 504, 'dropout_1': 0.6426541657653468, 'dropout_2': 0.15701314962088936, 'activation': 'gelu', 'lr': 0.00036379170639746475, 'annealing_rounds': 4, 'swa_epochs': 24, 'var_p': 0.985402540576579}. Best is trial 0 with value: -4.160335540771484.
Training
1512 128 0.5 0.1 gelu
Saving preprocessor and DNN
/home/grf/anaconda3/lib/python3.11/site-packages/sklearn/impute/_base.py:555: UserWarning: Skipping features without any observed values: [161]. At least one non-missing value is needed for imputation with strategy='median'.
  warnings.warn(
Saving intermediate results:
Starting tune_and_fit with data with dim (2000,7883)
Preprocessing...
Training with descriptors + fingerprints
/home/grf/anaconda3/lib/python3.11/site-packages/sklearn/utils/extmath.py:1051: RuntimeWarning: invalid value encountered in divide
  updated_mean = (last_sum + new_sum) / updated_sample_count
/home/grf/anaconda3/lib/python3.11/site-packages/sklearn/utils/extmath.py:1056: RuntimeWarning: invalid value encountered in divide
  T = new_sum / new_sample_count
/home/grf/anaconda3/lib/python3.11/site-packages/sklearn/utils/extmath.py:1076: RuntimeWarning: invalid value encountered in divide
  new_unnormalized_variance -= correction**2 / new_sample_count
/home/grf/anaconda3/lib/python3.11/site-packages/sklearn/impute/_base.py:555: UserWarning: Skipping features without any observed values: [161]. At least one non-missing value is needed for imputation with strategy='median'.
  warnings.warn(
/home/grf/anaconda3/lib/python3.11/site-packages/sklearn/impute/_base.py:555: UserWarning: Skipping features without any observed values: [161]. At least one non-missing value is needed for imputation with strategy='median'.
  warnings.warn(
Creating DNN
Param search
[I 2024-03-14 08:48:25,290] A new study created in RDB with name: all-nnet-fold-0-dnn
Starting 1 trials
1512 128 0.5 0.1 gelu
1512 128 0.5 0.1 gelu
[I 2024-03-14 08:48:39,449] Trial 0 finished with value: -4.234142303466797 and parameters: {'hidden_1': 1024, 'T0': 76, 'hidden_2': 504, 'dropout_1': 0.43952405102723974, 'dropout_2': 0.008242625580012741, 'activation': 'gelu', 'lr': 0.0006919435019413977, 'annealing_rounds': 2, 'swa_epochs': 62, 'var_p': 0.9523263952785952}. Best is trial 0 with value: -4.234142303466797.
Training
1512 128 0.5 0.1 gelu
Saving preprocessor and DNN
/home/grf/anaconda3/lib/python3.11/site-packages/sklearn/impute/_base.py:555: UserWarning: Skipping features without any observed values: [161]. At least one non-missing value is needed for imputation with strategy='median'.
  warnings.warn(
Saving intermediate results:
Starting tune_and_fit with data with dim (2000,7883)
Preprocessing...
Training with descriptors + fingerprints
/home/grf/anaconda3/lib/python3.11/site-packages/sklearn/utils/extmath.py:1051: RuntimeWarning: invalid value encountered in divide
  updated_mean = (last_sum + new_sum) / updated_sample_count
/home/grf/anaconda3/lib/python3.11/site-packages/sklearn/utils/extmath.py:1056: RuntimeWarning: invalid value encountered in divide
  T = new_sum / new_sample_count
/home/grf/anaconda3/lib/python3.11/site-packages/sklearn/utils/extmath.py:1076: RuntimeWarning: invalid value encountered in divide
  new_unnormalized_variance -= correction**2 / new_sample_count
/home/grf/anaconda3/lib/python3.11/site-packages/sklearn/impute/_base.py:555: UserWarning: Skipping features without any observed values: [161]. At least one non-missing value is needed for imputation with strategy='median'.
  warnings.warn(
/home/grf/anaconda3/lib/python3.11/site-packages/sklearn/impute/_base.py:555: UserWarning: Skipping features without any observed values: [161]. At least one non-missing value is needed for imputation with strategy='median'.
  warnings.warn(
Creating DNN
Param search
[I 2024-03-14 08:48:58,261] A new study created in RDB with name: all-nnet-fold-1-dnn
Starting 1 trials
1512 128 0.5 0.1 gelu
1512 128 0.5 0.1 gelu
[I 2024-03-14 08:49:31,645] Trial 0 finished with value: -4.438678741455078 and parameters: {'hidden_1': 512, 'T0': 96, 'hidden_2': 194, 'dropout_1': 0.4055590680734231, 'dropout_2': 0.15816022573005328, 'activation': 'relu', 'lr': 0.00011171945732357645, 'annealing_rounds': 5, 'swa_epochs': 67, 'var_p': 0.9010780255081337}. Best is trial 0 with value: -4.438678741455078.
Training
1512 128 0.5 0.1 gelu
Saving preprocessor and DNN
/home/grf/anaconda3/lib/python3.11/site-packages/sklearn/impute/_base.py:555: UserWarning: Skipping features without any observed values: [161]. At least one non-missing value is needed for imputation with strategy='median'.
  warnings.warn(
Saving intermediate results:
Starting tune_and_fit with data with dim (2000,7883)
Preprocessing...
Training fingerprints
Creating DNN
Param search
[I 2024-03-14 08:50:09,546] A new study created in RDB with name: fingerprints-nnet-fold-0-dnn
Starting 1 trials
1512 128 0.5 0.1 gelu
1512 128 0.5 0.1 gelu
[I 2024-03-14 08:50:27,180] Trial 0 finished with value: -5.472141265869141 and parameters: {'hidden_1': 4096, 'T0': 88, 'hidden_2': 241, 'dropout_1': 0.3778126805963095, 'dropout_2': 0.01656127690096625, 'activation': 'leaky_relu', 'lr': 0.0006990636444916975, 'annealing_rounds': 4, 'swa_epochs': 74, 'var_p': 0.9639329811169135}. Best is trial 0 with value: -5.472141265869141.
Training
1512 128 0.5 0.1 gelu
Saving preprocessor and DNN
Saving intermediate results:
Starting tune_and_fit with data with dim (2000,7883)
Preprocessing...
Training fingerprints
Creating DNN
Param search
[I 2024-03-14 08:50:44,802] A new study created in RDB with name: fingerprints-nnet-fold-1-dnn
Starting 1 trials
1512 128 0.5 0.1 gelu
1512 128 0.5 0.1 gelu
[I 2024-03-14 08:50:50,855] Trial 0 finished with value: -5.442287445068359 and parameters: {'hidden_1': 1024, 'T0': 23, 'hidden_2': 246, 'dropout_1': 0.6981270463787431, 'dropout_2': 0.13096031356183355, 'activation': 'relu', 'lr': 0.0007966335294675382, 'annealing_rounds': 5, 'swa_epochs': 19, 'var_p': 0.9959355520742824}. Best is trial 0 with value: -5.442287445068359.
Training
1512 128 0.5 0.1 gelu
Saving preprocessor and DNN
Saving intermediate results:
Saving final results
        mae     medae      mape  fold      features
0  0.483156  0.196014  0.002238     0   descriptors
0  0.601092  0.325729  0.002866     1   descriptors
0  0.741114  0.385704  0.003534     0           all
0  0.421855  0.188080  0.002109     1           all
0  0.331962  0.191437  0.001631     0  fingerprints
0  0.691406  0.339706  0.003271     1  fingerprints

Process finished with exit code 0