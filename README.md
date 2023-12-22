# CMM-RT
This code implements methods for the accurate prediction of Retention Times 
(RTs) for a given Chromatographic Method (CM) using machine learning, as 
described in:

> García, C.A., Gil-de-la-Fuente, A., Barbas, C. et al. Probabilistic metabolite annotation using retention time prediction and meta-learned projections. J Cheminform 14, 33 (2022). https://doi.org/10.1186/s13321-022-00613-8. 


Used to create the yml file:
conda env create -f environment.yml

You can:
conda env create -f environment.yml --name cmmrt_env
conda activate cmmrt_env






Finally, note that you may find additional options for running your experiments
by consulting the `help` option of the Python scripts. E.g.:
```bash
$(PYTHON_INTERPRETER) cmmrt/rt/train_model.py --help
# usage: train_model.py [-h] [--storage STORAGE] [--study STUDY] [--train_size TRAIN_SIZE]
#                       [--param_search_folds PARAM_SEARCH_FOLDS] [--trials TRIALS] [--smoke_test]
#                       [--random_state RANDOM_STATE] [--save_to SAVE_TO]
# 
# Train blender and all base-models
# 
# optional arguments:
#   -h, --help            show this help message and exit
#   --storage STORAGE     SQLITE DB for storing the results of param search (e.g.: sqlite:///train.db
#   --study STUDY         Study name to identify param search results withing the DB
#   --train_size TRAIN_SIZE
#                         Percentage of the training set to train the base classifiers. The remainder
#                         is used to train the meta-classifier
#   --param_search_folds PARAM_SEARCH_FOLDS
#                         Number of folds to be used in param search
#   --trials TRIALS       Number of trials in param search
#   --smoke_test          Use small model and subsample training data for quick testing.
#                         param_search_folds and trials are also overridden
#   --random_state RANDOM_STATE
#                         Random state for reproducibility or reusing param search results
#   --save_to SAVE_TO     folder where to save the preprocessor and regressor models
```


