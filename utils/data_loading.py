"""Functions to load the SMRT dataset and fingerprints/descriptors computed with Alvadesc"""
import bz2
import os
import pickle
import numpy as np
import pandas as pd
from utils.cure_descriptors_and_fingerprints import cure


def get_my_data(common_cols):
    # Check if we have the file with both databases already merged, and if not, merge them
    if os.path.exists("../resources/descriptors_and_fingerprints.pklz"):
        with bz2.BZ2File("../resources/descriptors_and_fingerprints.pklz", "rb") as f:
            X, y, desc_cols, fgp_cols = pickle.load(f)
    else:
        raw_descriptors = pd.read_csv("../resources/metlin_descriptors_raw.csv")
        raw_fingerprints = pd.read_csv("../resources/metlin_fingerprints_raw.csv")

        # Remove bloat columns and add a number for identification and the correct ccs
        descriptors, fingerprints = cure(raw_descriptors, raw_fingerprints)

        print('Merging')
        descriptors = descriptors.drop_duplicates()
        descriptors_and_fingerprints = pd.merge(descriptors, fingerprints, on=common_cols)

        X_desc = descriptors_and_fingerprints[descriptors.drop(common_cols, axis=1).columns].values
        X_fgp = descriptors_and_fingerprints[fingerprints.drop(common_cols, axis=1).columns].values

        X = np.concatenate([X_desc, X_fgp], axis=1)
        y = descriptors_and_fingerprints['correct_ccs_avg'].values.flatten(),
        desc_cols = np.arange(X_desc.shape[1], dtype='int'),
        fgp_cols = np.arange(X_desc.shape[1], X.shape[1], dtype='int')

        # for key in data:
        #     setattr(data, key, data[key])

        print('Saving')
        with bz2.BZ2File("../resources/descriptors_and_fingerprints.pklz", "wb") as f:
            pickle.dump([X, y, desc_cols, fgp_cols], f)

    X = X.astype('float32')
    y = np.array(y).astype('float32').flatten()

    return X, y, desc_cols, fgp_cols






