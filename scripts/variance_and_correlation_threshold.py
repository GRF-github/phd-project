import bz2
import pandas
import pickle
import numpy

print("Data loading")
with bz2.BZ2File("./resources/descriptors_and_fingerprints.pklz", "rb") as f:
    X, y, desc_cols, fgp_cols = pickle.load(f)

# Separate descriptors and fingerprints:
descriptors_array = X[:, :desc_cols[-1]]
fingerprints_array = X[:, fgp_cols[0]:]

# Convert descriptors and fingerprints to DataFrames
descriptors_df = pandas.DataFrame(descriptors_array)

# Convert fingerprints_array to DataFrame
fingerprints_df = pandas.DataFrame(fingerprints_array)

