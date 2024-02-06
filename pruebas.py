import pandas as pd
import bz2
import pickle

# Load the data from fingerprints.pklz
file_path = './rt_data/fingerprints.pklz'

# Load the compressed pickle file using bz2
with bz2.BZ2File(file_path, 'rb') as f:
    # Load the pickled data
    data = pickle.load(f)

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

# Print the names of the first four columns
print("First four columns:")
print(df.columns[-8:].tolist())

