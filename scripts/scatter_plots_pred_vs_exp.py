import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

# PARAMETERS
mre_threshold = 15
# ----------


# FIRST PART: explore predictions individually and add them to a list
data_files_list = os.listdir('/home/grf/PycharmProjects/cmmrt/results/scatter_plots')

pred_vs_exp_df_list = []
for file in data_files_list:
    # Read the CSV file into a DataFrame
    pred_vs_exp_df = pd.read_csv(f'/home/grf/PycharmProjects/cmmrt/results/scatter_plots/{file}', header=None)

    # Rename the columns
    pred_vs_exp_df.columns = ['index', 'exp', 'pred']

    # Create a scatter plot
    plt.figure()
    plt.scatter(pred_vs_exp_df['exp'], pred_vs_exp_df['pred'], s=2, c='blue', label='Data points')
    plt.xlabel('exp')
    plt.ylabel('pred')
    plt.title(f'Scatter Plot of {file[12:-4]}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'/home/grf/PycharmProjects/cmmrt/results/scatter_plots/{file[:-3]}png')

    # Add MRE column
    pred_vs_exp_df['MRE'] = 100 * np.abs(pred_vs_exp_df['pred'] - pred_vs_exp_df['exp']) / pred_vs_exp_df['exp']

    # Add dataframe to the dataframe list
    pred_vs_exp_df_list.append(pred_vs_exp_df)


# SECOND PART: identify each prediction
# Open metlin_fingerprints_raw.csv
fingerprints_df = pd.read_csv('/home/grf/PycharmProjects/cmmrt/resources/metlin_data/metlin_fingerprints_raw.csv')
fingerprints_df.dropna(subset=['Molecule Name'], inplace=True)
fingerprints_df.drop(fingerprints_df[fingerprints_df['Molecule Name'].str.contains("Tm_")].index, inplace=True)
fingerprints_df.reset_index(drop=True, inplace=True)

# Keep only the 'Adduct' and 'InChIKEY' columns
fingerprints_df = fingerprints_df[['Adduct', 'InChIKEY']]

# Keep only the first 14 characters of 'InChIKEY'
fingerprints_df['InChIKEY'] = fingerprints_df['InChIKEY'].str.slice(0, 14)

# Introduce the MREs
for i in range(0, len(data_files_list)):
    fingerprints_df[f'MRE_{data_files_list[i][12:-4]}'] = pred_vs_exp_df_list[i]['MRE']

print(fingerprints_df.shape)
print(fingerprints_df.head(10))

"""
import os
import zipfile

if os.path.exists('/home/grf/PycharmProjects/cmmrt/resources/50Mcompounds_split.zip/'):

    # Unzip the file
    with zipfile.ZipFile('/home/grf/PycharmProjects/cmmrt/resources/50Mcompounds_split.zip', 'r') as f:
        f.extractall('/home/grf/PycharmProjects/cmmrt/resources/Classification_files/')

remove 50M

if os.path.exists('/home/grf/PycharmProjects/cmmrt/resources/50Mcompounds_split.zip/'):
print(f"Files have been extracted")

"""