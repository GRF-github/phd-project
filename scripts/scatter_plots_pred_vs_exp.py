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
    pred_vs_exp_df.columns = ['exp', 'pred']

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
# Correct CCS value
def calculate_average_ccs(row):
    """
    Calculate the average Collision Cross Section (CCS) value from three different experiments.

    Parameters:
    - row (pandas.Series): A row from a DataFrame containing columns 'CCS1', 'CCS2', and 'CCS3' with CCS values.

    Returns:
    - float: The average CCS value rounded to two decimal places.
    """

    # Extract values from columns CCS1, CCS2, CCS3 for the current row
    ccs1 = row['CCS1']
    ccs2 = row['CCS2']
    ccs3 = row['CCS3']

    # Calculate the average and round to two decimals
    average_ccs = round((ccs1 + ccs2 + ccs3) / 3, 2)

    return average_ccs


fingerprints['correct_ccs_avg'] = fingerprints.apply(calculate_average_ccs, axis=1)

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