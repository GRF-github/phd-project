import pandas as pd
import os
import matplotlib.pyplot as plt

data_files_list = os.listdir('/home/grf/PycharmProjects/cmmrt/results/scatter_plots')

for file in data_files_list:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(f'/home/grf/PycharmProjects/cmmrt/results/scatter_plots/{file}', header=None)

    # Rename the columns
    df.columns = ['exp', 'pred']

    # Create a scatter plot
    plt.figure()
    plt.scatter(df['exp'], df['pred'], s=2, c='blue', label='Data points')
    plt.xlabel('exp')
    plt.ylabel('pred')
    plt.title(f'Scatter Plot of {file[12:-4]}')
    plt.legend()
    plt.grid(True)
    plt.show()
