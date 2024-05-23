import pandas as pd

# Define the file path
file_path = ''

# Read the CSV file into a DataFrame
df = pd.read_csv('/home/grf/PycharmProjects/cmmrt/results/prueba_scatter.csv', header=None)

# Rename the columns
df.columns = ['exp', 'pred']

# Display the DataFrame
print(df.head())
