
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

file_paths = []
# Find all measurement files
distances = [5,10,15,20,25,30,37]

for i in distances:
    file_paths.append(f'measures_{i}.csv')

if not file_paths:
    print("No measurement CSV files found matching 'measures_*.csv'. Please ensure files are in the working directory.")
else:
    # Containers for data and statistics
    data_dict = {}
    stats = []

    # Read data and compute statistics
    for path in file_paths:
        # Extract mu value from filename

        # Read the measurements
        df = pd.read_csv(path, header=None)
        values = df.iloc[:, 0].values[1:] * 100

        # Compute mean (MLE) and variance (MLE)
        mean_est = np.mean(values)
        var_est = np.var(values) 

        data_dict[path] = values
        stats.append({'mu': path, 'mean': mean_est, 'variance': var_est})


    # Plot 1: PDF of all distributions
    plt.figure()
    # Define a common x-axis range
    all_values = np.concatenate(list(data_dict.values()))
    x_range = np.linspace(all_values.min(), all_values.max(), 1000)

    for path, values in data_dict.items():
        mean_est = np.mean(values)
        std_est = np.sqrt(np.var(values))
        pdf_vals = norm.pdf(x_range, mean_est, std_est)
        mu  = path.split("_")[-1].split(".")[0]
        plt.plot(x_range, pdf_vals, label=f'μ={mu}')
        print(f"For distance {mu} means estimate is {mean_est:.4f} and var is {np.var(values):.4f}")

    plt.title('Estimated Gaussian PDFs')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.savefig("pdfs.png")

    # Plot 2: Histograms of all distributions
    plt.figure()

    for path, values in data_dict.items():
        plt.hist(values, bins=30, density=True, alpha=0.5, label=f'μ={path.split("_")[-1].split(".")[0]}')

    plt.title('Histograms of All Distributions')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig("histo.png")

