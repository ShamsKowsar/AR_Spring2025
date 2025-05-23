import math


import matplotlib.pyplot as plt

import pandas as pd


from matplotlib import ticker

df = pd.read_csv('rotation_data_60.csv')


cw = df[df['Clockwise'] == 1]
ccw = df[df['Clockwise'] == 0]


metrics = [
    'v_hat',
    'omega_hat',
    'gamma',
    'x0',
    'y0',
    'theta0',
    'x1',
    'y1',
    'theta1',
    'x*',
    'y*',
    'r*',
    'delta_theta',
]




for direction_label, df_dir in zip(['CW', 'CCW'], [cw, ccw]):
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
    fig.suptitle(f'Position and Orientation - {direction_label}', fontsize=16)

    axes[0, 0].hist(df_dir['x0'], bins=10, color='cornflowerblue', edgecolor='black')
    axes[0, 0].set_title('x0 (before)')
    axes[0, 1].hist(df_dir['x1'], bins=10, color='lightcoral', edgecolor='black')
    axes[0, 1].set_title('x1 (after)')

    axes[1, 0].hist(df_dir['y0'], bins=10, color='cornflowerblue', edgecolor='black')
    axes[1, 0].set_title('y0 (before)')
    axes[1, 1].hist(df_dir['y1'], bins=10, color='lightcoral', edgecolor='black')
    axes[1, 1].set_title('y1 (after)')

    axes[2, 0].hist(df_dir['theta0'], bins=10, color='cornflowerblue', edgecolor='black')
    axes[2, 0].set_title('theta0 (before)')
    axes[2, 1].hist(df_dir['theta1'], bins=10, color='lightcoral', edgecolor='black')
    axes[2, 1].set_title('theta1 (after)')

    for ax_row in axes:
        for ax in ax_row:
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
            ax.set_ylabel('Frequency')
            ax.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{direction_label}_x_y_theta_before_after.png')
    plt.show()




other_metrics = ['x*', 'y*', 'r*', 'delta_theta']
                        




NCOLS = 2
for direction_label, df_dir in zip(['CW', 'CCW'], [cw, ccw]):
    n = len(other_metrics)
    

    nrows = math.ceil(n / NCOLS)

    fig, axes = plt.subplots(nrows, NCOLS, figsize=(12, 4 * nrows), constrained_layout=True)
    fig.suptitle(f'Other Variables – {direction_label}', fontsize=16)
    axes = axes.flat

    for ax, metric in zip(axes, other_metrics):
        ax.hist(df_dir[metric], bins=10, edgecolor='black')
        ax.set_title(metric)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
        ax.set_ylabel('Frequency')
        ax.grid(True)

    for ax in list(axes)[len(other_metrics) :]:
        ax.axis('off')

    plt.savefig(f'{direction_label}_other_metrics.png')
    plt.show()




separate_metrics = ['v_hat', 'omega_hat', 'gamma']


for metric in separate_metrics:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'{metric} ', fontsize=14)

    for ax, direction_label, df_dir in zip(axes, ['CW', 'CCW'], [cw, ccw]):
        ax.hist(df_dir[metric], bins=10, color='mediumseagreen', edgecolor='black')
        ax.set_title(direction_label)
        ax.set_xlabel(metric)
        ax.set_ylabel('Frequency')
        ax.grid(True)
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{metric}_CW_vs_CCW.png')
    plt.show()